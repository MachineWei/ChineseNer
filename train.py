# -*- coding: utf-8 -*

import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
# from torch.optim import AdamW

from transformers import WEIGHTS_NAME, AdamW
from transformers import ElectraConfig, ElectraTokenizer, ElectraModel
from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel, AlbertPreTrainedModel
from transformers import get_linear_schedule_with_warmup

from models.optimizater.lamb import Lamb
from models.electra_for_ner import ElectraCrfForNer
from models.bert_for_ner import BertCrfForNer
from models.albert_for_ner import AlBertCrfForNer

from datasets.make_token import collate_fn, load_and_cache_examples
from datasets.read_data import NerProcessor
from tools.progressbar import ProgressBar
from tools.common import seed_everything
from tools.common import init_logger, logger
from tools.train_argparse import get_argparse

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


MODEL_CLASSES = {
                'electra': (ElectraConfig, ElectraCrfForNer, ElectraTokenizer),
                'bert': (BertConfig, BertCrfForNer, BertTokenizer),
                'albert': (AlbertConfig, AlBertCrfForNer, BertTokenizer),
                }


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # 载入数据
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    # 总训练步数
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # 优化器
    if args.optimizer.lower() == "adamw":
        no_decay = ["bias", "LayerNorm.weight"]
        base_model_param_optimizer = list(model.BaseModel.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in base_model_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in base_model_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate}
        ]

        if "lstm" in args.model_type.lower():
             lstm_param_optimizer = list(model.Bilstm.named_parameters())
             optimizer_grouped_parameters.extend([{'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
                                                   'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                                                  {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)],
                                                   'weight_decay': 0.0,'lr': args.crf_learning_rate}])

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = Lamb(model.parameters())
    # 学习率
    args.warmup_steps = int(t_total * args.warmup_proportion)    # 学习率预热
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
            logger.info("using fp16 !!!")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # 多GPU训练 (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps,
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # 复现
    for _ in range(int(args.num_train_epochs)):
        logger.info(f"############### Epoch_{_} ###############")
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            if args.model_type != "distilbert":   # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0] 
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:  # 验证集
                    logger.info("\n global_step： %s", global_step)
                    logger.info("average tr_loss: %s", tr_loss/global_step)
                    evaluate(args, model, tokenizer)
                if args.save_steps > 0 and global_step % args.save_steps == 0:  # 保存模型
                    logger.info("global_step： %s 模型已保存！", global_step)
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval
    logger.info("***** Running evaluation %s *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    true_labels = []
    predict_labels = []
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            if args.model_type != "distilbert":   # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            batch_predict_labels = model.crf.decode(logits, inputs['attention_mask'])
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        batch_predict_labels = batch_predict_labels.squeeze(0).cpu().numpy().tolist()
        batch_true_labels = batch[3].squeeze(0).cpu().numpy().tolist()
        pbar(step)
        for index, input_length in enumerate(batch[4]):
            batch_true = [args.id2label.get(i) for i in batch_true_labels[index][:input_length]]
            batch_predict = [args.id2label.get(i) for i in batch_predict_labels[index][:input_length]]
            true_labels.append(batch_true)
            predict_labels.append(batch_predict)
    
    logger.info("\n")
    logger.info("average eval_loss: %s", str(eval_loss/nb_eval_steps))
    logger.info("accuary: %s", str(accuracy_score(true_labels, predict_labels)))
    logger.info("p: %s", str(precision_score(true_labels, predict_labels)))
    logger.info("r: %s", str(recall_score(true_labels, predict_labels)))
    logger.info("f1: %s", str(f1_score(true_labels, predict_labels)))
    logger.info("classification report: ")
    logger.info(str(classification_report(true_labels, predict_labels, mode='strict', scheme=IOB2)))


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []   # 全部测试结果
    error_results=[]   # 预测错误结果
    true_labels = []   # 真实标签
    predict_labels = []   # 预测标签
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.txt")
    error_predict_file = os.path.join(pred_output_dir, prefix, "Error_test_prediction.txt")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, torch.nn.DataParallel):  # 多GPU训练
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, 'input_lens': batch[4]}
            if args.model_type != "distilbert":   # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags  = tags.squeeze(0).cpu().numpy().tolist()
        labels = batch[3].squeeze(0).cpu().numpy().tolist()[1:-1]
        input_ids = inputs["input_ids"].squeeze(0).cpu().numpy().tolist()[1:-1]
        preds = tags[0][1:-1]    # [CLS]XXXX[SEP]

        sent = ""

        ifError=False
        for input_id,pre,lab in zip(input_ids,preds,labels):
            sent+=" ".join([tokenizer.ids_to_tokens[input_id],args.id2label[lab],args.id2label[pre]])+"\n"
            if args.id2label[lab]!=args.id2label[pre]:
                ifError=True
        sent+="\n"
        results.append(sent)
        if ifError:
            error_results.append(sent)
            ifError = False
        pbar(step)
        # 计算测试集 acc, recall, f1
        batch_true = [args.id2label.get(i) for i in preds]
        batch_predict = [args.id2label.get(i) for i in labels]
        assert len(batch_true) == len(batch_predict)
        true_labels.append(batch_true)
        predict_labels.append(batch_predict)

    logger.info("\n测试集结果统计：")
    logger.info("accuary: %s", str(accuracy_score(true_labels, predict_labels)))
    logger.info("p: %s", str(precision_score(true_labels, predict_labels)))
    logger.info("r: %s", str(recall_score(true_labels, predict_labels)))
    logger.info("f1: %s", str(f1_score(true_labels, predict_labels)))
    logger.info("classification report: ")
    logger.info(str(classification_report(true_labels, predict_labels, mode='strict', scheme=IOB2)))
    logger.info("\n")

    with open(output_predict_file, "w",encoding="utf-8") as writer:
        for record in results:
            writer.write(record)

    with open(error_predict_file, "w",encoding="utf-8") as writer:
        for record in error_results:
            writer.write(record)


def main():
    args = get_argparse().parse_args()
    # 打印参数
    print("="*20+" args "+"="*20)
    for para in args.__dict__:
        print(" " * (20 - len(para)), para, "=", args.__dict__[para])
    # 模型保存目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    # CUDA, GPU
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = torch.device('cpu')
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    processor = NerProcessor()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = model_class(config=config)
    # 基础预训练模型
    if args.model_type.lower() == "electra" and args.do_train:
        model.BaseModel = ElectraModel.from_pretrained(args.model_name_or_path)
        logger.info(f"Loading Electra from {args.model_name_or_path}...")
    if args.model_type.lower() == "bert" and args.do_train:
        model.BaseModel = BertModel.from_pretrained(args.model_name_or_path)
        logger.info(f"Loading Bert from {args.model_name_or_path}...")        
    if args.model_type.lower() == "albert" and args.do_train:
        model.BaseModel = AlbertModel.from_pretrained(args.model_name_or_path)
        logger.info(f"Loading AlBert from {args.model_name_or_path}...")                     
    # 继续训练
    if  args.do_train and args.continue_train:
        model = model_class.from_pretrained(args.continue_train_checkpoint, config=config)
        print(f"Loading model from {args.continue_train_checkpoint}")

    print(model)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # 训练
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, lr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, lr_loss)
        # 保存
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # 测试集
    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)


if __name__ == "__main__":
    main()





