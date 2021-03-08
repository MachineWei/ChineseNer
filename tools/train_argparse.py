# -*- coding: utf-8 -*-

import argparse


def get_argparse():
     parser = argparse.ArgumentParser()
     # Required parameters

     parser.add_argument("--task_name", default="ner", type=str,
                         help="The name of the task to train selected in the list: ")
     ##数据集
     parser.add_argument("--data_dir", default="datasets/data", type=str,
                         help="The input data dir", )

     ## 预训练模型路径&名称
     parser.add_argument("--model_type", default="electra", type=str,
                         help="Model type selected in the list: [bert,albert,albert_lstm,electra]")

     parser.add_argument("--model_name_or_path", default="pre_trained_models/electra_zh_small", type=str,
                         help="Path to pre-trained model " )

     parser.add_argument("--config_name", default="pre_trained_models/electra_zh_small/config.json", type=str,
                         help="【配置文件】Pretrained config name or path if not the same as model_name")

     parser.add_argument("--tokenizer_name", default="pre_trained_models/electra_zh_small/vocab.txt", type=str,
                         help="【词表路径vocab.txt】Pretrained tokenizer name or path if not the same as model_name", )

     ## 优化器
     parser.add_argument("--optimizer", default="AdamW", type=str,
                         help="optimizer:AdamW/Lamb" )

     ## 模型保存及预测结果
     parser.add_argument("--output_dir", default="outputs/", type=str,
                         help="The output directory where the model predictions and checkpoints will be written.", )
     parser.add_argument("--logging_steps", type=int, default=1000,
                         help="Log every X updates steps.")
     parser.add_argument("--save_steps", type=int, default=1000, 
                         help="Save checkpoint every X updates steps.")
     # 训练参数
     parser.add_argument("--do_train", action="store_true", default=False,
                         help="是否进行训练")
     parser.add_argument("--do_eval", action="store_true", default=False,
                         help="是否进行验证")
     parser.add_argument("--do_predict", action="store_true", default=True,
                         help="是否进行预测")

     parser.add_argument("--num_train_epochs", default=4, type=float,
                         help="Total number of training epochs to perform.")     
     parser.add_argument("--n_gpu", default=1, type=int,
                         help="GPU个数")
     parser.add_argument("--per_gpu_train_batch_size", default=128, type=int,
                         help="Batch size per GPU/CPU for training.")
     parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                         help="Batch size per GPU/CPU for evaluation.")

     parser.add_argument("--train_max_seq_length", default=128, type=int,
                         help="训练集tokenization之后的序列最大长度，长截断，短padded.", )

     parser.add_argument("--eval_max_seq_length", default=128, type=int,
                         help="验证集tokenization之后的序列最大长度，长截断，短padded.", )

     parser.add_argument("--continue_train", default=False, type=bool,
                         help="是否从上一步保存开始训练", )

     parser.add_argument("--continue_train_checkpoint", default="outputs/electra/checkpoint-13100 acc: 0.8940 - recall: 0.8485 - f1: 0.8707", type=str,
                         help="从上一步保存开始训练", )

     parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                         help="Number of updates steps to accumulate before performing a backward/update pass.", )
     parser.add_argument("--learning_rate", default=5e-5, type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--crf_learning_rate", default=1e-5, type=float,
                         help="The initial learning rate for crf and linear layer.")
     parser.add_argument("--weight_decay", default=0.01, type=float,
                         help="Weight decay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                         help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     parser.add_argument("--max_steps", default=-1, type=int,
                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
     # 其他
     parser.add_argument("--warmup_proportion", default=0.1, type=float,
                         help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
     parser.add_argument("--no_cuda", action="store_true", default=True, 
                         help="Avoid using CUDA when available")
     parser.add_argument("--overwrite_output_dir", action="store_true", default=True,
                         help="Overwrite the content of the output directory")
     parser.add_argument("--overwrite_cache", action="store_true", default=False,
                         help="Overwrite the cached training and evaluation sets")
     parser.add_argument("--seed", type=int, default=42, 
                         help="random seed for initialization")
     parser.add_argument("--fp16", action="store_true",default=False,
                         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.貌似无法加速推理", )
     parser.add_argument("--fp16_opt_level", type=str, default="O1",
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html", )

     parser.add_argument('--fix', default=False, type=bool, 
                         help="是否使用后期修复")

     parser.add_argument('--markup', default='bio', type=str, choices=['bios', 'bio'],
                         help="字符标识方式（bio、bios）")

     parser.add_argument('--loss_type', default='ce', type=str,
                         choices=['lsr', 'focal', 'ce'])

     parser.add_argument("--cache_dir", default="", type=str,
                         help="缓存数据目录", )

     parser.add_argument("--evaluate_during_training", action="store_true", default="True",
                         help="每一轮训练结束之后是否进行验证集测评", )

     parser.add_argument("--do_lower_case", action="store_true",
                         help="Set this flag if you are using an uncased model.")

     # adversarial training
     parser.add_argument("--do_adv", action="store_true", default=False,
                         help="Whether to adversarial training.")
     parser.add_argument('--adv_epsilon', default=1.0, type=float,
                         help="Epsilon for adversarial.")
     parser.add_argument('--adv_name', default='word_embeddings', type=str,
                         help="name for adversarial layer.")

     parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
     parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
     return parser


