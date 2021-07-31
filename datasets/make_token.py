
import logging
import os
import copy
import json
import torch
from torch.utils.data import TensorDataset
from .read_data import NerProcessor

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids, start_ids, end_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len
        self.start_ids = start_ids
        self.end_ids = end_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    # all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids, all_start_ids, all_end_ids = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_input_mask[:, :max_len]
    all_token_type_ids = all_segment_ids[:, :max_len]
    all_labels = all_label_ids[:, :max_len]
    all_start_ids = all_start_ids[:, :max_len]
    all_end_ids = all_end_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens, all_start_ids, all_end_ids


def convert_examples_to_features(examples,
                                label_list,
                                max_seq_length,
                                tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-100,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    span_labels = []
    for label in label_list:
        label = label.split('-')[-1]
        if label not in span_labels:
            span_labels.append(label)
    span_map = {label: i for i, label in enumerate(span_labels)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # tokens = tokenizer.tokenize(example.text_a)
        tokens = example.text_a
        label_ids = [label_map[x] if x in label_map else label_map["O"] for x in example.labels]

        entities = get_entities(example.labels)
        start_ids = [span_map['O']] * len(label_ids)
        end_ids = [span_map['O']] * len(label_ids)
        for entity in entities:
            start_ids[entity[1]] = span_map[entity[0]]
            end_ids[entity[-1]] = span_map[entity[0]]

        special_tokens_count = 2  #  [CLS] and [SEP]
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        # [SEP]
        tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        label_ids += [pad_token_label_id]
        start_ids += [pad_token_label_id]
        end_ids += [pad_token_label_id]
        # [CLS]
        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
            label_ids += [pad_token_label_id]
            start_ids += [pad_token_label_id]
            end_ids += [pad_token_label_id]           
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            label_ids = [pad_token_label_id] + label_ids
            start_ids = [pad_token_label_id] + start_ids
            end_ids = [pad_token_label_id] + end_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)  # 包含 CLS SEP
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            start_ids = ([pad_token_label_id] * padding_length) + start_ids
            end_ids = ([pad_token_label_id] * padding_length) + end_ids          
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            start_ids += [pad_token_label_id] * padding_length
            end_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, 
                                    input_mask=input_mask,
                                    input_len = input_len,
                                    segment_ids=segment_ids, 
                                    label_ids=label_ids,
                                    start_ids=start_ids,
                                    end_ids=end_ids))
    return features


def load_and_cache_examples(args, task, tokenizer, labels, pad_token_label_id, data_type='train'):
    processor = NerProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached-{}_{}_{}_{}'.format(
                                        data_type,
                                        list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
                                        str(task)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                    tokenizer=tokenizer,
                    label_list=label_list,
                    max_seq_length=args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length,
                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                    pad_on_left=bool(args.model_type in ['xlnet']),
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    # pad on the left for xlnet
                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                    pad_token_label_id=pad_token_label_id
                    )
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids, all_start_ids, all_end_ids)
    return dataset


def get_entities(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return set(chunks)

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def get_entities_span(starts, ends):
    if any(isinstance(s, list) for s in starts):
        starts = [item for sublist in starts for item in sublist + ['<SEP>']]
    if any(isinstance(s, list) for s in ends):
        ends = [item for sublist in ends for item in sublist + ['<SEP>']]
    chunks = []
    for start_index, start in enumerate(starts):
        if start in ['O', '<SEP>']:
            continue
        for end_index, end in enumerate(ends[start_index:]):
            if start == end:
                chunks.append((start, start_index, start_index + end_index))
                break
            elif end == '<SEP>':
                break
    return set(chunks)

def convert_span_to_bio(starts, ends):
    labels = []
    for start, end in zip(starts, ends):
        entities = get_entities_span(start, end)
        label = ['O'] * len(start)
        for entity in entities:
            label[entity[1]] = 'B-{}'.format(entity[0])
            label[entity[1] + 1: entity[2] + 1] = ['I-{}'.format(entity[0])] * (entity[2] - entity[1])
        labels.append(label)
    return labels