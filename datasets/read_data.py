# -*- coding: utf-8 -*
'''
修改两个分隔符line 54、line 102
自定义标签line 100
'''


import os
import copy
import json


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_text(self, input_file, delimiter='\t'):
        # delimiter: word与label之间的分隔符
        lines = []
        with open(input_file, 'r', encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n" or line == "end\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(delimiter)               # 注意文本分隔符
                    words.append(splits[0].strip().lower())         # 加入小写
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines


class NerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._read_text(os.path.join(data_dir, "train.txt"))
        return self._create_examples(examples, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = self._read_text(os.path.join(data_dir, "dev.txt"))
        return self._create_examples(examples, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = self._read_text(os.path.join(data_dir, "test.txt"))
        return self._create_examples(examples, "test")

    def get_labels(self):
        """See base class."""
        return ['O', "B-T", "I-T", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PER", "I-PER"]

    def _create_examples(self, lines, set_type, delimiter='-'):
        """
        Creates examples for the training and dev sets.
        delimiter : bio和tag之间的分隔符，- or _，例如“B-PER”,"B_PER"
        """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # 转 BIO
            labels = []
            for x in line['labels']:
                if 'M'+delimiter in x:
                    labels.append(x.replace('M'+delimiter, 'I'+delimiter))
                elif 'E'+delimiter in x:
                    labels.append(x.replace('E'+delimiter, 'I'+delimiter))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


