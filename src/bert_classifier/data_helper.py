#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import json
import os
import random
import io
import re
from src.bert import tokenization


class TrainData(object):
    def __init__(self, config):

        self.__vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.__output_path = config["output_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)
        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长
        self._batch_size = config["batch_size"]

    @staticmethod
    def read_data(file_path):
        """
        读取数据
        :param file_path:
        :return: 返回分词后的文本内容和标签，inputs = [], labels = []
        """
        inputs = []
        labels_1 = []
        labels_2 = []
        labels_3 = []
        with io.open(file_path, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                item = line.strip().split("\t")
                # inputs.append(TrainData.text_cleaner(item[0]))
                inputs.append(item[0])
                labels_1.append(item[1])
                labels_2.append(item[2])
                labels_3.append(item[3])

        return inputs, labels_1, labels_2, labels_3

    @staticmethod
    def text_cleaner(text):
        """
        cleaning spaces, html tags, etc
        parameters: (string) text input to clean
        return: (string) clean_text
        """
        text = text.replace(".", "")
        text = text.replace("[", " ")
        text = text.replace(",", " ")
        text = text.replace("]", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("\"", "")
        text = text.replace("-", "")
        text = text.replace("=", "")
        rules = [
            {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
            {r'\s+': u' '},  # replace consecutive spaces
            {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
            {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
            {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
            {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
            {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
            {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
            {r'^\s+': u''}  # remove spaces at the beginning
        ]
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
        clean_text = text.lower()
        return clean_text

    def trans_to_index(self, inputs):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []
        for text in inputs:
            text = tokenization.convert_to_unicode(text)
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))

        return input_ids, input_masks, segment_ids

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        labels_idx = [label_to_index[label] for label in labels]
        return labels_idx

    def padding(self, input_ids, input_masks, segment_ids):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids = [], [], []
        for input_id, input_mask, segment_id in zip(input_ids, input_masks, segment_ids):
            if len(input_id) < self._sequence_length:
                pad_input_ids.append(input_id + [0] * (self._sequence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (self._sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self._sequence_length - len(segment_id)))
            else:
                pad_input_ids.append(input_id[:self._sequence_length])
                pad_input_masks.append(input_mask[:self._sequence_length])
                pad_segment_ids.append(segment_id[:self._sequence_length])

        return pad_input_ids, pad_input_masks, pad_segment_ids

    def gen_data(self, file_path, is_training=True):
        """
        生成数据
        :param file_path:
        :param is_training:
        :return:
        """

        # 1，读取原始数据
        inputs, labels_1, labels_2, labels_3 = self.read_data(file_path)
        print("read finished")

        if is_training:
            uni_label_1 = list(set(labels_1))
            label_to_index_1 = dict(zip(uni_label_1, list(range(len(uni_label_1)))))
            with io.open(os.path.join(self.__output_path, "label_to_index_1.json"), "w", encoding="utf-8") as fw:
                fw.write(str(json.dumps(label_to_index_1, ensure_ascii=False)))
                # json.dump(label_to_index_1, fw, indent=0, ensure_ascii=False)

            uni_label_2 = list(set(labels_2))
            label_to_index_2 = dict(zip(uni_label_2, list(range(len(uni_label_2)))))
            with io.open(os.path.join(self.__output_path, "label_to_index_2.json"), "w", encoding="utf-8") as fw:
                fw.write(str(json.dumps(label_to_index_2, ensure_ascii=False)))
                # json.dump(label_to_index_2, fw, indent=0, ensure_ascii=False)

            uni_label_3 = list(set(labels_3))
            label_to_index_3 = dict(zip(uni_label_3, list(range(len(uni_label_3)))))
            with io.open(os.path.join(self.__output_path, "label_to_index_3.json"), "w", encoding="utf-8") as fw:
                fw.write(str(json.dumps(label_to_index_3, ensure_ascii=False)))
                # json.dump(label_to_index_3, fw, indent=0, ensure_ascii=False)

        else:
            with io.open(os.path.join(self.__output_path, "label_to_index_1.json"), "r", encoding="utf-8") as fr:
                label_to_index_1 = json.load(fr)
            with io.open(os.path.join(self.__output_path, "label_to_index_2.json"), "r", encoding="utf-8") as fr:
                label_to_index_2 = json.load(fr)
            with io.open(os.path.join(self.__output_path, "label_to_index_3.json"), "r", encoding="utf-8") as fr:
                label_to_index_3 = json.load(fr)

        # 2，输入转索引
        inputs_ids, input_masks, segment_ids = self.trans_to_index(inputs)
        print("index transform finished")

        inputs_ids, input_masks, segment_ids = self.padding(inputs_ids, input_masks, segment_ids)

        # 3，标签转索引
        labels_ids_1 = self.trans_label_to_index(labels_1, label_to_index_1)
        labels_ids_2 = self.trans_label_to_index(labels_2, label_to_index_2)
        labels_ids_3 = self.trans_label_to_index(labels_3, label_to_index_3)
        print("label index transform finished")

        for i in range(5):
            print("line {}: *****************************************".format(i))
            print("input: ", inputs[i])
            print("input_id: ", inputs_ids[i])
            print("input_mask: ", input_masks[i])
            print("segment_id: ", segment_ids[i])
            print("label_id_1: ", labels_ids_1[i])
            print("label_id_2: ", labels_ids_2[i])
            print("label_id_3: ", labels_ids_3[i])

        return inputs_ids, input_masks, segment_ids, labels_ids_1, label_to_index_1, labels_ids_2, label_to_index_2, \
               labels_ids_3, label_to_index_3

    def next_batch(self, input_ids, input_masks, segment_ids, label_ids_1, label_ids_2, label_ids_3):
        """
        生成batch数据
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids:
        :return:
        """
        z = list(zip(input_ids, input_masks, segment_ids, label_ids_1, label_ids_2, label_ids_3))
        random.shuffle(z)
        input_ids, input_masks, segment_ids, label_ids_1, label_ids_2, label_ids_3 = zip(*z)

        num_batches = len(input_ids) // self._batch_size

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size
            batch_input_ids = input_ids[start: end]
            batch_input_masks = input_masks[start: end]
            batch_segment_ids = segment_ids[start: end]
            batch_label_ids_1 = label_ids_1[start: end]
            batch_label_ids_2 = label_ids_2[start: end]
            batch_label_ids_3 = label_ids_3[start: end]

            yield dict(input_ids=batch_input_ids,
                       input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids,
                       label_ids_1=batch_label_ids_1,
                       label_ids_2=batch_label_ids_2,
                       label_ids_3=batch_label_ids_3)
