#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import json
import os
import tensorflow as tf
from src.bert_classifier.model_HATC import BertClassifier
# from src.bert_classifier.model_base import BertClassifier
# from src.bert_classifier.model_flat_base import BertClassifier
from src.bert import tokenization
import io


class Predictor(object):
    def __init__(self, config):
        self.model = None
        self.config = config

        self.output_path = config["output_path"]
        self.vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.label_to_index_1, self.label_to_index_2, self.label_to_index_3 = self.load_vocab()
        self.index_to_label_1 = {value: key for key, value in self.label_to_index_1.items()}
        self.index_to_label_2 = {value: key for key, value in self.label_to_index_2.items()}
        self.index_to_label_3 = {value: key for key, value in self.label_to_index_3.items()}
        self.word_vectors = None
        self.sequence_length = self.config["sequence_length"]

        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def load_vocab(self):
        # 将词汇-索引映射表加载出来
        with io.open(os.path.join(self.output_path, "label_to_index_1.json"), "r") as f:
            label_to_index_1 = json.load(f)
        with io.open(os.path.join(self.output_path, "label_to_index_2.json"), "r") as f:
            label_to_index_2 = json.load(f)
        with io.open(os.path.join(self.output_path, "label_to_index_3.json"), "r") as f:
            label_to_index_3 = json.load(f)
        return label_to_index_1, label_to_index_2, label_to_index_3

    def padding(self, input_ids, input_masks, segment_ids):
        """
        对序列进行补全
        :param input_id:
        :param input_mask:
        :param segment_id:
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids = [], [], []
        for input_id, input_mask, segment_id in zip(input_ids, input_masks, segment_ids):
            if len(input_id) < self.sequence_length:
                pad_input_ids.append(input_id + [0] * (self.sequence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (self.sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self.sequence_length - len(segment_id)))
            else:
                pad_input_ids.append(input_id[:self.sequence_length])
                pad_input_masks.append(input_mask[:self.sequence_length])
                pad_segment_ids.append(segment_id[:self.sequence_length])

        return pad_input_ids, pad_input_masks, pad_segment_ids

    def sentence_to_idx(self, inputs):
        """
        将分词后的句子转换成idx表示
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=True)

        input_ids = []
        input_masks = []
        segment_ids = []
        for text in inputs:
            text = tokenization.convert_to_unicode(text)
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            # input_mask = [1] * len(input_id)
            # segment_id = [0] * len(input_id)
            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))

        input_id, input_mask, segment_id = self.padding(input_ids, input_masks, segment_ids)

        return input_id, input_mask, segment_id

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
                inputs.append(item[0])
                labels_1.append(item[1])
                labels_2.append(item[2])
                labels_3.append(item[3])

        return inputs, labels_1, labels_2, labels_3

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))

    def create_model(self):
        """
                根据config文件选择对应的模型，并初始化
                :return:
                """
        self.model = BertClassifier(config=self.config, is_training=False)

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
        input_ids, input_masks, segment_ids, label_ids_1, label_ids_2, label_ids_3 = zip(*z)

        num_batches = len(input_ids) // self.config['batch_size']

        for i in range(num_batches):
            start = i * self.config['batch_size']
            end = start + self.config['batch_size']
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

    def predict(self, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        prediction_1, prediction_2, prediction_3 = self.model.infer(self.sess, dict(input_ids=batch["input_ids"],
                                                                                    input_masks=batch["input_masks"],
                                                                                    segment_ids=batch["segment_ids"],
                                                                                    label_ids_1=batch["label_ids_1"],
                                                                                    label_ids_2=batch["label_ids_2"],
                                                                                    label_ids_3=batch["label_ids_3"]))

        # label_1 = self.index_to_label_1[prediction_1]
        # label_2 = self.index_to_label_1[prediction_2]
        # label_3 = self.index_to_label_1[prediction_3]
        return prediction_1, prediction_2, prediction_3

        # def predict(self, text):
        #     """
        #     给定分词后的句子，预测其分类结果
        #     :param text:
        #     :return:
        #     """
        #     input_ids, input_masks, segment_ids = self.sentence_to_idx(text)
        #
        #     prediction_1, prediction_2, prediction_3 = self.model.infer(self.sess,
        #                                   dict(input_ids=input_ids,
        #                                        input_masks=input_masks,
        #                                        segment_ids=segment_ids)).tolist()[0]
        #     label_1 = self.index_to_label_1[prediction_1]
        #     label_2 = self.index_to_label_1[prediction_2]
        #     label_3 = self.index_to_label_1[prediction_3]
        #
        #     return label_1, label_2, label_3

    def predict_flat(self, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        prediction_3 = self.model.infer(self.sess, dict(input_ids=batch["input_ids"],
                                                        input_masks=batch["input_masks"],
                                                        segment_ids=batch["segment_ids"],
                                                        label_ids_1=batch["label_ids_1"],
                                                        label_ids_2=batch["label_ids_2"],
                                                        label_ids_3=batch["label_ids_3"]))

        return prediction_3
