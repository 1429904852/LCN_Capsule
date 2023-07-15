#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import os
import tensorflow as tf

from src.bert import optimization, modeling
from src.bert_classifier.capsule_label_routing import Capsule_Label_Routing


class BertClassifier(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(config["bert_model_path"], "bert_config.json")
        self.__num_classes_1 = config["num_classes_1"]
        self.__num_classes_2 = config["num_classes_2"]
        self.__num_classes_3 = config["num_classes_3"]
        self.__learning_rate = config["learning_rate"]
        self.sequence_len = config["sequence_length"]

        self.hidden_size = config["hidden_size"]
        self.sc_num = config["sc_num"]
        self.sc_dim = config["sc_dim"]
        self.cc_dim = config["cc_dim"]
        self.filter_size = config["filter_size"]
        self.iter_routing = config["iter_routing"]

        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
        self.label_ids_1 = tf.placeholder(dtype=tf.float32, shape=[None], name="label_ids_1")
        self.label_ids_2 = tf.placeholder(dtype=tf.float32, shape=[None], name="label_ids_2")
        self.label_ids_3 = tf.placeholder(dtype=tf.float32, shape=[None], name="label_ids_3")

        self.built_model()
        self.init_saver()

    def built_model(self):
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)
        model = modeling.BertModel(config=bert_config,
                                   is_training=self.__is_training,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_masks,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=False)
        # output_layer = model.get_pooled_output()

        # [-1, max_len, hidden_size]
        output_layer = model.get_sequence_output()
        hidden_size = output_layer.shape[-1].value

        if self.__is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        self.capsule_model = Capsule_Label_Routing(embedding=output_layer,
                                                   max_sen_len=self.sequence_len,
                                                   n_class_1=self.__num_classes_1,
                                                   n_class_2=self.__num_classes_2,
                                                   n_class_3=self.__num_classes_3,
                                                   hidden_size=hidden_size,
                                                   sc_num=self.sc_num,
                                                   sc_dim=self.sc_dim,
                                                   cc_dim=self.cc_dim,
                                                   filter_size=self.filter_size,
                                                   iter_routing=self.iter_routing,
                                                   level_1=self.label_ids_1,
                                                   level_2=self.label_ids_2,
                                                   level_3=self.label_ids_3)

        if self.__is_training:
            with tf.name_scope('train_op'):
                self.train_op = optimization.create_optimizer(
                    self.capsule_model.loss, self.__learning_rate, self.__num_train_step, self.__num_warmup_step,
                    use_tpu=False)

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """

        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids_1: batch["label_ids_1"],
                     self.label_ids_2: batch["label_ids_2"],
                     self.label_ids_3: batch["label_ids_3"]}

        # 训练模型
        _, loss, predictions_1, predictions_2, predictions_3 = sess.run([self.train_op, self.capsule_model.loss,
                                                                         self.capsule_model.predictions_1,
                                                                         self.capsule_model.predictions_2,
                                                                         self.capsule_model.predictions_3],
                                                                        feed_dict=feed_dict)
        return loss, predictions_1, predictions_2, predictions_3

    def eval(self, sess, batch):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids_1: batch["label_ids_1"],
                     self.label_ids_2: batch["label_ids_2"],
                     self.label_ids_3: batch["label_ids_3"]
                     }

        loss, predictions_1, predictions_2, predictions_3 = sess.run(
            [self.capsule_model.loss, self.capsule_model.predictions_1, self.capsule_model.predictions_2,
             self.capsule_model.predictions_3], feed_dict=feed_dict)
        return loss, predictions_1, predictions_2, predictions_3

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids_1: batch["label_ids_1"],
                     self.label_ids_2: batch["label_ids_2"],
                     self.label_ids_3: batch["label_ids_3"]}

        predict_1, predict_2, predict_3 = sess.run([self.capsule_model.predictions_1, self.capsule_model.predictions_2,
                                                    self.capsule_model.predictions_3],
                                                   feed_dict=feed_dict)

        return predict_1, predict_2, predict_3
