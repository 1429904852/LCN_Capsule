#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import os
import tensorflow as tf

from src.bert import optimization, modeling


class BertClassifier(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(config["bert_model_path"], "bert_config.json")
        self.__num_classes_1 = config["num_classes_1"]
        self.__num_classes_2 = config["num_classes_2"]
        self.__num_classes_3 = config["num_classes_3"]
        self.__learning_rate = config["learning_rate"]
        self.sequence_len = config["sequence_length"]
        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
        self.label_ids_1 = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids_1")
        self.label_ids_2 = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids_2")
        self.label_ids_3 = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids_3")

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
        output_layer = model.get_pooled_output()

        # hmtc
        hidden_size = output_layer.shape[-1].value

        if self.__is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        with tf.name_scope("output"):
            # hmtc
            output_weights_3 = tf.get_variable(
                "output_weights_3", [self.__num_classes_3, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias_3 = tf.get_variable(
                "output_bias_3", [self.__num_classes_3], initializer=tf.zeros_initializer())
            logits_3 = tf.matmul(output_layer, output_weights_3, transpose_b=True)
            logits_3 = tf.nn.bias_add(logits_3, output_bias_3)
            self.predictions_3 = tf.argmax(logits_3, axis=-1, name="predictions_3")

        if self.__is_training:
            with tf.name_scope("loss"):
                losses_3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=self.label_ids_3)
                self.loss = tf.reduce_mean(losses_3, name="loss_3")

            with tf.name_scope('train_op'):
                self.train_op = optimization.create_optimizer(
                    self.loss, self.__learning_rate, self.__num_train_step, self.__num_warmup_step, use_tpu=False)

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
        _, loss, predictions_3 = sess.run([self.train_op, self.loss, self.predictions_3], feed_dict=feed_dict)
        return loss, predictions_3

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

        loss, predictions_3 = sess.run([self.loss, self.predictions_3], feed_dict=feed_dict)
        return loss, predictions_3

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

        predict_3 = sess.run(self.predictions_3, feed_dict=feed_dict)

        return predict_3
