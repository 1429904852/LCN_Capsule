#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import os
import tensorflow as tf

from src.bert import optimization, modeling
from src.module.att_layer import mlp_attention_layer


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
        # output_layer = model.get_pooled_output()

        # hmtc
        output_layer = model.get_sequence_output()
        hidden_size = output_layer.shape[-1].value

        if self.__is_training:
            # I.e., 0.1 dropout
            # hmtc
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        
        with tf.name_scope("output"):
            # hmtc
            output_weights_1 = tf.get_variable(
                "output_weights_1", [self.__num_classes_1, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias_1 = tf.get_variable(
                "output_bias_1", [self.__num_classes_1], initializer=tf.zeros_initializer())
            
            # output_layer_1 = tf.reduce_mean(output_layer, 1)
            # output_layer_1 = tf.reshape(output_layer_1, [-1, hidden_size])
            
            attention_1 = mlp_attention_layer(output_layer, self.sequence_len, hidden_size, 0.00001, 0.1, 1)
            output_layer_1 = tf.reshape(tf.squeeze(tf.matmul(attention_1, output_layer)), [-1, hidden_size])
            output_layer_1 = tf.cast(output_layer_1, dtype=tf.float64)
            output_layer_1 = tf.nn.dropout(output_layer_1, keep_prob=0.5)
            output_layer_1 = tf.cast(output_layer_1, dtype=tf.float32)
            output_layer_1 = tf.reshape(output_layer_1, [-1, hidden_size])

            logits_1 = tf.matmul(output_layer_1, output_weights_1, transpose_b=True)
            logits_1 = tf.nn.bias_add(logits_1, output_bias_1)
            self.predictions_1 = tf.argmax(logits_1, axis=-1, name="predictions_1")
            # [-1, __num_classes_1]

            self.predict_1 = tf.nn.softmax(logits_1)
            self.predict_11 = tf.expand_dims(self.predict_1, 1)
            self.predict_11 = tf.tile(self.predict_11, [1, self.sequence_len, 1])
            output_layer_2_1 = tf.concat([output_layer, self.predict_11], 2)
            
            output_weights_2 = tf.get_variable(
                "output_weights_2", [self.__num_classes_2, hidden_size + self.__num_classes_1],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias_2 = tf.get_variable(
                "output_bias_2", [self.__num_classes_2], initializer=tf.zeros_initializer())
            
            # output_layer_2 = tf.reduce_mean(output_layer_2_1, 1)
            attention_2 = mlp_attention_layer(output_layer_2_1, self.sequence_len, hidden_size + self.__num_classes_1, 0.00001, 0.1, 2)
            output_layer_2 = tf.reshape(tf.squeeze(tf.matmul(attention_2, output_layer_2_1)), [-1, hidden_size + self.__num_classes_1])
            output_layer_2 = tf.cast(output_layer_2, dtype=tf.float64)
            output_layer_2 = tf.nn.dropout(output_layer_2, keep_prob=0.5)
            output_layer_2 = tf.cast(output_layer_2, dtype=tf.float32)
            output_layer_2 = tf.reshape(output_layer_2, [-1, hidden_size + self.__num_classes_1])

            logits_2 = tf.matmul(output_layer_2, output_weights_2, transpose_b=True)
            logits_2 = tf.nn.bias_add(logits_2, output_bias_2)
            self.predictions_2 = tf.argmax(logits_2, axis=-1, name="predictions_2")
            # [-1, __num_classes_2]

            self.predict_2 = tf.nn.softmax(logits_2)
            self.predict_22 = tf.expand_dims(self.predict_2, 1)
            self.predict_22 = tf.tile(self.predict_22, [1, self.sequence_len, 1])
            output_layer_3_1 = tf.concat([output_layer, self.predict_22], 2)
            
            output_weights_3 = tf.get_variable(
                "output_weights_3", [self.__num_classes_3, hidden_size + self.__num_classes_2],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias_3 = tf.get_variable(
                "output_bias_3", [self.__num_classes_3], initializer=tf.zeros_initializer())
            
            # output_layer_3 = tf.reduce_mean(output_layer_3_1, 1)
            attention_3 = mlp_attention_layer(output_layer_3_1, self.sequence_len, hidden_size + self.__num_classes_2, 0.00001, 0.1, 3)
            output_layer_3 = tf.reshape(tf.squeeze(tf.matmul(attention_3, output_layer_3_1)), [-1, hidden_size + self.__num_classes_2])
            output_layer_3 = tf.cast(output_layer_3, dtype=tf.float64)
            output_layer_3 = tf.nn.dropout(output_layer_3, keep_prob=0.5)
            output_layer_3 = tf.cast(output_layer_3, dtype=tf.float32)
            output_layer_3 = tf.reshape(output_layer_3, [-1, hidden_size + self.__num_classes_2])

            logits_3 = tf.matmul(output_layer_3, output_weights_3, transpose_b=True)
            logits_3 = tf.nn.bias_add(logits_3, output_bias_3)
            self.predictions_3 = tf.argmax(logits_3, axis=-1, name="predictions_3")

        if self.__is_training:
            with tf.name_scope("loss"):
                losses_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=self.label_ids_1)
                self.loss_1 = tf.reduce_mean(losses_1, name="loss_1")
                losses_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=self.label_ids_2)
                self.loss_2 = tf.reduce_mean(losses_2, name="loss_2")
                losses_3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=self.label_ids_3)
                self.loss_3 = tf.reduce_mean(losses_3, name="loss_3")
                self.loss = self.loss_1 + self.loss_2 + self.loss_3

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
        _, loss, predictions_1, predictions_2, predictions_3 = sess.run([self.train_op, self.loss, self.predictions_1,
                                         self.predictions_2, self.predictions_3], feed_dict=feed_dict)
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

        loss, predictions_1, predictions_2, predictions_3 = sess.run([self.loss, self.predictions_1, self.predictions_2, self.predictions_3], feed_dict=feed_dict)
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

        predict_1, predict_2, predict_3 = sess.run([self.predictions_1, self.predictions_2, self.predictions_3], feed_dict=feed_dict)

        return predict_1, predict_2, predict_3
