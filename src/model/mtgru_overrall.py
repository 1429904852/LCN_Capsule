#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof
from src.module.capsule import *
import tensorflow as tf
from src.module.nn_layers import bi_dynamic_gru
from src.module.att_layer import self_attention, mlp_attention_layer


class multiGRU_OverRall(object):
    def __init__(self, max_sen_len, n_class_1, n_class_2, n_class_3, num_heads, embedding_document,
                 embedding_dim, hidden_size, random_base, l2_reg):
        self.max_sen_len = max_sen_len
        self.n_class_1 = n_class_1
        self.n_class_2 = n_class_2
        self.n_class_3 = n_class_3
        self.num_heads = num_heads
        self.embedding_document = embedding_document
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.random_base = random_base
        self.l2_reg = l2_reg

        with tf.name_scope('input'):
            self.input_x = tf.placeholder(tf.int32, [None, self.max_sen_len], name='input_x')
            self.sen_len = tf.placeholder(tf.int32, [None], name='input_sen_len')
            self.level_1 = tf.placeholder(tf.float32, [None, self.n_class_1], name='input_level_1')
            self.level_2 = tf.placeholder(tf.float32, [None, self.n_class_2], name='input_level_2')
            self.level_3 = tf.placeholder(tf.float32, [None, self.n_class_3], name='input_level_3')
            self.keep_prob = tf.placeholder(tf.float32, name='input_keep_prob')

        with tf.name_scope('weights'):
            self.weights = {
                'w_1': tf.Variable(
                    tf.random_uniform([2 * self.hidden_size + self.n_class_1, 2 * self.hidden_size], -self.random_base,
                                      self.random_base)),
                'w_2': tf.Variable(
                    tf.random_uniform([2 * self.hidden_size + self.n_class_2, 2 * self.hidden_size], -self.random_base,
                                      self.random_base)),
                'w_level_1': tf.Variable(
                    tf.random_uniform([self.n_class_1, 1], -self.random_base,
                                      self.random_base)),
                'w_head_1': tf.Variable(
                    tf.random_uniform([2 * self.hidden_size, 1], -self.random_base,
                                      self.random_base)),
                'w_level_2': tf.Variable(
                    tf.random_uniform([self.n_class_2, 1], -self.random_base,
                                      self.random_base)),
                'w_head_2': tf.Variable(
                    tf.random_uniform([2 * self.hidden_size, 1], -self.random_base,
                                      self.random_base)),
                'w_predict_1': tf.Variable(
                    tf.random_uniform([self.n_class_1, self.n_class_2], -self.random_base,
                                      self.random_base)),
                'w_predict_2': tf.Variable(
                    tf.random_uniform([self.n_class_2, self.n_class_3], -self.random_base,
                                      self.random_base)),
                'softmax_1': tf.Variable(
                    tf.random_uniform([2 * self.hidden_size, self.n_class_1], -self.random_base, self.random_base)),
                'softmax_2': tf.Variable(
                    tf.random_uniform([2 * self.hidden_size, self.n_class_2], -self.random_base, self.random_base)),
                'softmax_3': tf.Variable(
                    tf.random_uniform([2 * self.hidden_size, self.n_class_3], -self.random_base, self.random_base))
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax_1': tf.Variable(tf.random_uniform([self.n_class_1], -self.random_base, self.random_base)),
                'softmax_2': tf.Variable(tf.random_uniform([self.n_class_2], -self.random_base, self.random_base)),
                'softmax_3': tf.Variable(tf.random_uniform([self.n_class_3], -self.random_base, self.random_base))
            }

        with tf.name_scope('encoder'):
            self.x = tf.nn.embedding_lookup(self.embedding_document, self.input_x)

            # vocabulary_size = 250000
            # embeddings_matrix = tf.Variable(tf.random_normal([vocabulary_size, self.embedding_dim]))
            # self.x = tf.nn.embedding_lookup(embeddings_matrix, self.input_x)

            # self.x = tf.cast(self.x, dtype=tf.float32)
            # self.x = tf.nn.dropout(self.x, keep_prob=self.keep_prob)
            self.x = tf.cast(self.x, dtype=tf.float32)
            self.hidden = bi_dynamic_gru(self.x, self.sen_len, self.hidden_size, 'word')

            # level_1
            self.multi_head = self_attention(self.hidden, 2 * self.hidden_size, self.num_heads, 'multi-head-1')
            self.attention_1 = mlp_attention_layer(self.multi_head, self.sen_len, 2 * self.hidden_size, self.l2_reg, self.random_base, 1)
            self.output_1 = tf.reshape(tf.squeeze(tf.matmul(self.attention_1, self.multi_head)), [-1, 2 * self.hidden_size])
            # self.output_1 = tf.cast(self.output_1, dtype=tf.float64)
            self.output_1 = tf.nn.dropout(self.output_1, keep_prob=self.keep_prob)
            # self.output_1 = tf.cast(self.output_1, dtype=tf.float32)
            self.predict_1 = tf.matmul(self.output_1, self.weights['softmax_1']) + self.biases['softmax_1']
            self.predict_1_1 = tf.reshape(self.predict_1, [-1, self.n_class_1])
            self.predict_1 = tf.nn.softmax(self.predict_1_1)

            # level_2
            self.predict_11 = tf.expand_dims(self.predict_1, 1)
            self.predict_11 = tf.tile(self.predict_11, [1, self.max_sen_len, 1])

            # label injection
            self.predict_11 = tf.reshape(self.predict_11, [-1, self.n_class_1])
            self.multi_head_1 = tf.reshape(self.multi_head, [-1, 2 * self.hidden_size])
            gate_1 = tf.nn.sigmoid(tf.matmul(self.predict_11, self.weights['w_level_1']) + tf.matmul(self.multi_head_1, self.weights['w_head_1']))
            gate_1 = tf.reshape(gate_1, [-1, self.max_sen_len])
            gate_1 = tf.expand_dims(gate_1, -1)
            self.hidden_2 = self.multi_head * gate_1
            self.hidden_2 = tf.reshape(self.hidden_2, [-1, self.max_sen_len, 2 * self.hidden_size])

            # label concat
            # self.hidden_2 = tf.concat([self.multi_head, self.predict_11], 2)
            # self.hidden_2 = tf.reshape(self.hidden_2, [-1, 2 * self.hidden_size + self.n_class_1])
            # self.hidden_2 = tf.reshape(tf.matmul(self.hidden_2, self.weights['w_1']), [-1, self.max_sen_len, 2 * self.hidden_size])

            self.multi_head_2 = self_attention(self.hidden_2, 2 * self.hidden_size, self.num_heads, 'multi-head-2')
            # self.multi_head_2 = self_attention(self.multi_head, 2 * self.hidden_size, self.num_heads, 'multi-head-2')
            self.attention_2 = mlp_attention_layer(self.multi_head_2, self.sen_len, 2 * self.hidden_size, self.l2_reg, self.random_base, 2)
            self.output_2 = tf.reshape(tf.squeeze(tf.matmul(self.attention_2, self.multi_head_2)), [-1, 2 * self.hidden_size])

            # self.output_2 = tf.cast(self.output_2, dtype=tf.float64)
            self.output_2 = tf.nn.dropout(self.output_2, keep_prob=self.keep_prob)
            # self.output_2 = tf.cast(self.output_2, dtype=tf.float32)

            self.predict_2 = tf.matmul(self.output_2, self.weights['softmax_2']) + self.biases['softmax_2']
            self.predict_2_2 = tf.reshape(self.predict_2, [-1, self.n_class_2])
            
            # label re-routing
            # self.predict_1_2 = tf.reshape(tf.matmul(self.predict_1_1, self.weights['w_predict_1']), [-1, self.n_class_2])
            # self.predict_2_2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.predict_2_2)))
            # self.predict_1_2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.predict_1_2)))
            # tensor1_tensor2 = tf.reduce_sum(tf.multiply(self.predict_2_2, self.predict_1_2))
            # cosin_1 = tensor1_tensor2 / (self.predict_2_2_norm * self.predict_1_2_norm)
            # self.predict_2_2 = self.predict_2_2 + self.predict_1_2 * cosin_1

            self.predict_2 = tf.nn.softmax(self.predict_2_2)

            # level_3
            self.predict_22 = tf.expand_dims(self.predict_2, 1)
            self.predict_22 = tf.tile(self.predict_22, [1, self.max_sen_len, 1])

            # label injection
            self.predict_22 = tf.reshape(self.predict_22, [-1, self.n_class_2])
            self.multi_head_2 = tf.reshape(self.multi_head, [-1, 2 * self.hidden_size])
            gate_2 = tf.nn.sigmoid(tf.matmul(self.predict_22, self.weights['w_level_2']) + tf.matmul(self.multi_head_2, self.weights['w_head_2']))
            gate_2 = tf.reshape(gate_2, [-1, self.max_sen_len])
            gate_2 = tf.expand_dims(gate_2, -1)
            self.hidden_3 = self.multi_head * gate_2
            self.hidden_3 = tf.reshape(self.hidden_3, [-1, self.max_sen_len, 2 * self.hidden_size])

            # label concat
            # self.hidden_3 = tf.concat([self.multi_head_2, self.predict_22], 2)
            # self.hidden_3 = tf.reshape(self.hidden_3, [-1, 2 * self.hidden_size + self.n_class_2])
            # self.hidden_3 = tf.reshape(tf.matmul(self.hidden_3, self.weights['w_2']), [-1, self.max_sen_len, 2 * self.hidden_size])

            self.multi_head_3 = self_attention(self.hidden_3, 2 * self.hidden_size, self.num_heads, 'multi-head-3')
            # self.multi_head_3 = self_attention(self.multi_head, 2 * self.hidden_size, self.num_heads, 'multi-head-3')
            self.attention_3 = mlp_attention_layer(self.multi_head_3, self.sen_len, 2 * self.hidden_size, self.l2_reg, self.random_base, 3)
            self.output_3 = tf.reshape(tf.squeeze(tf.matmul(self.attention_3, self.multi_head_3)), [-1, 2 * self.hidden_size])

        with tf.name_scope('loss'):
            self.loss_1 = - tf.reduce_mean(self.level_1 * tf.log(self.predict_1))
            self.loss_2 = - tf.reduce_mean(self.level_2 * tf.log(self.predict_2))

            # self.output_3 = tf.cast(self.output_3, dtype=tf.float64)
            self.output_3 = tf.nn.dropout(self.output_3, keep_prob=self.keep_prob)
            
            # self.output_3 = tf.cast(self.output_3, dtype=tf.float32)

            # self.output_3 = tf.reshape(self.output_3, [-1, 2 * self.hidden_size])
            self.predict_3 = tf.matmul(self.output_3, self.weights['softmax_3']) + self.biases['softmax_3']
            self.predict_3_3 = tf.reshape(self.predict_3, [-1, self.n_class_3])

            # label re-routing
            # self.predict_2_3 = tf.reshape(tf.matmul(self.predict_2_2, self.weights['w_predict_2']), [-1, self.n_class_3])
            # self.predict_3_3_norm = tf.sqrt(tf.reduce_sum(tf.square(self.predict_3_3)))
            # self.predict_2_3_norm = tf.sqrt(tf.reduce_sum(tf.square(self.predict_2_3)))
            # tensor2_tensor3 = tf.reduce_sum(tf.multiply(self.predict_3_3, self.predict_2_3))
            # cosin_2 = tensor2_tensor3 / (self.predict_3_3_norm * self.predict_2_3_norm)
            # self.predict_3_3 = self.predict_3_3 + self.predict_2_3 * cosin_2

            self.predict_3 = tf.nn.softmax(self.predict_3_3)
            self.loss_3 = - tf.reduce_mean(self.level_3 * tf.log(self.predict_3))

            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = self.loss_1 + self.loss_2 + self.loss_3 + sum(reg_loss)
