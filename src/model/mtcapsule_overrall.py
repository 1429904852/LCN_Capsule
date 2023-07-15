#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof
from src.module.capsule import *
import tensorflow as tf
from src.module.nn_layers import bi_dynamic_gru
from src.utils.data_helper import margin_loss

class multiCapsule_Overrall(object):
    def __init__(self, max_sen_len, n_class_1, n_class_2, n_class_3, embedding_document, embedding_dim,
                 hidden_size, sc_num, sc_dim, cc_dim, filter_size, iter_routing, random_base, l2_reg):
        self.max_sen_len = max_sen_len

        self.n_class_1 = n_class_1
        self.n_class_2 = n_class_2
        self.n_class_3 = n_class_3

        self.embedding_document = embedding_document
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.sc_num = sc_num
        self.sc_dim = sc_dim

        self.cc_dim = cc_dim
        self.filter_size = filter_size
        self.iter_routing = iter_routing

        self.random_base = random_base
        self.l2_reg = l2_reg

        with tf.name_scope('input'):
            self.input_x = tf.placeholder(tf.int32, [None, self.max_sen_len], name='input_x')
            self.sen_len = tf.placeholder(tf.int32, [None], name='input_sen_len')
            self.level_1 = tf.placeholder(tf.float32, [None, self.n_class_1], name='input_level_1')
            self.level_2 = tf.placeholder(tf.float32, [None, self.n_class_2], name='input_level_2')
            self.level_3 = tf.placeholder(tf.float32, [None, self.n_class_3], name='input_level_3')
            self.keep_prob = tf.placeholder(tf.float64, name='input_keep_prob')

        with tf.name_scope('word_cap'):
            self.x = tf.nn.embedding_lookup(self.embedding_document, self.input_x)
            
            # vocabulary_size = 250000
            # embeddings_matrix = tf.Variable(tf.random_normal([vocabulary_size, self.embedding_dim]))
            # self.x = tf.nn.embedding_lookup(embeddings_matrix, self.input_x)

            # self.x = tf.cast(self.x, dtype=tf.float64)
            self.x = tf.nn.dropout(self.x, keep_prob=self.keep_prob)
            self.x = tf.cast(self.x, dtype=tf.float32)

            self.hidden = bi_dynamic_gru(self.x, self.sen_len, self.hidden_size, 'word')

        with tf.name_scope('level_1_cap'):
            # [batch_size, sen_len, 2 * hidden, 1]
            self.hidden_level_1 = tf.expand_dims(self.hidden, -1)
            batch_size = tf.shape(self.hidden_level_1)[0]

            with tf.variable_scope('FeatCap_SemanCap_1'):
                SemanCap = CapsLayer(batch_size=batch_size, num_outputs=self.sc_num, vec_len=self.sc_dim,
                                     iter_routing=self.iter_routing,
                                     with_routing=False, layer_type='CONV')
                self.caps1 = SemanCap(input=self.hidden_level_1, kernel_size=self.filter_size, stride=1,
                                      embedding_dim=2 * self.hidden_size)

            with tf.variable_scope('ASC_ClassCap_1'):
                ASC_ClassCap = CapsLayer(batch_size=batch_size, num_outputs=self.n_class_1, vec_len=self.cc_dim,
                                         iter_routing=self.iter_routing,
                                         with_routing=True, layer_type='FC')
                self.ASC_caps_1 = ASC_ClassCap(self.caps1)

                self.ASC_sv_length_1 = tf.sqrt(tf.reduce_sum(tf.square(self.ASC_caps_1), axis=2, keepdims=True) + 1e-9)
                self.predict_1 = tf.reshape(self.ASC_sv_length_1, [batch_size, self.n_class_1])

        with tf.name_scope('level_2_cap'):
            self.hidden_level_2 = tf.reshape(self.level_1, [-1, self.n_class_1])
            # self.hidden_level_2 = tf.reshape(self.predict_1, [-1, self.n_class_1])
            self.hidden_level_2 = tf.expand_dims(self.hidden_level_2, 1)
            self.hidden_level_2 = tf.tile(self.hidden_level_2, [1, self.max_sen_len, 1])
            self.hidden_level_2 = tf.concat([self.hidden, self.hidden_level_2], 2)
            self.hidden_level_2 = tf.expand_dims(self.hidden_level_2, -1)
            batch_size = tf.shape(self.hidden_level_2)[0]

            # self.hidden_level_2 = tf.reshape(self.ASC_caps_1, [-1, self.n_class_1, self.cc_dim])
            # self.outputs_2 = tf.expand_dims(self.hidden_level_2, -1)

            with tf.variable_scope('FeatCap_SemanCap_2'):
                SemanCap = CapsLayer(batch_size=batch_size, num_outputs=self.sc_num, vec_len=self.sc_dim,
                                     iter_routing=self.iter_routing,
                                     with_routing=False, layer_type='CONV')
                self.caps2 = SemanCap(input=self.hidden_level_2, kernel_size=self.filter_size, stride=1,
                                      embedding_dim=2 * self.hidden_size + self.n_class_1)

            with tf.variable_scope('ASC_ClassCap_2'):
                ASC_ClassCap = CapsLayer(batch_size=batch_size, num_outputs=self.n_class_2, vec_len=self.cc_dim,
                                         iter_routing=self.iter_routing,
                                         with_routing=True, layer_type='FC')
                self.ASC_caps_2 = ASC_ClassCap(self.caps2)

                self.ASC_sv_length_2 = tf.sqrt(tf.reduce_sum(tf.square(self.ASC_caps_2), axis=2, keepdims=True) + 1e-9)
                self.predict_2 = tf.reshape(self.ASC_sv_length_2, [batch_size, self.n_class_2])

        with tf.name_scope('level_3_cap'):
            # self.hidden_level_3 = tf.reshape(self.predict_2, [-1, self.n_class_2])
            self.hidden_level_3 = tf.reshape(self.level_2, [-1, self.n_class_2])
            self.hidden_level_3 = tf.expand_dims(self.hidden_level_3, 1)
            self.hidden_level_3 = tf.tile(self.hidden_level_3, [1, self.max_sen_len, 1])
            self.hidden_level_3 = tf.concat([self.hidden, self.hidden_level_3], 2)
            self.hidden_level_3 = tf.expand_dims(self.hidden_level_3, -1)
            batch_size = tf.shape(self.hidden_level_3)[0]

            with tf.variable_scope('FeatCap_SemanCap_3'):
                SemanCap = CapsLayer(batch_size=batch_size, num_outputs=self.sc_num, vec_len=self.sc_dim,
                                     iter_routing=self.iter_routing,
                                     with_routing=False, layer_type='CONV')
                self.caps3 = SemanCap(input=self.hidden_level_3, kernel_size=self.filter_size, stride=1,
                                      embedding_dim=2 * self.hidden_size + self.n_class_2)

            with tf.variable_scope('ASC_ClassCap_3'):
                ASC_ClassCap = CapsLayer(batch_size=batch_size, num_outputs=self.n_class_3, vec_len=self.cc_dim,
                                         iter_routing=self.iter_routing,
                                         with_routing=True, layer_type='FC')
                self.ASC_caps_3 = ASC_ClassCap(self.caps3)

                self.ASC_sv_length_3 = tf.sqrt(tf.reduce_sum(tf.square(self.ASC_caps_3), axis=2, keepdims=True) + 1e-9)
                self.predict_3 = tf.reshape(self.ASC_sv_length_3, [batch_size, self.n_class_3])

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss_1 = margin_loss(self.level_1, self.predict_1)
            self.loss_2 = margin_loss(self.level_2, self.predict_2)
            self.loss_3 = margin_loss(self.level_3, self.predict_3)
            self.loss = self.loss_1 + self.loss_2 + self.loss_3 + sum(reg_loss)
