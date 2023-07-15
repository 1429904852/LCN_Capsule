#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof
from src.module.capsule_variant_routing import CapsLayer_variant_routing
from src.module.capsule import CapsLayer
import tensorflow as tf
from src.module.nn_layers import bi_dynamic_gru
from src.utils.data_helper import margin_loss


class multiCapsule_Label_Routing(object):
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
            self.x = tf.nn.dropout(self.x, keep_prob=self.keep_prob)
            self.x = tf.cast(self.x, dtype=tf.float32)
            self.hidden = bi_dynamic_gru(self.x, self.sen_len, self.hidden_size, 'word')

        with tf.name_scope('level_1_cap'):
            # [batch_size, sen_len, 2 * hidden, 1]
            self.hidden_level_1 = tf.expand_dims(self.hidden, -1)
            batch_size = tf.shape(self.hidden_level_1)[0]

            with tf.variable_scope('FeatCap_SemanCap_1'):
                SemanCap_1 = CapsLayer(batch_size=batch_size, num_outputs=self.sc_num, vec_len=self.sc_dim,
                                       iter_routing=self.iter_routing,
                                       with_routing=False, layer_type='CONV')
                self.caps1 = SemanCap_1(input=self.hidden_level_1, kernel_size=self.filter_size, stride=1,
                                        embedding_dim=2 * self.hidden_size)

            with tf.variable_scope('ASC_ClassCap_1'):
                ASC_ClassCap_1 = CapsLayer(batch_size=batch_size, num_outputs=self.n_class_1, vec_len=self.cc_dim,
                                           iter_routing=self.iter_routing,
                                           with_routing=True, layer_type='FC')
                self.ASC_caps_1 = ASC_ClassCap_1(self.caps1)

                self.ASC_sv_length_1 = tf.sqrt(tf.reduce_sum(tf.square(self.ASC_caps_1), axis=2, keepdims=True) + 1e-9)
                self.predict_1 = tf.reshape(self.ASC_sv_length_1, [batch_size, self.n_class_1])

        with tf.name_scope('level_2_cap'):
            # self.hidden_level_2 = tf.reshape(self.predict_1, [-1, self.n_class_1])
            self.hidden_level_2 = tf.reshape(self.level_1, [-1, self.n_class_1])
            self.hidden_level_2 = tf.expand_dims(self.hidden_level_2, 1)
            self.hidden_level_2 = tf.tile(self.hidden_level_2, [1, self.max_sen_len, 1])
            self.hidden_level_2 = tf.concat([self.hidden, self.hidden_level_2], 2)
            self.hidden_level_2 = tf.expand_dims(self.hidden_level_2, -1)
            batch_size = tf.shape(self.hidden_level_2)[0]

            # self.hidden_level_2 = tf.reshape(self.ASC_caps_1, [-1, self.n_class_1, self.cc_dim])
            # self.outputs_2 = tf.expand_dims(self.hidden_level_2, -1)

            # self.predict_11 = tf.expand_dims(self.predict_1, 1)
            self.level_11 = tf.expand_dims(self.level_1, 1)

            with tf.variable_scope('FeatCap_SemanCap_2'):
                SemanCap_2 = CapsLayer_variant_routing(label=self.level_11, batch_size=batch_size,
                                                       num_outputs=self.sc_num, vec_len=self.sc_dim,
                                                       iter_routing=self.iter_routing,
                                                       with_routing=False, layer_type='CONV')
                (self.caps2, self.label_gate_2) = SemanCap_2(input=self.hidden_level_2, kernel_size=self.filter_size,
                                                             stride=1,
                                                             embedding_dim=2 * self.hidden_size + self.n_class_1)

            with tf.variable_scope('ASC_ClassCap_2'):
                ASC_ClassCap_2 = CapsLayer_variant_routing(label=self.level_11, batch_size=batch_size,
                                                           num_outputs=self.n_class_2, vec_len=self.cc_dim,
                                                           iter_routing=self.iter_routing,
                                                           with_routing=True, layer_type='FC')
                # routing
                self.v_1 = tf.reshape(self.ASC_caps_1, [-1, self.n_class_1, self.cc_dim])
                self.v_1 = tf.expand_dims(self.v_1, -1)
                (self.ASC_caps_2, self.c_ij_2) = ASC_ClassCap_2(self.caps2, self.v_1)

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

            # self.predict_22 = tf.expand_dims(self.predict_2, 1)
            self.level_22 = tf.expand_dims(self.level_2, 1)

            with tf.variable_scope('FeatCap_SemanCap_3'):
                SemanCap_3 = CapsLayer_variant_routing(label=self.level_22, batch_size=batch_size,
                                                       num_outputs=self.sc_num, vec_len=self.sc_dim,
                                                       iter_routing=self.iter_routing,
                                                       with_routing=False, layer_type='CONV')
                (self.caps3, self.label_gate_3) = SemanCap_3(input=self.hidden_level_3, kernel_size=self.filter_size,
                                                             stride=1,
                                                             embedding_dim=2 * self.hidden_size + self.n_class_2)

            with tf.variable_scope('ASC_ClassCap_3'):
                ASC_ClassCap_3 = CapsLayer_variant_routing(label=self.level_22, batch_size=batch_size,
                                                           num_outputs=self.n_class_3, vec_len=self.cc_dim,
                                                           iter_routing=self.iter_routing,
                                                           with_routing=True, layer_type='FC')

                # routing
                self.v_2 = tf.reshape(self.ASC_caps_2, [-1, self.n_class_2, self.cc_dim])
                self.v_2 = tf.expand_dims(self.v_2, -1)
                (self.ASC_caps_3, self.c_ij_3) = ASC_ClassCap_3(self.caps3, self.v_2)

                self.ASC_sv_length_3 = tf.sqrt(tf.reduce_sum(tf.square(self.ASC_caps_3), axis=2, keepdims=True) + 1e-9)
                self.predict_3 = tf.reshape(self.ASC_sv_length_3, [batch_size, self.n_class_3])

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss_1 = margin_loss(self.level_1, self.predict_1)
            self.loss_2 = margin_loss(self.level_2, self.predict_2)
            self.loss_3 = margin_loss(self.level_3, self.predict_3)
            # self.loss = self.loss_1 + self.loss_2 + self.loss_3 + sum(reg_loss)

            w_convert_1 = tf.get_variable(name='w_convert_1', shape=[self.n_class_1, self.n_class_2],
                                          initializer=tf.random_normal_initializer(stddev=0.01))
            w_convert_2 = tf.get_variable(name='w_convert_2', shape=[self.n_class_2, self.n_class_3],
                                          initializer=tf.random_normal_initializer(stddev=0.01))

            self.predict1 = tf.matmul(self.predict_1, w_convert_1)
            self.margin_loss_1 = tf.reduce_mean(tf.maximum(self.predict1 - self.predict_2 + 0.1, 0.0))
            self.predict2 = tf.matmul(self.predict_2, w_convert_2)
            self.margin_loss_2 = tf.reduce_mean(tf.maximum(self.predict2 - self.predict_3 + 0.1, 0.0))

            self.loss = self.loss_1 + self.loss_2 + self.loss_3 + 0.1 * self.margin_loss_1 + 0.1 * self.margin_loss_2 + sum(reg_loss)
