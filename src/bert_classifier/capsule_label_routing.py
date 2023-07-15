#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof
from src.module.capsule_variant_routing import CapsLayer_variant_routing
from src.module.capsule import CapsLayer
import tensorflow as tf
from src.utils.data_helper import margin_loss


class Capsule_Label_Routing(object):
    def __init__(self, embedding, max_sen_len, n_class_1, n_class_2, n_class_3, hidden_size, sc_num, sc_dim, cc_dim,
                 filter_size, iter_routing, level_1, level_2, level_3):
        self.max_sen_len = max_sen_len

        self.n_class_1 = n_class_1
        self.n_class_2 = n_class_2
        self.n_class_3 = n_class_3

        self.embedding = embedding
        self.hidden_size = hidden_size

        self.sc_num = sc_num
        self.sc_dim = sc_dim

        self.cc_dim = cc_dim
        self.filter_size = filter_size
        self.iter_routing = iter_routing

        self.level_1 = level_1
        self.level_2 = level_2
        self.level_3 = level_3

        with tf.name_scope('word_cap'):
            self.x = tf.cast(self.embedding, dtype=tf.float32)
            self.hidden = tf.reshape(self.x, [-1, self.max_sen_len, self.hidden_size])

            # (outputs_fw_q, outputs_bw_q), state = tf.nn.bidirectional_dynamic_rnn(
            #     cell_fw=tf.nn.rnn_cell.GRUCell(self.hidden_size),
            #     cell_bw=tf.nn.rnn_cell.GRUCell(self.hidden_size),
            #     inputs=self.x,
            #     dtype=tf.float32,
            #     scope='word'
            # )
            # self.hidden = tf.concat([outputs_fw_q, outputs_bw_q], -1)
            # self.hidden = tf.reshape(self.hidden, [-1, self.max_sen_len, 2 * self.hidden_size])
        with tf.name_scope('level_1_cap'):
            # [batch_size, sen_len, 2 * hidden, 1]
            self.hidden_level_1 = tf.expand_dims(self.hidden, -1)
            batch_size = tf.shape(self.hidden_level_1)[0]

            with tf.variable_scope('FeatCap_SemanCap_1'):
                SemanCap_1 = CapsLayer(batch_size=batch_size, num_outputs=self.sc_num, vec_len=self.sc_dim,
                                       iter_routing=self.iter_routing,
                                       with_routing=False, layer_type='CONV')
                # self.caps1 = SemanCap_1(input=self.hidden_level_1, kernel_size=self.filter_size, stride=1,
                #                         embedding_dim=2 * self.hidden_size)
                self.caps1 = SemanCap_1(input=self.hidden_level_1, kernel_size=self.filter_size, stride=1,
                                        embedding_dim=self.hidden_size)

            with tf.variable_scope('ASC_ClassCap_1'):
                ASC_ClassCap_1 = CapsLayer(batch_size=batch_size, num_outputs=self.n_class_1, vec_len=self.cc_dim,
                                           iter_routing=self.iter_routing,
                                           with_routing=True, layer_type='FC')
                self.ASC_caps_1 = ASC_ClassCap_1(self.caps1)

                self.ASC_sv_length_1 = tf.sqrt(tf.reduce_sum(tf.square(self.ASC_caps_1), axis=2, keepdims=True) + 1e-9)
                self.predict_1 = tf.reshape(self.ASC_sv_length_1, [batch_size, self.n_class_1])

        with tf.name_scope('level_2_cap'):
            self.hidden_level_2 = tf.reshape(self.predict_1, [-1, self.n_class_1])

            self.level_1 = tf.cast(self.level_1, dtype=tf.int32)
            self.level_1 = tf.one_hot(self.level_1, depth=self.n_class_1)
            # self.hidden_level_2 = tf.reshape(self.level_1, [-1, self.n_class_1])

            self.hidden_level_2 = tf.expand_dims(self.hidden_level_2, 1)
            self.hidden_level_2 = tf.tile(self.hidden_level_2, [1, self.max_sen_len, 1])
            self.hidden_level_2 = tf.concat([self.hidden, self.hidden_level_2], 2)
            self.hidden_level_2 = tf.expand_dims(self.hidden_level_2, -1)
            batch_size = tf.shape(self.hidden_level_2)[0]

            self.predict_11 = tf.expand_dims(self.predict_1, 1)
            # self.level_11 = tf.expand_dims(self.level_1, 1)

            with tf.variable_scope('FeatCap_SemanCap_2'):
                SemanCap_2 = CapsLayer_variant_routing(label=self.predict_11, batch_size=batch_size,
                                                       num_outputs=self.sc_num, vec_len=self.sc_dim,
                                                       iter_routing=self.iter_routing,
                                                       with_routing=False, layer_type='CONV')
                # (self.caps2, self.label_gate_2) = SemanCap_2(input=self.hidden_level_2, kernel_size=self.filter_size,
                #                                              stride=1,
                #                                              embedding_dim=2 * self.hidden_size + self.n_class_1)
                (self.caps2, self.label_gate_2) = SemanCap_2(input=self.hidden_level_2, kernel_size=self.filter_size,
                                                             stride=1,
                                                             embedding_dim=self.hidden_size + self.n_class_1)

            with tf.variable_scope('ASC_ClassCap_2'):
                ASC_ClassCap_2 = CapsLayer_variant_routing(label=self.predict_11, batch_size=batch_size,
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
            self.hidden_level_3 = tf.reshape(self.predict_2, [-1, self.n_class_2])

            self.level_2 = tf.cast(self.level_2, dtype=tf.int32)
            self.level_2 = tf.one_hot(self.level_2, depth=self.n_class_2)
            # self.hidden_level_3 = tf.reshape(self.level_2, [-1, self.n_class_2])

            self.hidden_level_3 = tf.expand_dims(self.hidden_level_3, 1)
            self.hidden_level_3 = tf.tile(self.hidden_level_3, [1, self.max_sen_len, 1])
            self.hidden_level_3 = tf.concat([self.hidden, self.hidden_level_3], 2)
            self.hidden_level_3 = tf.expand_dims(self.hidden_level_3, -1)
            batch_size = tf.shape(self.hidden_level_3)[0]

            self.predict_22 = tf.expand_dims(self.predict_2, 1)
            # self.level_22 = tf.expand_dims(self.level_2, 1)

            with tf.variable_scope('FeatCap_SemanCap_3'):
                SemanCap_3 = CapsLayer_variant_routing(label=self.predict_22, batch_size=batch_size,
                                                       num_outputs=self.sc_num, vec_len=self.sc_dim,
                                                       iter_routing=self.iter_routing,
                                                       with_routing=False, layer_type='CONV')
                # (self.caps3, self.label_gate_3) = SemanCap_3(input=self.hidden_level_3, kernel_size=self.filter_size,
                #                                              stride=1,
                #                                              embedding_dim=2 * self.hidden_size + self.n_class_2)
                (self.caps3, self.label_gate_3) = SemanCap_3(input=self.hidden_level_3, kernel_size=self.filter_size,
                                                             stride=1,
                                                             embedding_dim=self.hidden_size + self.n_class_2)

            with tf.variable_scope('ASC_ClassCap_3'):
                ASC_ClassCap_3 = CapsLayer_variant_routing(label=self.predict_22, batch_size=batch_size,
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
            self.predictions_1 = tf.argmax(self.predict_1, axis=-1, name="predictions_1")
            self.predictions_2 = tf.argmax(self.predict_2, axis=-1, name="predictions_2")
            self.predictions_3 = tf.argmax(self.predict_3, axis=-1, name="predictions_3")

            self.level_3 = tf.cast(self.level_3, dtype=tf.int32)
            self.level_3 = tf.one_hot(self.level_3, depth=self.n_class_3)

            self.loss_1 = margin_loss(self.level_1, self.predict_1)
            self.loss_2 = margin_loss(self.level_2, self.predict_2)
            self.loss_3 = margin_loss(self.level_3, self.predict_3)

            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = self.loss_1 + self.loss_2 + self.loss_3 + sum(reg_loss)

            # w_convert_1 = tf.get_variable(name='w_convert_1', shape=[self.n_class_1, self.n_class_2],
            #                               initializer=tf.random_normal_initializer(stddev=0.01))
            # w_convert_2 = tf.get_variable(name='w_convert_2', shape=[self.n_class_2, self.n_class_3],
            #                               initializer=tf.random_normal_initializer(stddev=0.01))
            #
            # self.predict1 = tf.matmul(self.predict_1, w_convert_1)
            # self.margin_loss_1 = tf.reduce_mean(tf.maximum(self.predict1 - self.predict_2 + 0.1, 0.0))
            # self.predict2 = tf.matmul(self.predict_2, w_convert_2)
            # self.margin_loss_2 = tf.reduce_mean(tf.maximum(self.predict2 - self.predict_3 + 0.1, 0.0))
            #
            # self.loss = self.loss_1 + self.loss_2 + self.loss_3 + 0.1 * self.margin_loss_1 + 0.1 * self.margin_loss_2 + sum(
            #     reg_loss)
