#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.flags.DEFINE_integer('hidden_size', 200, 'number of hidden unit')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')

tf.flags.DEFINE_integer('n_class_1', 10, 'number of distinct class')
tf.flags.DEFINE_integer('n_class_2', 10, 'number of distinct class')
tf.flags.DEFINE_integer('n_class_3', 10, 'number of distinct class')
tf.flags.DEFINE_integer('num_heads', 8, 'number of distinct class')

tf.flags.DEFINE_integer("grad_clip", 5, "grad clip to prevent gradient explode")

tf.flags.DEFINE_integer('filter_size', 3, 'filter_size')
tf.flags.DEFINE_integer('sc_num', 16, 'sc_num')
tf.flags.DEFINE_integer('sc_dim', 16, 'sc_dim')
tf.flags.DEFINE_integer('cc_num', 10, 'cc_num')
tf.flags.DEFINE_integer('cc_dim', 32, 'cc_dim')
tf.flags.DEFINE_integer('iter_routing', 3, 'routing iteration')

tf.flags.DEFINE_integer('max_sentence_len', 500, 'max number of tokens per sentence')

tf.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.flags.DEFINE_float('random_base', 0.1, 'initial random base')
tf.flags.DEFINE_integer('n_iter', 100, 'number of train iter')
tf.flags.DEFINE_float('keep_prob', 0.5, 'dropout keep prob')

tf.flags.DEFINE_string('dataset', 'DBpedia', 'DBpedia')
tf.flags.DEFINE_string('phase', 'train', ['hierarchical_preprocess', 'train', 'test'])
tf.flags.DEFINE_string('model_type', 'han_model', ['han_model', 'han_model6', 'HAN_capsule'])

tf.flags.DEFINE_string('train_file_path', 'data/beer/train1.txt', 'training file')
tf.flags.DEFINE_string('validate_file_path', 'data/beer/dev1.txt', 'testing file')
tf.flags.DEFINE_string('embedding_file_path', 'data/beer/ret_emb', 'embedding file')
tf.flags.DEFINE_string('saver_checkpoint', 'data/beer/checkpoint', 'prob')

tf.flags.DEFINE_string('config_path', 'src/bert_classifier/config/dbpedia_config.json', 'prob')

tf.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
tf.flags.DEFINE_string('true_file', 'true1.txt', 'prob')