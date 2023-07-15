#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import numpy as np
import tensorflow as tf
import os
from src.utils.batch import batch_iter
from sklearn.preprocessing import LabelBinarizer


tf.flags.DEFINE_string("checkpoint_dir", "han_capsule_db_label_rout_test_3", "checkpoint")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string('output_path', 'result_capsule.txt', 'prob')

tf.flags.DEFINE_string('output_c_ij_path_2', 'c_ij_2.txt', 'prob')
tf.flags.DEFINE_string('output_c_ij_path_3', 'c_ij_3.txt', 'prob')
tf.flags.DEFINE_string('output_label_weight_path_2', 'label_w_2.txt', 'prob')
tf.flags.DEFINE_string('output_label_weight_path_3', 'label_w_3.txt', 'prob')


FLAGS = tf.flags.FLAGS


def test():
    base_path = os.path.join('HMTC/data', FLAGS.dataset)
    base_path_1 = os.path.join('HMTC/ckpt_model', FLAGS.dataset)

    # base_path = os.path.join('./data/', FLAGS.dataset)

    processed_base_path = os.path.join(base_path, 'hierarchical_processed')
    test_path = os.path.join(processed_base_path, 'test.npz')
    test_dataset = np.load(test_path)

    output_file = os.path.join(base_path_1, FLAGS.output_path)

    output_c_ij_file_2 = os.path.join(base_path_1, FLAGS.output_c_ij_path_2)
    output_c_ij_file_3 = os.path.join(base_path_1, FLAGS.output_c_ij_path_3)
    output_label_weight_file_2 = os.path.join(base_path_1, FLAGS.output_label_weight_path_2)
    output_label_weight_file_3 = os.path.join(base_path_1, FLAGS.output_label_weight_path_3)

    print("\nTest...\n")
    checkpoint_dir = os.path.join(base_path_1, FLAGS.checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input/input_x").outputs[0]
            sen_len = graph.get_operation_by_name("input/input_sen_len").outputs[0]
            level_1 = graph.get_operation_by_name("input/input_level_1").outputs[0]
            level_2 = graph.get_operation_by_name("input/input_level_2").outputs[0]
            level_3 = graph.get_operation_by_name("input/input_level_3").outputs[0]
            keep_prob = graph.get_operation_by_name("input/input_keep_prob").outputs[0]

            acc_num_1 = graph.get_operation_by_name("acc_number_1").outputs[0]
            acc_num_2 = graph.get_operation_by_name("acc_number_2").outputs[0]
            acc_num_3 = graph.get_operation_by_name("acc_number_3").outputs[0]

            true_y_1 = graph.get_operation_by_name("true_y_1").outputs[0]
            true_y_2 = graph.get_operation_by_name("true_y_2").outputs[0]
            true_y_3 = graph.get_operation_by_name("true_y_3").outputs[0]

            pred_y_1 = graph.get_operation_by_name("pred_y_1").outputs[0]
            pred_y_2 = graph.get_operation_by_name("pred_y_2").outputs[0]
            pred_y_3 = graph.get_operation_by_name("pred_y_3").outputs[0]

            label_gate_weights_2 = graph.get_operation_by_name("level_2_cap/FeatCap_SemanCap_2/label_gate_weights").outputs[0]
            label_gate_weights_3 = graph.get_operation_by_name("level_3_cap/FeatCap_SemanCap_3/label_gate_weights").outputs[0]

            pre_c_ij_2 = graph.get_operation_by_name("level_2_cap/ASC_ClassCap_2/pre_c_ij").outputs[0]
            pre_c_ij_3 = graph.get_operation_by_name("level_3_cap/ASC_ClassCap_3/pre_c_ij").outputs[0]

            def test_step(te_x, te_sen_len, te_label_1, te_label_2, te_label_3):
                feed_dict = {
                    input_x: te_x,
                    sen_len: te_sen_len,
                    level_1: te_label_1,
                    level_2: te_label_2,
                    level_3: te_label_3,
                    keep_prob: 1.0
                }
                tf_acc_1, tf_acc_2, tf_acc_3, true_1, pred_1, true_2, pred_2, true_3, pred_3, label_w_2, label_w_3, c_ij_2, c_ij_3,  = sess.run(
                    [acc_num_1, acc_num_2, acc_num_3, true_y_1, pred_y_1, true_y_2, pred_y_2, true_y_3, pred_y_3,
                     label_gate_weights_2, label_gate_weights_3, pre_c_ij_2, pre_c_ij_3],
                    feed_dict)
                return tf_acc_1, tf_acc_2, tf_acc_3, true_1, pred_1, true_2, pred_2, true_3, pred_3, label_w_2, label_w_3, c_ij_2, c_ij_3

            tag1, tag2, tag3 = 0, 0, 0
            t_label_1, t_label_2, t_label_3, p_label_1, p_label_2, p_label_3 = [], [], [], [], [], []

            c_ij_list_2, c_ij_list_3 = [], []
            label_w_list_2, label_w_list_3 = [], []

            batches_test = batch_iter(
                list(zip(test_dataset['x'], test_dataset['sen_len'], test_dataset['label_1'], test_dataset['label_2'],
                         test_dataset['label_3'])), 100, 1, False)
            for batch_ in batches_test:
                te_x_batch, te_sen_len_batch, te_label_1_batch, te_label_2_batch, te_label_3_batch = zip(*batch_)
                acc_num_11, acc_num_22, acc_num_33, true_11, pred_11, true_22, pred_22, true_33, pred_33, \
                label_w_22, label_w_33, c_ij_22, c_ij_33 = test_step(te_x_batch, te_sen_len_batch, te_label_1_batch,
                                                                     te_label_2_batch, te_label_3_batch)
                tag1 += acc_num_11
                tag2 += acc_num_22
                tag3 += acc_num_33

                t_label_1 += list(true_11)
                t_label_2 += list(true_22)
                t_label_3 += list(true_33)
                p_label_1 += list(pred_11)
                p_label_2 += list(pred_22)
                p_label_3 += list(pred_33)

                label_w_list_2 += list(label_w_22)
                label_w_list_3 += list(label_w_33)
                c_ij_list_2 += list(c_ij_22)
                c_ij_list_3 += list(c_ij_33)

            print(
                'all samples={}, level1 correct prediction={}, level2 correct prediction={}, ''level3 correct prediction={}'.format(
                    len(test_dataset['x']), tag1, tag2, tag3))

            accuracy_1 = tag1 / float(len(test_dataset['x']))
            accuracy_2 = tag2 / float(len(test_dataset['x']))
            accuracy_3 = tag3 / float(len(test_dataset['x']))
            print(
                'level1 accuracy={}, level2 accuracy={}, level3 accuracy={}'.format(accuracy_1, accuracy_2, accuracy_3))