#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import datetime
import os

import numpy as np

from src.config.config import *
from src.train.make_model import make_model
from src.utils.batch import batch_iter
from sklearn.preprocessing import LabelBinarizer


def train():
    base_path = os.path.join('data', FLAGS.dataset)
    # base_path_1 = os.path.join('/data1/zhaof/HMTC/ckpt_model', FLAGS.dataset)
    base_path_1 = os.path.join('/home/zhaof/HMTC/ckpt_model', FLAGS.dataset)
    
    processed_base_path = os.path.join(base_path, 'hierarchical_processed')
    train_path = os.path.join(processed_base_path, 'train.npz')
    dev_path = os.path.join(processed_base_path, 'dev.npz')
    embedding_path = os.path.join(processed_base_path, 'embedding.npy')
    
    train_dataset = np.load(train_path)
    dev_dataset = np.load(dev_path)

    # encoder = LabelBinarizer()
    # label_1 = encoder.fit_transform(dev_dataset['label_1'])
    # label_2 = encoder.fit_transform(dev_dataset['label_2'])
    # label_3 = encoder.fit_transform(dev_dataset['label_3'])
    #
    # print(encoder.inverse_transform(label_1))
    # print(encoder.inverse_transform(label_2))
    # print(encoder.inverse_transform(label_3))

    # print(list(encoder.inverse_transform(res_data_1)))
    # print(list(encoder.inverse_transform(res_data_2)))
    # print(list(encoder.inverse_transform(res_data_3)))

    word_embedding = np.load(embedding_path)
    print(word_embedding.shape)
    word_embedding = tf.constant(word_embedding, name='word_embedding')

    model = make_model(FLAGS, word_embedding)

    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

    # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss_t, tvars), FLAGS.grad_clip)
    # grads_and_vars = tuple(zip(grads, tvars))
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # level_1
    true_y_1 = tf.argmax(model.level_1, 1, name='true_y_1')
    pred_y_1 = tf.argmax(model.predict_1, 1, name='pred_y_1')
    correct_pred_1 = tf.equal(pred_y_1, true_y_1)
    acc_num_1 = tf.reduce_sum(tf.cast(correct_pred_1, tf.int32), name='acc_number_1')

    # level_2
    true_y_2 = tf.argmax(model.level_2, 1, name='true_y_2')
    pred_y_2 = tf.argmax(model.predict_2, 1, name='pred_y_2')
    correct_pred_2 = tf.equal(pred_y_2, true_y_2)
    acc_num_2 = tf.reduce_sum(tf.cast(correct_pred_2, tf.int32), name='acc_number_2')

    # level_3
    true_y_3 = tf.argmax(model.level_3, 1, name='true_y_3')
    pred_y_3 = tf.argmax(model.predict_3, 1, name='pred_y_3')
    correct_pred_3 = tf.equal(pred_y_3, true_y_3)
    acc_num_3 = tf.reduce_sum(tf.cast(correct_pred_3, tf.int32), name='acc_number_3')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        def train_step(te_x_f, te_sen_len_f, te_label_1_f, te_label_2_f, te_label_3_f, kp):
            feed_dict = {
                model.input_x: te_x_f,
                model.sen_len: te_sen_len_f,
                model.level_1: te_label_1_f,
                model.level_2: te_label_2_f,
                model.level_3: te_label_3_f,
                model.keep_prob: kp
            }
            step, _, losses = sess.run([global_step, optimizer, model.loss], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, losses))

        def dev_step(de_x_f, de_sen_len_f, de_label_1_f, de_label_2_f, de_label_3_f):
            feed_dict = {
                model.input_x: de_x_f,
                model.sen_len: de_sen_len_f,
                model.level_1: de_label_1_f,
                model.level_2: de_label_2_f,
                model.level_3: de_label_3_f,
                model.keep_prob: 1.0
            }
            acc_1, true_1, pred_1, acc_2, true_2, pred_2, acc_3, true_3, pred_3, _loss = \
                sess.run([acc_num_1, true_y_1, pred_y_1,
                          acc_num_2, true_y_2, pred_y_2,
                          acc_num_3, true_y_3, pred_y_3, model.loss], feed_dict)

            return acc_1, true_1, pred_1, acc_2, true_2, pred_2, acc_3, true_3, pred_3, _loss

        checkpoint_dir = os.path.join(base_path_1, FLAGS.saver_checkpoint)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        total_labels = len(dev_dataset['x'])

        max_acc_1, max_acc_2, max_acc_3 = 0, 0, 0
        max_p_label, max_t_label = None, None

        batches_train = batch_iter(
            list(zip(train_dataset['x'], train_dataset['sen_len'], train_dataset['label_1'],
                     train_dataset['label_2'], train_dataset['label_3'])), FLAGS.batch_size, FLAGS.n_iter, True)

        for batch in batches_train:
            tr_x_batch, tr_sen_len_batch, tr_label_1_batch, tr_label_2_batch, tr_label_3_batch = zip(*batch)
            train_step(tr_x_batch, tr_sen_len_batch, tr_label_1_batch, tr_label_2_batch, tr_label_3_batch, FLAGS.keep_prob)

            current_step = tf.train.global_step(sess, global_step)
            if current_step % 300 == 0:
                batches_test = batch_iter(
                    list(zip(dev_dataset['x'], dev_dataset['sen_len'], dev_dataset['label_1'],
                             dev_dataset['label_2'], dev_dataset['label_3'])), 100, 1, False)
                p_label, t_label = None, None
                tag1, tag2, tag3, cost1 = 0, 0, 0, 0
                for batch_ in batches_test:
                    te_x_batch, te_sen_len_batch, te_label_1_batch, te_label_2_batch, te_label_3_batch = zip(*batch_)

                    acc_num_11, true_11, pred_11, acc_num_22, true_22, pred_22, acc_num_33, true_33, pred_33, cost = \
                        dev_step(te_x_batch, te_sen_len_batch, te_label_1_batch, te_label_2_batch, te_label_3_batch)
                    tag1 += acc_num_11
                    tag2 += acc_num_22
                    tag3 += acc_num_33

                    # p_label += pp_labell
                    # t_label += tt_labell
                    cost1 += cost

                print("\nEvaluation:")
                print('all samples={}, level1 correct prediction ={}, level2 correct prediction ={}, '
                      'level3 correct prediction ={}'.format(total_labels, tag1, tag2, tag3))

                accuracy_level_1 = tag1 / float(total_labels)
                accuracy_level_2 = tag2 / float(total_labels)
                accuracy_level_3 = tag3 / float(total_labels)
                test_loss = cost1 / float(total_labels)

                print("step {}, loss {}, level 1 acc {:g}, level 2 acc {:g}, level 3 acc {:g}"
                      .format(current_step, test_loss, accuracy_level_1, accuracy_level_2, accuracy_level_3))

                if accuracy_level_3 > max_acc_3:
                    max_acc_1 = accuracy_level_1
                    max_acc_2 = accuracy_level_2
                    max_acc_3 = accuracy_level_3
                    max_p_label = p_label
                    max_t_label = t_label
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                # if accuracy_level_1 > max_acc_1:
                #     max_acc_1 = accuracy_level_1
                #     max_acc_2 = accuracy_level_2
                #     max_acc_3 = accuracy_level_3
                #     max_p_label = p_label
                #     max_t_label = t_label
                #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #     print("Saved model checkpoint to {}\n".format(path))
                print("level1 topacc {:g}, level2 topacc {:g}, level3 topacc {:g}".
                      format(max_acc_1, max_acc_2, max_acc_3))
                print("\n")
        print("level1 topacc {:g}, level2 topacc {:g}, level3 topacc {:g}".format(max_acc_1, max_acc_2, max_acc_3))
        print("\n")
