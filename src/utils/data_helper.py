#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import numpy as np
import tensorflow as tf
import os
import re
from sklearn.preprocessing import LabelBinarizer
import random
from src.utils.embedding import load_w2v


def text_cleaner(text):
    """
    cleaning spaces, html tags, etc
    parameters: (string) text input to clean
    return: (string) clean_text
    """
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    clean_text = text.lower()
    return clean_text


def load_inputs_data_process(input_file, word_id_mapping, sentence_len, encoding='utf8'):
    word_to_id = word_id_mapping
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    level_1, level_2, level_3 = [], [], []
    import io

    lines = io.open(input_file, "r", encoding="utf8").readlines()
    for i in range(len(lines)):

        line = lines[i].strip().split("\t")
        level_1.append(line[1].strip().split()[0])
        level_2.append(line[2].strip().split()[0])
        level_3.append(line[3].strip().split()[0])

        # sen_len
        text = text_cleaner(line[0])
        doc_list = text.encode(encoding).decode(encoding).strip().lower().split()
        doc_word = []
        for doc_j in doc_list:
            if doc_j in word_to_id:
                doc_word.append(word_to_id[doc_j])
            else:
                doc_word.append(word_to_id['$t$'])

        min_doc_l = min(len(doc_word), sentence_len)
        sen_len.append(min_doc_l)
        mm = doc_word[:min_doc_l] + [0] * (sentence_len - min_doc_l)
        x.append(mm)

    encoder = LabelBinarizer()
    label_1 = encoder.fit_transform(level_1)
    label_2 = encoder.fit_transform(level_2)
    label_3 = encoder.fit_transform(level_3)

    print(np.asarray(x).shape)
    print(np.asarray(sen_len).shape)
    print(np.asarray(label_1).shape)
    print(np.asarray(label_2).shape)
    print(np.asarray(label_3).shape)
    return np.asarray(x), np.asarray(sen_len), \
           np.asarray(label_1), np.asarray(label_2), np.asarray(label_3)


def load_inputs_data_process_wos(input_file, word_id_mapping, sentence_len, encoding='utf8'):
    word_to_id = word_id_mapping
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    level_1, level_2 = [], []
    import io

    lines = io.open(input_file, "r", encoding="utf8").readlines()
    for i in range(len(lines)):

        line = lines[i].strip().split("\t")
        level_1.append(line[1].strip().split()[0])
        level_2.append(line[2].strip().split()[0])

        # sen_len
        text = text_cleaner(line[0])
        doc_list = text.encode(encoding).decode(encoding).strip().lower().split()
        doc_word = []
        for doc_j in doc_list:
            if doc_j in word_to_id:
                doc_word.append(word_to_id[doc_j])
            else:
                doc_word.append(word_to_id['$t$'])

        min_doc_l = min(len(doc_word), sentence_len)
        sen_len.append(min_doc_l)
        mm = doc_word[:min_doc_l] + [0] * (sentence_len - min_doc_l)
        x.append(mm)

    # print(level_1)
    # level_dict = {u'\u793e\u4f1a\u6c11\u751f': [1, 0], u'\u751f\u6d3b\u670d\u52a1': [0, 1]}
    # label_1 = [level_dict[label] for label in level_1]
    encoder = LabelBinarizer()
    label_1 = encoder.fit_transform(level_1)
    label_2 = encoder.fit_transform(level_2)
    print(np.asarray(x).shape)
    print(np.asarray(sen_len).shape)
    print(np.asarray(label_1).shape)
    print(np.asarray(label_2).shape)
    return np.asarray(x), np.asarray(sen_len), np.asarray(label_1), np.asarray(label_2)


def train_dev_split(x, sen_len, label_1, label_2, label_3, radio):
    # np.random.seed(10)
    # np.random.seed([1, len(tr_sentence_id)])
    np.random.seed(random.randint(1, len(x)))

    shuffle_indices = np.random.permutation(np.arange(len(x)))

    x_shuffled = x[shuffle_indices]
    sen_len_shuffled = sen_len[shuffle_indices]
    label_1_shuffled = label_1[shuffle_indices]
    label_2_shuffled = label_2[shuffle_indices]
    label_3_shuffled = label_3[shuffle_indices]

    dev_sample_index = -1 * int(radio * float(len(x)))

    tr_x_train, tr_x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    tr_sen_train, tr_sen_dev = sen_len_shuffled[:dev_sample_index], sen_len_shuffled[dev_sample_index:]
    tr_label_1_train, tr_label_1_dev = label_1_shuffled[:dev_sample_index], label_1_shuffled[dev_sample_index:]
    tr_label_2_train, tr_label_2_dev = label_2_shuffled[:dev_sample_index], label_2_shuffled[dev_sample_index:]
    tr_label_3_train, tr_label_3_dev = label_3_shuffled[:dev_sample_index], label_3_shuffled[dev_sample_index:]

    print("Train/Dev split: {:d}/{:d}".format(len(tr_x_train), len(tr_x_dev)))

    return tr_x_train, tr_sen_train, tr_label_1_train, tr_label_2_train, tr_label_3_train, \
           tr_x_dev, tr_sen_dev, tr_label_1_dev, tr_label_2_dev, tr_label_3_dev


def train_dev_split_wos(x, sen_len, label_1, label_2, radio):
    # np.random.seed(10)
    # np.random.seed([1, len(tr_sentence_id)])
    # np.random.seed(random.randint(1, len(x)))
    np.random.seed(1111)

    shuffle_indices = np.random.permutation(np.arange(len(x)))
    
    x_shuffled = x[shuffle_indices]
    sen_len_shuffled = sen_len[shuffle_indices]
    label_1_shuffled = label_1[shuffle_indices]
    label_2_shuffled = label_2[shuffle_indices]
    
    dev_sample_index = -1 * int(radio * float(len(x)))
    
    tr_x_train, tr_x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    tr_sen_train, tr_sen_dev = sen_len_shuffled[:dev_sample_index], sen_len_shuffled[dev_sample_index:]
    tr_label_1_train, tr_label_1_dev = label_1_shuffled[:dev_sample_index], label_1_shuffled[dev_sample_index:]
    tr_label_2_train, tr_label_2_dev = label_2_shuffled[:dev_sample_index], label_2_shuffled[dev_sample_index:]

    print("Train/Dev split: {:d}/{:d}".format(len(tr_x_train), len(tr_x_dev)))
    return tr_x_train, tr_sen_train, tr_label_1_train, tr_label_2_train, \
           tr_x_dev, tr_sen_dev, tr_label_1_dev, tr_label_2_dev


def load_inputs_data_preprocess(dataset, sentence_len, embedding_path, embedding_dim):
    base_path = os.path.join('data', dataset)
    base_path_1 = os.path.join('/home/zhaof/HMTC/', dataset)
    raw_base_path = os.path.join(base_path, 'raw')

    processed_base_path = os.path.join(base_path_1, 'hierarchical_processed')

    if not os.path.exists(processed_base_path):
        os.makedirs(processed_base_path)

    if dataset == "WebOfScience":
        raw_train_path = os.path.join(raw_base_path, 'total1.txt')

        processed_train_path = os.path.join(processed_base_path, 'train.npz')
        processed_test_path = os.path.join(processed_base_path, 'test.npz')

        embedding_source_path = os.path.join(raw_base_path, embedding_path)
        embedding_path = os.path.join(processed_base_path, 'embedding.npy')
        
        word_id_mapping, w2v = load_w2v(embedding_source_path, embedding_dim)
        np.save(embedding_path, w2v)

        x, sen_len, label_1, label_2 = load_inputs_data_process_wos(raw_train_path, word_id_mapping, sentence_len)
        tr_x, tr_sen_len, tr_label_1, tr_label_2, \
        te_x, te_sen_len, te_label_1, te_label_2 = train_dev_split_wos(x, sen_len, label_1, label_2, 0.1)

        np.savez(processed_train_path, x=tr_x, sen_len=tr_sen_len, label_1=tr_label_1, label_2=tr_label_2)
        np.savez(processed_test_path, x=te_x, sen_len=te_sen_len, label_1=te_label_1, label_2=te_label_2)
    elif dataset == "DBpedia":
        raw_train_path = os.path.join(raw_base_path, 'train2.txt')
        raw_test_path = os.path.join(raw_base_path, 'test2.txt')

        processed_train_path = os.path.join(processed_base_path, 'train.npz')
        processed_dev_path = os.path.join(processed_base_path, 'dev.npz')
        processed_test_path = os.path.join(processed_base_path, 'test.npz')

        embedding_source_path = os.path.join(raw_base_path, embedding_path)
        embedding_path = os.path.join(processed_base_path, 'embedding.npy')

        word_id_mapping, w2v = load_w2v(embedding_source_path, embedding_dim)
        np.save(embedding_path, w2v)

        x, sen_len, label_1, label_2, label_3 = load_inputs_data_process(raw_train_path, word_id_mapping, sentence_len)
        tr_x, tr_sen_len, tr_label_1, tr_label_2, tr_label_3, \
        de_x, de_sen_len, de_label_1, de_label_2, de_label_3 = train_dev_split(x, sen_len, label_1, label_2, label_3,
                                                                               0.1)

        te_x, te_sen_len, te_label_1, te_label_2, te_label_3 = load_inputs_data_process(raw_test_path, word_id_mapping,
                                                                                        sentence_len)

        np.savez(processed_train_path, x=tr_x, sen_len=tr_sen_len, label_1=tr_label_1, label_2=tr_label_2,
                 label_3=tr_label_3)
        np.savez(processed_dev_path, x=de_x, sen_len=de_sen_len, label_1=de_label_1, label_2=de_label_2,
                 label_3=de_label_3)
        np.savez(processed_test_path, x=te_x, sen_len=te_sen_len, label_1=te_label_1, label_2=te_label_2,
                 label_3=te_label_3)
    elif dataset == "query_intent":
        raw_train_path = os.path.join(raw_base_path, 'train.txt')
        raw_dev_path = os.path.join(raw_base_path, 'dev.txt')
        raw_test_path = os.path.join(raw_base_path, 'test.txt')

        processed_train_path = os.path.join(processed_base_path, 'train.npz')
        processed_dev_path = os.path.join(processed_base_path, 'dev.npz')
        processed_test_path = os.path.join(processed_base_path, 'test.npz')

        embedding_source_path = os.path.join(raw_base_path, embedding_path)
        embedding_path = os.path.join(processed_base_path, 'embedding.npy')

        word_id_mapping, w2v = load_w2v(embedding_source_path, embedding_dim)
        np.save(embedding_path, w2v)

        tr_x, tr_sen_len, tr_label_1, tr_label_2 = load_inputs_data_process_wos(raw_train_path, word_id_mapping,
                                                                                sentence_len)
        de_x, de_sen_len, de_label_1, de_label_2 = load_inputs_data_process_wos(raw_dev_path, word_id_mapping,
                                                                                sentence_len)
        te_x, te_sen_len, te_label_1, te_label_2 = load_inputs_data_process_wos(raw_test_path, word_id_mapping,
                                                                                sentence_len)

        np.savez(processed_train_path, x=tr_x, sen_len=tr_sen_len, label_1=tr_label_1, label_2=tr_label_2)
        np.savez(processed_dev_path, x=de_x, sen_len=de_sen_len, label_1=de_label_1, label_2=de_label_2)
        np.savez(processed_test_path, x=te_x, sen_len=te_sen_len, label_1=te_label_1, label_2=te_label_2)
    else:
        raw_train_path = os.path.join(raw_base_path, 'train.txt')
        raw_dev_path = os.path.join(raw_base_path, 'dev.txt')
        raw_test_path = os.path.join(raw_base_path, 'test.txt')

        processed_train_path = os.path.join(processed_base_path, 'train.npz')
        processed_dev_path = os.path.join(processed_base_path, 'dev.npz')
        processed_test_path = os.path.join(processed_base_path, 'test.npz')

        embedding_source_path = os.path.join(raw_base_path, embedding_path)
        embedding_path = os.path.join(processed_base_path, 'embedding.npy')

        word_id_mapping, w2v = load_w2v(embedding_source_path, embedding_dim)
        np.save(embedding_path, w2v)

        tr_x, tr_sen_len, tr_label_1, tr_label_2, tr_label_3 = load_inputs_data_process(raw_train_path, word_id_mapping,
                                                                                        sentence_len)
        de_x, de_sen_len, de_label_1, de_label_2, de_label_3 = load_inputs_data_process(raw_dev_path, word_id_mapping,
                                                                                        sentence_len)
        te_x, te_sen_len, te_label_1, te_label_2, te_label_3 = load_inputs_data_process(raw_test_path, word_id_mapping,
                                                                                        sentence_len)

        np.savez(processed_train_path, x=tr_x, sen_len=tr_sen_len, label_1=tr_label_1, label_2=tr_label_2,
                 label_3=tr_label_3)
        np.savez(processed_dev_path, x=de_x, sen_len=de_sen_len, label_1=de_label_1, label_2=de_label_2,
                 label_3=de_label_3)
        np.savez(processed_test_path, x=te_x, sen_len=te_sen_len, label_1=te_label_1, label_2=te_label_2,
                 label_3=te_label_3)


def margin_loss(y_true, y_pred):
    """
    :param y_true: [None, n_classes]
    :param y_pred: [None, n_classes]
    :return: a scalar loss value.
    """
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    assert_inf_L = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_inf(L))),
                             ['assert_inf_L', L], summarize=100)
    assert_nan_L = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(L))),
                             ['assert_nan_L', L], summarize=100)
    with tf.control_dependencies([assert_inf_L, assert_nan_L]):
        ret = tf.reduce_mean(tf.reduce_sum(L, axis=1))
    return ret


def kl_for_log_probs(log_p, log_q):
    p = tf.exp(log_p)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl
