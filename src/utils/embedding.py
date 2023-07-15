#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import numpy as np


def load_word_id_mapping(word_id_file, encoding='utf8'):
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.encode(encoding, 'ignore').decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print('\nload word-id mapping done!\n')
    return word_to_id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        # line = line.encode('GBK', 'ignore').decode('GBK', 'ignore').split()
        line = line.strip().split()
        if len(line) != embedding_dim + 1:
            # print(u'a bad word embedding: {}'.format(line[0]))
            continue
        cnt += 1
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    word_dict['$t$'] = (cnt + 1)
    return word_dict, w2v