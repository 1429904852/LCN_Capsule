#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import numpy as np


def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    # 需要循环多少个周期，每一个迭代周期需要运行多少次batch
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 每个周期运算多少个batch
        for batch_num in range(num_batches_per_epoch):
            # 开始位置
            start_index = batch_num * batch_size
            # 结束位置
            end_index = min((batch_num + 1) * batch_size, data_size)
            # 生成器生成不同的batch
            yield shuffled_data[start_index:end_index]