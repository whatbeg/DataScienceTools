# ==================================
# Author: whatbeg (Qiu Hu)
# Created by: 2017. 5
# Personal Site: http://whatbeg.com
# ==================================

"""Utilities for data transform and preprocessing"""

import numpy as np


def Normalize(tensor, mean, std):
    """
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel to `channel = (channel - mean) / std`
    Given mean: GreyLevel and std: GrayLevel,
    will normalize the channel to `channel = (channel - mean) / std`

    :param tensor: image tensor to be normalized
    :param mean: mean of every channel
    :param std: standard variance of every channel
    :return: normalized tensor
    """
    import copy
    _tensor = copy.deepcopy(tensor)
    for t, m, s in zip(_tensor, mean, std):
        t -= m
        t /= s
    return _tensor


def generate_data(data, label, batchSize, shuffle=True):
    """
    generate a batch of data, according to (data, label)

    :param data: data tensor, Type: ndarray
    :param label: labels, Type: ndarray
    :param batchSize: how many samples in one batch
    :param shuffle: if shuffle primitive data and labels
    :return: Iterator of every mini-batch (data, label)
    """
    assert batchSize > 0
    data_len = data.shape[0]
    total_batch = data_len / batchSize + (1 if data_len % batchSize != 0 else 0)
    if shuffle:
        indices = np.random.permutation(data_len)
        data = data[indices]
        label = label[indices]
    for idx in range(total_batch):
        start = idx * batchSize
        end = min((idx + 1) * batchSize, data_len)
        yield data[start:end], label[start:end]
