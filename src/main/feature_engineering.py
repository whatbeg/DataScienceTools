# ==================================
# Author: whatbeg (Qiu Hu)
# Created by: 2017. 5
# Personal Site: http://whatbeg.com
# ==================================

"""Utilities for feature engineering"""

import numpy as np


def binary_search(val, array, start=0):
    """
    binary search implementation

    :param val: value to search
    :param array: data array to be searched
    :param start: 0 if array starts with 0 else 1
    :return: location of val in array, or bucket fall in if not in array
    """
    low = start
    high = len(array) - 1 + start
    while low <= high:
        mid = (low + high) / 2
        if array[mid] == val:
            return mid
        elif array[mid] > val:
            high = mid-1
        else:
            low = mid+1
    return low


def bucketized_column(column, boundaries):
    """
    transform every value of a column to corresponding bucket according to boundaries

    :param column: primitive column, type is list
    :param boundaries: boundaries to bucketize
    :return: bucketized column
    """
    import copy
    _column = copy.deepcopy(column)
    for i in range(len(_column)):
        _column[i] = binary_search(_column[i], boundaries)
    return _column


def discretize_for_lookupTable(df, columns, start=0):
    """
    discretize for BigDL's lookupTable's requirement: elements of input should be little than or equal to $nIndex + 1

    :param df: data tensor. Type must be numpy.ndarray
    :param columns: columns to do discretize
    :param start: index that starts from
    :return: discretized data tensor
    """
    for col in columns:
        total = sorted({}.fromkeys(df[:, col]).keys())
        total_dict = {k: i+start
                      for i, k in enumerate(total)}
        for _ in range(len(df[:, col])):
            if df[_, col] not in total_dict.keys():
                df[_, col] = 1
            else:
                df[_, col] = total_dict[df[_, col]]
    return df


def cross_column(columns, hash_backet_size=1e4):
    """
    generate cross column feature from `columns` with hash bucket.

    :param columns: columns to use to generate cross column, Type must be ndarray
    :param hash_backet_size: hash bucket size to bucketize cross columns to fixed hash bucket
    :return: cross column, represented as a ndarray
    """
    assert columns.shape[0] > 0 and columns.shape[1] > 0
    _crossed_column = np.zeros((columns.shape[0], 1))
    for i in range(columns.shape[0]):
        _crossed_column[i, 0] = (hash("_".join(map(str, columns[i, :]))) % hash_backet_size
                                 + hash_backet_size) % hash_backet_size
    return _crossed_column


def sparse_column(column, vocab_size):
    """
    convert integer id to sparse representation.
    For example, 3 -> [0, 0, 0, 1, 0, ...]

    :param column: the whole column with integer ids of this feature, Type: ndarray
    :param vocab_size: length of sparse vector
    :return: new column consist of converted sparse features, Type: ndarray
    """
    n = column.shape[0]
    assert n > 0 and vocab_size > 0
    _newcol = np.zeros((n, vocab_size))
    for row in range(n):
        ind = int(column[row])
        assert 0 <= ind < vocab_size
        np.put(_newcol[row], ind, 1)
    return _newcol

