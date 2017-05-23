import numpy as np
import pandas as pd


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

    :param column: primitive column
    :param boundaries: boundaries to bucketize
    :return: bucketized column
    """
    import copy
    _column = copy.deepcopy(column)
    for i in range(len(_column)):
        _column[i] = binary_search(_column[i], boundaries)
    return _column

