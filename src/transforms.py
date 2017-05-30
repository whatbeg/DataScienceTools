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
