import numpy as np
import pandas as pd

import src.feature_engineering as feng


class feature_engineeringSpec():

    def __init__(self):
        pass

    def binarySearchSpec(self):

        array = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        assert feng.binary_search(7, array) == 0
        # print ("feng.binary_search(7, array) == {}".format(feng.binary_search(7, array)))
        assert feng.binary_search(20, array) == 1
        assert feng.binary_search(80, array) == 10
        assert feng.binary_search(-1, array) == 0

    def bucketized_columnSpec(self):

        column = [5, 29, 30, 43, 64, 89]
        boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        feature_column = feng.bucketized_column(column, boundaries)
        # print (feature_column)
        assert feature_column == [0, 2, 2, 5, 9, 10]

    def doTest(self):
        self.binarySearchSpec()
        self.bucketized_columnSpec()
        print ("All Test Passed!")

if __name__ == '__main__':
    feSpec = feature_engineeringSpec()
    feSpec.doTest()


