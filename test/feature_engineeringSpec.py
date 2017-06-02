# ==================================
# Author: whatbeg (Qiu Hu)
# Created by: 2017. 5
# Personal Site: http://whatbeg.com
# ==================================

import numpy as np
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
        assert feature_column == [0, 2, 2, 5, 9, 10]

    def discretize_for_lookupTableSpec(self):

        column = [1, 2, 3]
        data_tensor = np.array([
            [45, 3, 12, 2],
            [3, 4, 5, 9],
            [24, 6, 2, 9]
        ])
        data_tensor = feng.discretize_for_lookupTable(data_tensor, column, 1)
        assert (data_tensor == np.array([[45, 1, 3, 1], [3,  2,  2,  2], [24,  3,  1,  2]])).all()

    def cross_columnSpec(self):

        columns = np.array([
            ['lisa', 'doctor', 30],
            ['william', 'worker', 23],
            ['allen', 'lawyer', 20]
        ])
        name_occupation = feng.cross_column(columns[:, :2], 100)
        assert name_occupation.shape == (3, 1)
        assert (name_occupation == np.array([[38], [22], [39]])).all()
        name_occupation_salary = feng.cross_column(columns, 300)
        assert (name_occupation_salary == np.array([[183], [95], [279]])).all()

    def doTest(self):
        self.binarySearchSpec()
        self.bucketized_columnSpec()
        self.discretize_for_lookupTableSpec()
        self.cross_columnSpec()
        print ("All Test Passed!")

if __name__ == '__main__':
    feSpec = feature_engineeringSpec()
    feSpec.doTest()


