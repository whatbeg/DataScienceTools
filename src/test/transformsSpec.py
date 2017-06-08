# ==================================
# Author: whatbeg (Qiu Hu)
# Created by: 2017. 5
# Personal Site: http://whatbeg.com
# ==================================

import numpy as np

import src.main.transforms as trans


class transformsSpec(object):

    def __init__(self):
        pass

    def normalizeSpec(self):

        tensor = np.random.uniform(0.3, 0.1, size=(1, 2, 2, 1))
        after_tensor = trans.Normalize(tensor, (0.3, ), (0.1, ))
        for _t in tensor:
            _t -= 0.3
            _t /= 0.1
        assert (tensor == after_tensor).all()

    def generate_dataSpec(self):

        tensor = np.random.uniform(0.3, 0.1, size=(100, 2, 2, 1))
        labels = np.random.binomial(100, 0.5, size=(100, ))
        for data, label in trans.generate_data(tensor, labels, batchSize=32, shuffle=False):
            assert data.shape == (32, 2, 2, 1) or data.shape == (4, 2, 2, 1)
            assert label.shape == (32, ) or label.shape == (4, )

    def doTest(self):
        self.normalizeSpec()
        self.generate_dataSpec()
        print("All Tests Passed!")

if __name__ == '__main__':
    tranSpec = transformsSpec()
    tranSpec.doTest()

