# ==================================
# Author: whatbeg (Qiu Hu)
# Created by: 2017. 5
# Personal Site: http://whatbeg.com
# ==================================

import numpy as np
import src.transforms as trans


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

    def doTest(self):
        self.normalizeSpec()
        print("All Tests Passed!")

if __name__ == '__main__':
    tranSpec = transformsSpec()
    tranSpec.doTest()

