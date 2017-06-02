# ==================================
# Author: whatbeg (Qiu Hu)
# Created by: 2017. 5
# Personal Site: http://whatbeg.com
# ==================================

import numpy as np
import src.plot_log as plog


class plot_logSpec(object):

    def __init__(self):
        pass

    def analyse_bigdlSpec(self):

        plog.analyse_bigdl(["../Spec_data/plot_log_bigdl.log", ], "Converge Speed")

    def doTest(self):
        self.analyse_bigdlSpec()
        print ("All Test Passed!")


if __name__ == '__main__':
    plSpec = plot_logSpec()
    plSpec.doTest()
