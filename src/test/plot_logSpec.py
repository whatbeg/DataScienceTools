# ==================================
# Author: whatbeg (Qiu Hu)
# Created by: 2017. 5
# Personal Site: http://whatbeg.com
# ==================================

import src.main.plot_log as plog


class plot_logSpec(object):

    def __init__(self):
        pass

    def analyse_bigdlSpec(self):

        plog.analyse_bigdl(["./Spec_data/plot_log_bigdl.log", ], "Converge Speed")

    def analyse_pytorchSpec(self):

        plog.analyse_pytorch(["./Spec_data/plot_log_pytorch.log", ], "Analyse pytorch log")

    def doTest(self):
        self.analyse_bigdlSpec()
        self.analyse_pytorchSpec()
        print ("All Test Passed!")


if __name__ == '__main__':
    plSpec = plot_logSpec()
    plSpec.doTest()

