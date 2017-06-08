# ==================================
# Author: whatbeg (Qiu Hu)
# Created by: 2017. 5
# Personal Site: http://whatbeg.com
# ==================================

"""Plot Log files on different forms"""

import numpy as np
import matplotlib.pyplot as plt


def analyse_bigdl(files, title):
    """
    analyse log files in files(list), and plot 3 figures for bigdl-form training or converge process.
    the newest (up to 2017. 5. 30) bigdl log is like:
    `2017-05-29 14:26:53 INFO  DistriOptimizer$:280 - [Epoch 1 0/32561][Iteration 1][Wall Clock 0.0s] Train 256 in 0.318120388seconds. Throughput is 804.7268 records/second. Loss is 30565.014.`
    `2017-05-29 14:26:58 INFO  DistriOptimizer$:568 - [Wall Clock 5.592995018s] Validate model...`
    `2017-05-29 14:26:59 INFO  DistriOptimizer$:610 - Top1Accuracy is Accuracy(correct: 11752, count: 16281, accuracy: 0.7218229838462011)`
    `2017-05-29 14:26:59 INFO  DistriOptimizer$:610 - Loss is (Loss: 250195.16, count: 2036, Average Loss: 122.885635)`

    3 figures is following:
    Wall clock -- Top1 Accuracy
    Epoch -- Top1 Accuracy
    Wall clock -- Train Loss

    :param files: log file list
    :param title: figure title
    :return: None
    """
    assert len(files) > 0
    for filename in files:
        wallclock, trainwallclock, top1acc, loss, throughput, testloss = ([], [], [], [], [], [])
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line.count("Validate model..."):
                    wallclock.append(line.strip().split(' ')[8][:-2])
                elif line.count("Top1Accuracy is"):
                    top1acc.append(line.strip().split(' ')[13][:-1])
                elif line.count("records/second. Loss is"):
                    loss.append(line.strip().split(' ')[22][:-1])
                    trainwallclock.append(line.strip().split(' ')[11][:-2])
                elif line.count("Throughput is"):
                    throughput.append(line.strip().split(' ')[18][:-1])
                elif line.count("Loss is (Loss:"):
                    testloss.append(line.strip().split(' ')[14][:-1])
            plt.figure(1)
            plt.title(title)
            plt.ylabel("Top1 Accuracy")
            plt.xlabel("Wall Clock (s)")
            plt.plot(wallclock, top1acc, label=filename[:-4])
            plt.legend(loc="lower right")
            plt.grid()
            plt.figure(2)
            plt.title(title)
            plt.ylabel("Top1 Accuracy")
            plt.xlabel("Epoch")
            plt.plot(range(1, len(top1acc)+1), top1acc, label=filename[:-4])
            plt.legend(loc="lower right")
            plt.grid()
            plt.figure(3)
            plt.title(title)
            plt.ylabel("Train Loss")
            plt.xlabel("Wall Clock (s)")
            plt.plot(trainwallclock, loss, label=filename[:-4])
            plt.legend(loc="upper right")
            plt.grid()
    plt.show()


def analyse_pytorch(files, title):
    """
    analyse pytorch like log.
    For example,
    `Train Epoch: 1 [20480/32561 (62%)]	Loss: 0.598629`
    `Test set: Average loss: 0.0024, Accuracy: 12302/16281 (75.6%)`

    :param files: log file list
    :param title: figures title
    :return: None
    """
    assert len(files) > 0
    for filename in files:
        testloss, top1acc = [], []
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line.count('Test set: Average loss'):
                    testloss.append(float(line.strip().split(' ')[6][:-1]))
                    top1acc.append(float(line.strip().split(' ')[9][1:-2]))

        plt.figure(1)
        plt.title(title)
        plt.ylabel("Top1 Accuracy (%)")
        plt.xlabel("Epoch")
        plt.plot(range(1, len(top1acc) + 1), top1acc, label=filename[:-4])
        plt.legend(loc="lower right")
        plt.grid()
        plt.figure(2)
        plt.title(title)
        plt.ylabel("Test Loss")
        plt.xlabel("Epoch")
        plt.plot(range(1, len(testloss) + 1), testloss, label=filename[:-4])
        plt.legend(loc="upper right")
        plt.grid()
    plt.show()
