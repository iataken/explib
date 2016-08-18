# -*- coding:utf8 -*-
import sys
sys.path.append("..")
from base import expEval
from sklearn.metrics import mean_squared_error
import math
import numpy as np


class expEvalPATK(expEval):
    name = "P @ K"
    desc = "P @ K"

    def __init__(self, K = 3):
        super(expEvalPATK,self).__init__()
        self.K = K

    def evaluate(self, true_values, pre_values, outputs=None):

        testing_data = true_values
        result_matrix = pre_values

        # 利用result_matrix和testing_data组合下面两个dict
        authors_predict = []
        authors_real = []

        num_confs = 20

        for author in testing_data.keys():

            author_predict = result_matrix[author]
            author_real = []
            if len(testing_data[author]) != 0:
                for index in range(0, num_confs):
                    if index in testing_data[author]:
                        author_real.append(1)
                    else:
                        author_real.append(0)
            else:
                for index in range(0, num_confs):
                    author_real.append(0)

            authors_real.append(author_real)
            authors_predict.append(author_predict)

        # 构建用于分析的特殊dict

        authors_predict = np.array(authors_predict)
        authors_real = np.array(authors_real)

        n = len(authors_real)
        total = 0.0
        for cur_real, cur_pre in zip(authors_real, authors_predict):
            pre_sorted_reverse = np.argsort(cur_pre)[::-1]
            cur_real_np = np.array(cur_real)
            temp = cur_real_np[pre_sorted_reverse][:self.K]
            sum_temp = 0.0
            for x in temp:
                sum_temp += x
            score_temp = sum_temp / len(temp)
            total += score_temp
        values = total/n
        self.value.append(values)
