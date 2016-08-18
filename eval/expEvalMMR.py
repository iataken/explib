# -*- coding:utf8 -*-
import sys
sys.path.append("..")
from base import expEval
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np

class expEvalMMR(expEval):
    name = "MMR"
    desc = "MMR"

    def __init__(self):
        super(expEvalMMR,self).__init__()

    def evaluate(self, true_values, pre_values, outputs=None):

        testing_data = true_values
        result_matrix = pre_values

        # 利用result_matrix和testing_data组合下面两个dict
        authors_predict = []
        authors_real = []

        num_confs = 20

        for author in testing_data.keys():
            author_real = []
            author_predict = []

            if len(testing_data[author]) != 0:
                for conf in testing_data[author]:
                    author_real_temp = []
                    author_predict_temp = result_matrix[author]
                    for index in range(0, num_confs):
                        if index == conf:
                            author_real_temp.append(1)
                        else:
                            author_real_temp.append(0)
                    author_real.append(author_real_temp)
                    author_predict.append(author_predict_temp)
            else:
                author_real_temp = []
                author_predict_temp = result_matrix[author]
                for index in range(0, num_confs):
                    author_real_temp.append(0)
                author_real.append(author_real_temp)
                author_predict.append(author_predict_temp)

            authors_real.append(author_real)
            authors_predict.append(author_predict)

        # 构建用于分析的特殊dict
        n = len(authors_real)
        total = 0.0
        for cur_real, cur_pre in zip(authors_real, authors_predict):
            cur_eva = label_ranking_average_precision_score(cur_real, cur_pre)
            total += cur_eva
        values = total/n
        self.value.append(values)