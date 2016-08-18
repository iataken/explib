# -*- coding:utf8 -*-
from sklearn.cross_validation import KFold
import sys
sys.path.append("..")
from base import expSetting
from scipy.sparse import csc_matrix
import numpy as np
import random

class expSettingCV(expSetting):
    name = 'CV'
    desc = 'Cross-validation'

    def __init__(self):
        super(expSettingCV,self).__init__()

    def evaluate(self):
        evals = self.evals
        clf = self.model
        num_eval = len(evals)
        data = self.data.data
        target = self.data.target

        kf = KFold(len(target), n_folds=5)
        for train, test in kf:
            train_data = data[train]
            train_label = target[train]
            test_data = data[test]
            test_label = target[test]

            pre_labels = clf.predict(train_data,train_label,test_data)

            for i in xrange(num_eval):
                self.evals[i].evaluate(test_label,pre_labels)


class expSettingMMR(expSetting):
    name = 'MMR-PATK'
    desc = 'MMR-PATK'

    def __init__(self,five = 5, two = 2, test = 200):
        super(expSettingMMR,self).__init__()
        self.five = five
        self.two = two
        self.test = test

    def evaluate(self):
        evals = self.evals
        num_eval = len(evals)
        clf = self.model
        pub = self.data.pub
        adict = self.data.adict
        cdict = self.data.cdict
        author_cycle = self.data.author_cycle
        conf_cycle = self.data.conf_cycle

        # shuffle
        shuffle_dic = {}
        num_author = len(adict)
        while True:
            if len(shuffle_dic) == self.test:
                break
            randomint = random.randint(0, num_author-1)
            if randomint not in shuffle_dic.keys():
                shuffle_dic[randomint] = []
        # shuffle_dict容器中装着将成为测试集的序列号

        # 读取pub文件，一边读取一边组装training_data, testing_data
        training_data = {}
        testing_data = {}
        for line in pub:
            data = line.strip().split(',')
            author = data[0]
            conf = data[1]
            year = int(data[2])
            astart = author_cycle[author.decode('utf-8')]['start']

            if adict[author] not in shuffle_dic.keys():
                # 这个作者属于训练集
                if adict[author] not in training_data.keys():
                   training_data[adict[author]] = []
                if year - astart + 1 <= self.five and cdict[conf] not in training_data[adict[author]]:
                    training_data[adict[author]].append(cdict[conf])
                else:
                    pass
                    # 舍弃所有大于五年的数据
            else:
                # 这个作者属于测试集
                if adict[author] not in testing_data.keys():
                    testing_data[adict[author]] = []
                if adict[author] not in training_data.keys():
                    training_data[adict[author]] = []
                if year - astart + 1 <= self.two and cdict[conf] not in training_data[adict[author]]:
                    # 前两年的数据， 还应该加入训练集
                    training_data[adict[author]].append(cdict[conf])
                elif year - astart + 1 <= self.five and cdict[conf] not in testing_data[adict[author]]:
                    # 后三年的数据， 应该加入测试集
                    testing_data[adict[author]].append(cdict[conf])
                else:
                    pass
                    # 舍弃所有大于五年的数据

        # training_data和testing_data创建完毕

        # 下面构造四个dict， 用来创建matrix
        row_test = []
        col_test = []
        row_train = []
        col_train = []
        for author_index in training_data.keys():
            for conf_index in training_data[author_index]:
                row_train.append(author_index)
                col_train.append(conf_index)

        for author_index in testing_data.keys():
            for conf_index in testing_data[author_index]:
                row_test.append(author_index)
                col_test.append(conf_index)

        data = np.ones(len(row_train))
        matrix = csc_matrix((data, (row_train, col_train)), shape=(len(adict), len(cdict)))
        matrix_array1 = matrix.toarray()

        # 现在需要找出testing_data里面哪些是0，然后用平均值填充它
        # 先求所有平均值，待会啥需要就用啥
        average_col = {}
        for i in range(0,len(cdict)):
            col_temp = matrix_array1[:, i]
            sum_temp = 0.0
            for j in col_temp:
                sum_temp += j
            aver_temp = sum_temp/len(col_temp)
            average_col[i] = aver_temp

        # 构造新data，待会直接往里面加东西
        data = []
        for i in range(0, len(row_train)):
            data.append(1.0)

        for test in testing_data.keys():
            for i in range(0, len(cdict)):
                if i not in testing_data[test]:
                    # 测试集这个会议信息为0，用平均值填充
                    # test是author编号，i是会议编号
                    row_train.append(test)
                    col_train.append(i)
                    data.append(average_col[i])
        # 至此，training_data修改完毕
        data = np.array(data)
        matrix = csc_matrix((data, (row_train, col_train)), shape=(len(adict), len(cdict)))

        # =========================predict=========================
        print 'predicting...'
        result_matrix = clf.predict(matrix, 0, 0)
        # =========================predict=========================

        # =========================evaluate=========================
        print 'evaluating...'
        for i in xrange(num_eval):
            self.evals[i].evaluate(testing_data, result_matrix)
        # =========================evaluate=========================


