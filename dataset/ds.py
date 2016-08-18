# -*- coding:utf8 -*-
import random

import numpy as np
from DBLP.author_cycle import get_author_cycle
from bunch import Bunch
from scipy.sparse import csc_matrix
from sklearn.datasets import load_iris

from dataset.DBLP.conf_cycle import get_conf_cycle
import sys
sys.path.append("..")
from base import expDataset


class expDatasetIris(expDataset):
    name = 'Dataset-Iris'
    desc = 'Iris'

    def __init__(self,):
        super(expDatasetIris,self).__init__()

    def load(self):
        data = Bunch()
        data.data = load_iris().data
        data.target = load_iris().target
        return data

class expDatasetDBLP(expDataset):
    name = 'Dataset-DBLP'
    desc = 'DBLP'

    def __init__(self,):
        super(expDatasetDBLP,self).__init__()

    def load(self):
        five = 5
        two = 2
        test = 200
        author_cycle = get_author_cycle('./dataset/DBLP/dblp_authors_20confs.dic')
        conf_cycle = get_conf_cycle('./dataset/DBLP/dblp_authors_20confs.dic')
        pub = open('./dataset/DBLP/pub.csv', 'r')

        af = open('./dataset/DBLP/authors.csv','r')
        cf = open('./dataset/DBLP/conf.csv','r')
        adict = {}
        for line in af:
            data = line.strip().split(',')
            author = data[1]
            index = int(data[0])
            adict[author] = index
        af.close()
        cdict = {}
        for line in cf:
            data = line.strip().split(',')
            conf = data[1]
            index = int(data[0])
            cdict[conf] = index
        cf.close()
        # adict和cdict建立完毕
        # 数据集包括pub，adict，cdict，author_cycle，conf_cycle

        dataset = Bunch()
        dataset.pub = pub
        dataset.adict = adict
        dataset.cdict = cdict
        dataset.author_cycle = author_cycle
        dataset.conf_cycle = conf_cycle
        return dataset




