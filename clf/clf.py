from sklearn.svm import SVC
import sys
sys.path.append("..")
from base import expModel
import nimfa
from scipy.sparse import csr_matrix
from bunch import Bunch

class expModelSVC(expModel):
    name = 'SVC'
    desc = 'support vector classifier'

    def __init__(self,C = 1.0):
        super(expModelSVC,self).__init__()
        self.C = C

    def predict(self,train_data,train_label,test_data):
        model = SVC(C = self.C)
        # training 
        model.fit(train_data,train_label)
        # predict
        predicted = model.predict(test_data)
        return predicted

class expModelNMF(expModel):
    name = 'NMF'
    desc = 'NMF'

    def __init__(self,beta=1e-3,rank=10):
        super(expModelNMF,self).__init__()
        self.beta = beta
        self.rank = rank

    def predict(self,train_data,train_label,test_data):
        matrix = train_data
        # nmf = nimfa.Snmf(matrix,beta=self.beta,rank=self.rank)
        nmf = nimfa.Nmf(matrix)
        nmf()
        w_array = csr_matrix(nmf.W).toarray()
        h_array = csr_matrix(nmf.H).toarray()
        result_matrix = w_array.dot(h_array)

        return result_matrix


