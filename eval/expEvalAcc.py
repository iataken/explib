import sys
sys.path.append("..")
from base import expEval
from sklearn.metrics import accuracy_score


class expEvalAcc(expEval):
    name = "Acc"
    desc = "Accuracy"

    def __init__(self):
        super(expEvalAcc,self).__init__()

    def evaluate(self,trueValues,preValues,outputs=None):
        if len(trueValues)!=len(preValues):
            raise Exception('evaluation[RMSE]: dim mismatch of label and prelabel')
        value = accuracy_score(trueValues,preValues)
        # append the lastest value to the value list
        # self.value.append(value)
