import sys
sys.path.append("..")
from base import expEval
from sklearn.metrics import mean_squared_error
import math


class expEvalRMSE(expEval):
    name = "RMSE"
    desc = "Root Mean Squared Error"

    def __init__(self):
        super(expEvalRMSE,self).__init__()

    def evaluate(self,trueValues,preValues,outputs=None):
        if len(trueValues)!=len(preValues):
            raise Exception('evaluation[RMSE]: dim mismatch of label and prelabel')
        value = math.sqrt(mean_squared_error(trueValues, preValues))
        # append the lastest value to the value list
        # self.value.append(value)
