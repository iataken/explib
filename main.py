from base import expProfile
from clf.clf import expModelNMF
from dataset.ds import expDatasetDBLP
from eval.expEvalMMR import expEvalMMR
from eval.expEvalPATK import expEvalPATK
from setting.setting import expSettingMMR
from utils import merge_all,result_path

# for rank in [7]:
#     for beta in[1e-4]:
# expDataset,expSetting,expModel,expEvals,resultPath
expDataset = expDatasetDBLP()
expSetting = expSettingMMR()
expModel = expModelNMF()
expEvals = [expEvalPATK()]
resultPath = result_path(expDataset,expModel,expSetting)
a = expProfile(expDataset,expSetting,expModel,expEvals,resultPath)
a.run(overwrite=True)
merge_all()