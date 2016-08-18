import os
import pickle
import json
import csv
import StringIO
import hashlib
from itertools import chain
from path import Path

###############################################################################
def savepkl(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        output.close()

def loadpkl(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)
        f.close()

###############################################################################
def dicts2csv(source, output_file):
    def build_row(dict_obj, keys):
        return [dict_obj.get(k) for k in keys]
    keys = sorted(set(chain.from_iterable([o.keys() for o in source])))
    rows = [build_row(d, keys) for d in source]

    cw = csv.writer(output_file)
    cw.writerow(keys)
    for row in rows:
        cw.writerow([c.encode('utf-8') if isinstance(c, str) or isinstance(c, unicode) else c for c in row])

###############################################################################
def merge_result(ds,setting,folder="results"):
    result_base = '/'.join([folder,ds.name,setting.name,''])
    csv = open(ds.name+"-"+setting.name+'.csv','w')
    rows=[]
    for root,dirs,files in os.walk(result_base):
        for f in files:
            if f.startswith('.'):
                continue
            setting = loadpkl(root+'/'+f)
            evals = setting.evals
            n_evals = len(evals)
            n_results = len(evals[0].value)
            for i in xrange(n_results):
                row = setting.abstract.copy()
                for j in xrange(n_evals):
                    row.update({evals[j].name:evals[j].value[i]})
                rows.append(row)
    dicts2csv(rows,csv)

def merge_all(folder="results"):
    if not os.path.exists(folder):
        print "folder %s doesn't exist" %folder
        return 
    for ds in Path(folder).dirs():
        for setting in Path(ds).dirs():
            c1,c2,c3 = setting.split('\\')
            filename = c2+"-"+c3+'.csv'
            rows=[]
            csv = open(filename,'w')
            for f in Path(setting).files("*.pkl"):
                rst = loadpkl(f)
                evals = rst.evals
                n_evals = len(evals)
                n_results = len(evals[0].value)
                for i in xrange(n_results):
                    row = rst.abstract.copy()
                    for j in xrange(n_evals):
                        row.update({evals[j].name:evals[j].value[i]})
                    rows.append(row)
            dicts2csv(rows,csv)
            print "result file %s generated." %filename
        csv.close()

###############################################################################
def result_path(ds,model,setting,folder="results",setting_parameters=False):
    def check_result_path(dataset_name,setting_name,para_summary):
        md5 = hashlib.md5(para_summary).hexdigest()
        result_base = '/'.join([folder,dataset_name,setting_name,''])
        result_file = result_base + md5 + '.pkl' 
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(folder + '/' + dataset_name):
            os.makedirs(folder + '/' + dataset_name)
        if not os.path.exists(result_base):
            os.makedirs(result_base)
        return result_file
    def build_summary(model_name,dataset_dict,model_dict,setting_dict={}):
        summary =model_name + '_' +json.dumps(dataset_dict) + "_" + json.dumps(model_dict)
        if len(setting_dict) != 0:
            summary = summary + "_" + json.dumps(setting_dict)
        return summary

    """
    model_dict  = model.__dict__.copy()
    for key in ['groups']:
        if key in model_dict:
            del model_dict[key]

    if setting_para:
        setting_dict = setting.__dict__
        for key in ['evals','model','data','abstract','label','timeTrain','timeTrain','timeTotal']:
            if key in setting_dict:
                del setting_dict[key]
    """
    if setting_parameters:
        summary = build_summary(model.name,ds.__dict__,model.__dict__,setting_dict=setting.__dict__)
    else:
        summary = build_summary(model.name,ds.__dict__,model.__dict__)

    return check_result_path(ds.name,setting.name,summary)





