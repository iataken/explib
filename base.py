import abc
import numpy as np
from bunch import Bunch
import pickle
import os
import sys
import copy
import warnings
from utils import savepkl
import six



class expBase(object):
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

###############################################################################
class expSetting(expBase):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self.timeTotal = []
        self.timeTrain = []
        self.timeTest = []
        self.abstract={}
    def setModel(self,model):
        self.model = model
    def setEval(self,evalMethods):
        self.evals =  evalMethods
    def update_abstract(self,abstract):
        self.abstract.update(abstract)
    @abc.abstractmethod
    def evaluate(self):
        """
        excute evaluation
        to be implemented in subclasses
        """
        return

###############################################################################
class expDataset(expBase):
    __metaclass__ = abc.ABCMeta
    #def __init__(self,name,desc=""):
        #self.abstract={}
    @abc.abstractmethod
    def load(self):
        """
        load dataset
        to be implemented in subclasses
        """
        return

###############################################################################
class expEval(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self.value = []
        self.stat = Bunch()

    def mean(self):
        meanVal = np.mean(self.value)
        self.stat.mean = meanVal
        return meanVal

    def std(self):
        stdVal = np.std(self.value)
        self.stat.std = stdVal
        return stdVal
    @abc.abstractmethod
    def evaluate(self,trueLabels,preLables):
        """
        excute evaluation to get values
        to be implemented in subclasses
        """

###############################################################################
class expModel(expBase):
    __metaclass__ = abc.ABCMeta
    #def __init__(self):
        #self.name = name
        #self.desc = desc
        #self.timeTotal = 0
        #self.timeTrain = 0
        #self.timeTest= 0 
        #self.expSetting = None
        #self.abstract={}

    @abc.abstractmethod
    def predict(self,train_data,train_label,test_data):
        """
        excute classification
        to be implemented in subclasses
        """
        #self.abstract['classifier'] = self.name
        return 

###############################################################################
class expProfile(object):
    def __init__(self,expDataset,expSetting,expModel,expEvals,resultPath):
        self.dataset = expDataset
        self.setting = expSetting
        self.model = expModel
        self.evals = expEvals
        self.resultPath = resultPath

    def run(self, overwrite=False):
        if not overwrite and os.path.exists(self.resultPath):
            print "[expProfile]: result file already exists, explib will skip this run."
            return

        ## add parameters to the abstract dictionary 
        self.setting.update_abstract(self.dataset.__dict__)
        self.setting.update_abstract(self.model.__dict__)
        self.setting.update_abstract(self.setting.__dict__)
        self.setting.update_abstract({"model":self.model.name})
        ## build up the setting module
        self.setting.model = self.model
        self.setting.evals = self.evals
        self.setting.data  = self.dataset.load()

        ## begin evaluation
        self.setting.evaluate()
        self.setting.update_abstract({"timeTrain":self.setting.timeTrain})
        self.setting.update_abstract({"timeTest":self.setting.timeTest})
        self.setting.update_abstract({"timeTotal":self.setting.timeTotal})


        ## clean up dataset and model to save space 
        self.setting.data    = None
        self.setting.model   = None 

        ## remove big chunk from abstract 
        for key in ["abstract"]:
            self.setting.abstract.pop(key)
        
        ## save pkl
        try: 
            savepkl(self.setting,self.resultPath)
        except IOError:
            print "[expProfile:run]:IOError"
