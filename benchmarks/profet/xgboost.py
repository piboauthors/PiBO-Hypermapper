import numpy as np
import sys
sys.path.append('benchmarks/profet/emukit')
from emukit.examples.profet.meta_benchmarks import (meta_fcnet, meta_svm, meta_xgboost)
import json
import os
sys.path.append('scripts')
import bopro
import datetime

def define_function(function_id, function_family, path_to_files):
    """
    Get black box function and parameter space from one of the supported benchmarks.
    """
    fname_objective = "%s/samples/%s/sample_objective_%d.pkl" % (path_to_files, function_family, function_id)
    fname_cost="%s/samples/%s/sample_cost_%d.pkl" % (path_to_files, function_family, function_id)

    if function_family == "svm":
        fcn, parameter_space = meta_svm(fname_objective=fname_objective, fname_cost=fname_cost, noise=False)
    elif function_family == "xgboost":
        fcn, parameter_space = meta_xgboost(fname_objective=fname_objective, fname_cost=fname_cost, noise=False)
    elif function_family == "fcnet":
        fcn, parameter_space = meta_fcnet(fname_objective=fname_objective, fname_cost=fname_cost, noise=False)
    return fcn, parameter_space

bbox_function = None
def black_box_function_wrapper(configuration):
    X = np.array([[configuration[param] for param in configuration]])
    result = bbox_function(X)
    try:
        y, c = result
    except ValueError:
        y = result
    return float(y[0,0])

if __name__ == "__main__":
    function_id = 42
    function_family = "xgboost"

    path_to_files = "benchmarks/profet/profet_data"
    fcn, parameter_space = define_function(function_id, function_family, path_to_files)
    bbox_function = fcn
    parameters_file = "benchmarks/profet/xgboost_42_scenario.json"
    bopro.optimize(parameters_file, black_box_function_wrapper)
