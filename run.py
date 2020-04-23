# Libraries
import numpy as np
import sys
import os, shutil
import sklearn.datasets as skdata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from copy import deepcopy
from joblib import Parallel, delayed
import pandas as pd

# Internal imports
from pynsgp.Nodes.BaseNode import Node
from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.Fitness.FitnessFunction import SymbolicRegressionFitness
from pynsgp.Evolution.Evolution import pyNSGP
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

np.random.seed(42)

problem = sys.argv[1]
n_runs = int(sys.argv[2])
use_model = sys.argv[3].lower().startswith('m')

### LOAD DATASET ###

datasets_path = './datasets'


def load_dataset_from_disk(dataset_name):
    Xy = np.genfromtxt(datasets_path + '/' + dataset_name)
    X = Xy[:, :-1]
    y = Xy[:, -1]
    return X, y


dataset_loaders = {
    'airfoil' : load_dataset_from_disk,
    'california': skdata.fetch_california_housing,
    'boston': skdata.load_boston,
    'diabetes': skdata.load_diabetes,
    'dowchemical' : load_dataset_from_disk,
    'tower' : load_dataset_from_disk,
    'linnerud': skdata.load_linnerud,
    'yacht': load_dataset_from_disk,
    'energyheating': load_dataset_from_disk,
    'energycooling': load_dataset_from_disk,
    'keijzer6' : load_dataset_from_disk,
    'korns12' : load_dataset_from_disk,
    'vladislavleva4' : load_dataset_from_disk,
    'nguyen7' : load_dataset_from_disk,
    'pagie1' : load_dataset_from_disk,
}

dataset_loader = dataset_loaders[problem]
if dataset_loader.__name__ == load_dataset_from_disk.__name__:
    X, y = dataset_loader(problem + '_full.dat')
elif dataset_loader.__module__ == skdata.load_boston.__module__:
    X, y = dataset_loader(return_X_y=True)
else:
    raise ValueError('Unrecognized dataset loader name (maybe not specified yet for this dataset?)',
                     dataset_loader.__name__)

X = scale(X)
y_std = np.std(y)
y = scale(y)

### SETUP DIRECTORY ###

directory = problem + '_' + str(use_model)

if not os.path.exists(directory):
    os.makedirs(directory)
else:
    # cleanup
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


### RUN ###

def performOneRun(seed):
    # Take a dataset split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

    nsgp = NSGP(pop_size=1000, max_generations=100, verbose=False, max_tree_size=100,
                crossover_rate=0.9, mutation_rate=0.0, op_mutation_rate=1.0, min_depth=1,
                initialization_max_tree_height=6,
                tournament_size=2, use_linear_scaling=True, use_erc=True, use_interpretability_model=use_model,
                functions=[AddNode(), SubNode(), MulNode(), DivNode(), SinNode(), CosNode(), ExpNode(), LogNode()])
    nsgp.fit(X_train, y_train)
    pop = nsgp.get_population()

    pop_unique = set()
    df = pd.DataFrame(columns=["formula", "intercept", "scale", "mse_train", "mse_validation", "mse_test", "n_nodes", "phi"])
    fitness_with = SymbolicRegressionFitness(X_train, y_train, use_interpretability_model=True)
    fitness_without = SymbolicRegressionFitness(X_train, y_train, use_interpretability_model=False)
    for model in pop:
        model_str = str(model.GetHumanExpression())
        ls_a = model.ls_a
        ls_b = model.ls_b

        scaled_train_output = ls_a + ls_b * model.GetOutput(X_train)
        model_train_err = np.mean(np.square(y_train - scaled_train_output))

        scaled_validation_output = ls_a + ls_b * model.GetOutput(X_validation)
        model_validation_err = np.mean(np.square(y_validation - scaled_validation_output))

        scaled_test_output = ls_a + ls_b * model.GetOutput(X_test)
        model_test_err = np.mean(np.square(y_test - scaled_test_output))

        if model_str not in pop_unique:
            pop_unique.add(model_str)
            fitness_without.Evaluate(model)
            n_nodes = model.objectives[1]
            fitness_with.Evaluate(model)
            phi = model.objectives[1]
            df.loc[len(df) + 1] = [model_str, ls_a, ls_b, model_train_err, model_validation_err, model_test_err, n_nodes,
                                   phi]
    df.to_csv(directory + '/' + str(seed), index=False)


# performOneRun(1)
Parallel(n_jobs=n_runs)(delayed(performOneRun)(seed) for seed in range(1, n_runs + 1))
