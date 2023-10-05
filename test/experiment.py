# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

import csv
import sys
from time import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from pyoperon.sklearn import SymbolicRegressor
from pyoperon import MSE, R2, InfixFormatter, FitLeastSquares

from pmlb import fetch_data, dataset_names, classification_dataset_names, regression_dataset_names
#print(regression_dataset_names)

sys.path.insert(1, '../')

# Configure all `numpy` warning messages to be suppressed.
np.seterr(all='ignore')
warnings.simplefilter('ignore')

# Reproducible random state.
seed = 0
np.random.seed(seed)

airfoil = []
# with open('./examples/core/program/evolution/airfoil.csv', 'r') as f:
with open('./airfoil.csv', 'r') as f:
    csv_file = csv.reader(f)
    for row in csv_file:
        airfoil.append(row)
airfoil = np.array(airfoil[1:], dtype=np.float64)
inputs_, targets_ = airfoil[:, :-1], airfoil[:, -1].reshape(-1, 1)
n_fc = len(inputs_)
cases_train = np.random.choice(n_fc, size=1000, replace=False)

# breiman_1 = []
# # with open('./examples/core/program/evolution/breiman_1.csv', 'r') as f:
# with open('./breiman_1.csv', 'r') as f:
#     csv_file = csv.reader(f)
#     for row in csv_file:
#         breiman_1.append(row)
# breiman_1 = np.array(breiman_1[1:], dtype=np.float64)
# inputs_, targets_ = breiman_1[:, :-1], breiman_1[:, -1].reshape(-1, 1)
# n_fc = len(inputs_)
# cases_train = np.random.choice(n_fc, size=5000, replace=False)

# chemical_1 = []
# # with open('./examples/core/program/evolution/chemical_1.csv', 'r') as f:
# with open('./chemical_1.csv', 'r') as f:
#     csv_file = csv.reader(f)
#     for row in csv_file:
#         chemical_1.append(row)
# chemical_1 = np.array(chemical_1[1:], dtype=np.float64)
# inputs_, targets_ = chemical_1[:, :-1], chemical_1[:, -1].reshape(-1, 1)
# n_fc = len(inputs_)
# cases_train = np.random.choice(n_fc, size=711, replace=False)

# concrete = []
# # with open('./examples/core/program/evolution/concrete.csv', 'r') as f:
# with open('./concrete.csv', 'r') as f:
#     csv_file = csv.reader(f)
#     for row in csv_file:
#         concrete.append(row)
# concrete = np.array(concrete[1:], dtype=np.float64)
# inputs_, targets_ = concrete[:, :-1], concrete[:, -1].reshape(-1, 1)
# n_fc = len(inputs_)
# cases_train = np.random.choice(n_fc, size=1000, replace=False)

# friedman_1 = []
# # with open('./examples/core/program/evolution/friedman_1.csv', 'r') as f:
# with open('./friedman_1.csv', 'r') as f:
#     csv_file = csv.reader(f)
#     for row in csv_file:
#         friedman_1.append(row)
# friedman_1 = np.array(friedman_1[1:], dtype=np.float64)
# inputs_, targets_ = friedman_1[:, :-1], friedman_1[:, -1].reshape(-1, 1)
# n_fc = len(inputs_)
# cases_train = np.random.choice(n_fc, size=5000, replace=False)

# friedman_2 = []
# # with open('./examples/core/program/evolution/friedman_2.csv', 'r') as f:
# with open('./friedman_2.csv', 'r') as f:
#     csv_file = csv.reader(f)
#     for row in csv_file:
#         friedman_2.append(row)
# friedman_2 = np.array(friedman_2[1:], dtype=np.float64)
# inputs_, targets_ = friedman_2[:, :-1], friedman_2[:, -1].reshape(-1, 1)
# n_fc = len(inputs_)
# cases_train = np.random.choice(n_fc, size=5000, replace=False)

# gp_challenge = []
# # with open('./examples/core/program/evolution/gp_challenge.csv', 'r') as f:
# with open('./gp_challenge.csv', 'r') as f:
#     csv_file = csv.reader(f)
#     for row in csv_file:
#         gp_challenge.append(row)
# gp_challenge = np.array(gp_challenge[1:], dtype=np.float64)
# inputs_, targets_ = gp_challenge[:, :-1], gp_challenge[:, -1].reshape(-1, 1)
# n_fc = len(inputs_)
# cases_train = np.random.choice(n_fc, size=5000, replace=False)

# poly_10 = []
# # with open('./examples/core/program/evolution/poly_10.csv', 'r') as f:
# with open('./poly_10.csv', 'r') as f:
#     csv_file = csv.reader(f)
#     for row in csv_file:
#         poly_10.append(row)
# poly_10 = np.array(poly_10[1:], dtype=np.float64)
# inputs_, targets_ = poly_10[:, :-1], poly_10[:, -1].reshape(-1, 1)
# n_fc = len(inputs_)
# cases_train = np.random.choice(n_fc, size=250, replace=False)

# pagie_1 = []
# # with open('./examples/core/program/evolution/pagie_1.csv', 'r') as f:
# with open('./pagie_1.csv', 'r') as f:
#     csv_file = csv.reader(f)
#     for row in csv_file:
#         pagie_1.append(row)
# pagie_1 = np.array(pagie_1[1:], dtype=np.float64)
# inputs_, targets_ = pagie_1[:, :-1], pagie_1[:, -1].reshape(-1, 1)
# n_fc = len(inputs_)
# cases_train = np.arange(676)

cases_test = np.delete(np.arange(n_fc), cases_train)
inputs_, inputs_test_ = inputs_[cases_train], inputs_[cases_test]
targets, targets_test = targets_[cases_train], targets_[cases_test]

standard = StandardScaler()
inputs = standard.fit_transform(inputs_)
# inputs = inputs_
# targets = f(inputs_)

inputs_test = standard.transform(inputs_test_)
# inputs_test = inputs_test_
# targets_test = f(inputs_test_)

X_train = inputs
y_train = targets
X_test = inputs_test
y_test = targets_test

reg = SymbolicRegressor(
      # allowed_symbols='add,mul,aq,sin,constant,variable',
      # allowed_symbols='add,mul,aq,sin,fmin,fmax,constant,variable',
      # allowed_symbols='sin,tanh,exp,log,sqrt,add,mul,aq,constant,variable',
      allowed_symbols='sin,tanh,exp,log,sqrt,add,mul,aq,fmin,fmax,constant,variable',
      population_size=512,
      generations=500,
      tournament_size=5,
      offspring_generator='basic',
      max_length=50,
      max_depth=10,
      initialization_method='btc',
      n_threads=24,
      # objectives = ['r2'],
      # objectives = ['mse'],
      objectives = ['r2', 'length'],
      # objectives = ['mse', 'length'],
      # objectives = ['mse', 'diversity'],
      epsilon = 1e-10,
      # reinserter='keep-best',
      reinserter='replace-worst',
      max_evaluations=int(1e15),
      symbolic_mode=False,
      local_iterations=10,
      # local_iterations=0,
      random_state=seed
      )

# Save current time to be able to measure the duration 
# of the following search procedure.
t0 = time()

reg.fit(X_train, y_train)

# Information about the search procedure.
t = time() - t0
print(f'\nDuration of search procedure: {t:.6f} seconds '
      f'({t/60:.6f} minutes---or {t/3600:.6f} hours)\n')

# print(reg.get_model_string(10))
print(reg.stats_)

# for model, model_vars in reg.pareto_front_:
#     y_pred_train = reg.evaluate_model(model, X_train)
#     y_pred_test = reg.evaluate_model(model, X_test)

#     scale, offset = FitLeastSquares(y_pred_train, y_train)
#     y_pred_train = scale * y_pred_train + offset
#     y_pred_test = scale * y_pred_test + offset

#     print(InfixFormatter.Format(model, model_vars, 3), model.Length, r2_score(y_train, y_pred_train), r2_score(y_train, y_pred_train))

r2 = R2()
mse = MSE() 

y_pred_train = reg.predict(X_train)
print('r2 train (sklearn.r2_score): ', r2_score(y_train, y_pred_train))
print('r2 train (operon.r2): ', -r2(y_pred_train, y_train))
print('MSE train (operon.MSE): ', mse(y_pred_train, y_train))

y_pred_test = reg.predict(X_test)
print('r2 test (sklearn.r2_score): ', r2_score(y_test, y_pred_test))
print('MSE test (operon.MSE): ', mse(y_test, y_pred_test))