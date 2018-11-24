#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 07:22:03 2018

Sample code for plotting convergence rate of AdalineGD
@author: richard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AdelineGD import AdalineGD
from AdelineSGD import AdalineSGD
from plot_decision_regions import plot_decision_regions


#Load Iris dataset from UCI repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
# extract sepal length and petal length from data vectors
X = df.iloc[0:150, [0, 2]].values
X = np.insert(X, 0, 1, axis=1)

y = df.iloc[0:150, [4]].values
for i in range(y.size):
    if y[i] == 'Iris-setosa':
        y[i] = 1
    elif y[i] == 'Iris-virginica' or y[i] == 'Iris-versicolor':
        y[i] = -1   
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.0001)
ada1.fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.0001')

ax[1].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')


y = df.iloc[0:150, [4]].values
for i in range(y.size):
    if y[i] == 'Iris-versicolor':
        y[i] = 1
    elif y[i] == 'Iris-setosa' or y[i] == 'Iris-virginica':
        y[i] = -1   
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.0001)
ada1.fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.0001')

ax[1].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')


y = df.iloc[0:150, [4]].values
for i in range(y.size):
    if y[i] == 'Iris-virginica':
        y[i] = 1
    elif y[i] == 'Iris-setosa' or y[i] == 'Iris-versicolor':
        y[i] = -1   
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.0001)
ada1.fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.0001')

ax[1].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

#plt.savefig('images/02_11.png', dpi=300)
plt.show()

##decision-region plotting
#test_X = df.iloc[0:150, [0, 2]].values
#
#plot_decision_regions(test_X, y, ada1)
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
#plt.legend(loc='upper left')
#
#plot_decision_regions(test_X, y, ada2)
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
#plt.legend(loc='upper left')
##
### plt.savefig('images/02_14.png', dpi=300)
#plt.show()