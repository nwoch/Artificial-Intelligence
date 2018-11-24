#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:38:43 2018

@author: richard
"""

#Class Code  W1D2

#Useful imports
#numpy for linear algebra calculations
import numpy as np
#pandas for data frames and IO
import pandas as pd
#plotting tools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Perceptron import Perceptron
from plot_decision_regions import plot_decision_regions


#Load Iris dataset from UCI repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
#Tail gives the last few rows of the dataframe
df.tail()
# extract sepal length and petal length from first 100 data vectors
X = df.iloc[0:100, [0, 2]].values
#Plot labeled data. Note the first 50 instances are of class setosa
#and the second 50 instances are of class versicolor
# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()

#assuming your Perceptron class computes the number of errors in each Epoch
#and stores the values in an array self.errors_, the following code will
#create a plot of the errors as a function of number of Epochs

ppn = Perceptron(eta=0.01, n_iter=10)
X = np.insert(X, 0, 1, axis=1)
y = df.iloc[0:100, [4]].values
for i in range(y.size):
    if y[i] == 'Iris-setosa':
        y[i] = 1
    elif y[i] == 'Iris-versicolor':
        y[i] = -1    
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

test_X = df.iloc[0:100, [0, 2]].values
plot_decision_regions(test_X, y, ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
## plt.savefig('images/02_08.png', dpi=300)
plt.show()