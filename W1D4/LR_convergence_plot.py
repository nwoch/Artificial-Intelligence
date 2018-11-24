# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:15:59 2018

@author: Nicole
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
import copy

def train():
    # load Iris dataset from UCI repository
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', header=None)
    # extract sepal length and petal length from data vectors
    X_vals = df.iloc[0:150, [0, 2]].values
    X = np.insert(X_vals, 0, 1, axis=1)
    # extract flower labels from data vectors
    y_labels = df.iloc[0:150, [4]].values
    
    for i in range(3):
        lr = LogisticRegression(n_iter=50, eta=0.001)
        y = copy.deepcopy(y_labels)
        if i == 0:
            f_type = 'Iris-setosa'
        elif i == 1:
            f_type = 'Iris-versicolor'
        elif i == 2:
            f_type = 'Iris-virginica'
        for i in range(y.size):
            if y[i] == f_type:
                y[i] = 1
            else:
                y[i] = 0 
               
        lr.fit(X, y)
        plot_errors(lr)
    
def plot_errors(lr):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].plot(range(1, len(lr.errors_) + 1), np.log10(lr.errors_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Error)')
    ax[0].set_title('Logistic Regression - Learning rate 0.001')
    ax[1].plot(range(1, len(lr.errors_) + 1), lr.errors_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Error')
    ax[1].set_title('Logistic Regression - Learning rate 0.001')
    
    
train()
