#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement the Preceptron Algorithm
"""
import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = []
        self.errors_ = []

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(scale=4, size=len(X[0]))
        
        for i in range(self.n_iter):
            error_count = 0
            for j in range(y.size):
                h = self.net_input(X[j])
                f = self.net_output(h)
                w_update = self.eta * np.multiply((y[j][0] - f), X[j])  
                self.w_ = np.add(self.w_, w_update)
                if (y[j][0] - f) != 0:
                    error_count += 1
            self.errors_.append(error_count)

    def net_input(self, X):
        """Calculate net input"""
        if X.size == self.w_.size or or X[0].size == self.w_.size:
            return np.dot(X, self.w_)
        else:
            return (np.dot(X, self.w_[1:]) + self.w_[0])
    
    def net_output(self, h):
        """Threshold/Calculate net output"""
        if h > 0:
            return 1
        elif h < 0:
            return -1
    
    def predict(self, X):
        """Return class label after unit step"""
        h = self.net_input(X)
        labels = []
        for input in h:
            labels.append(self.net_output(input))
        labels = np.array(labels)
        return labels
        
