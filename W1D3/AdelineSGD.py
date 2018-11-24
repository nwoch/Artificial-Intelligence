#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement the Adaline learning machine with gradient descent training

"""

import numpy as np

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier with stochastic gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset. (Epochs!)
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = []
        self.cost_ = [] 

    def fit(self, X, y):
        """ Fit training data using a stochastic update rule.

        Parameters
        ----------
        X : n_samples by n_features data matrix
        y : a vector of labels of length n_samples

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(scale=1, size=len(X[0]))
    
        for i in range(self.n_iter):
            total_cost = 0
            for j in range(y.size):
                error_calc = y[j][0] - self.net_input(X[j])
                # Objective function of w
                obj_w = np.square(error_calc)
                # Gradient(Y - (x*w))^2 = Gradient(obj(w)) = -2x(Y-(x*w))
                gradient_obj_w = -2 * np.multiply((X[j]), error_calc)
                w_update = self.eta * gradient_obj_w
                total_cost += obj_w
                self.w_ = np.subtract(self.w_, w_update)
            self.cost_.append(total_cost)
        return self
            
    def net_input(self, X):
        """Calculate net input for a feature vector of length n_features"""
        if X.size == self.w_.size or X[0].size == self.w_.size:
            return (np.dot(X, self.w_) / np.linalg.norm(self.w_))
        else:
            return ((np.dot(X, self.w_[1:]) + self.w_[0]) / np.linalg.norm(self.w_))
    
    def net_output(self, h):
        """Threshold/Calculate net output"""
        if h > 0:
            return 1
        elif h < 0:
            return -1

    def predict(self, X):
        """Return class label for the feature vector x"""
        h = self.net_input(X)
        labels = []
        for input in h:
            labels.append(self.net_output(input))
        labels = np.array(labels)
        return labels
        

