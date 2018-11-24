# Honor Code Upheld: Nicole Woch

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment CP365 W1D4 : Logistic Regression Machine with Regularization

"""

import numpy as np

class LogisticRegression(object):
    """

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    reg : non-negative float
        Regularization term.  reg == 0  is no regularization

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, reg=0, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.reg = reg
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
        IMPORTANT: Labels for Logistic Regression must be 1 or 0!

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(scale=1, size=len(X[0]))
        
        for i in range(self.n_iter):
            total_error = 0
            w_sum = 0
            for j in range(y.size):
                h = self.net_input(X[j])
                sigmoid = self.activation(h)
                # Objective function of w
                obj_w = -((y[j][0] * np.log(sigmoid)) + ((1-y[j][0]) * np.log(1-sigmoid)))
                # Add regularization term
                obj_w += (self.reg * (np.square(np.linalg.norm(self.w_)) / 2))
                # Gradient(obj(w))
                gradient_obj_w = -((y[j][0] * np.true_divide(X[j], (1 + np.exp(h)))) 
                                 + ((y[j][0]-1) * X[j] * sigmoid))
                # Delta w with learning rate and regularization term 
                w_update = self.eta * (gradient_obj_w + (self.reg * self.w_))
                # Update weight vector sum
                w_sum = np.add(w_sum, w_update)
                # Update error
                total_error += obj_w  
            self.w_ = np.subtract(self.w_, np.true_divide(w_sum, y.size))
            self.errors_.append(total_error)
        return self
       
    def net_input(self, X):
        """Calculate net input"""
        if X.size == self.w_.size or X[0].size == self.w_.size:
            return (np.dot(X, self.w_) / np.linalg.norm(self.w_))
        else:
            return ((np.dot(X, self.w_[1:]) + self.w_[0]) / np.linalg.norm(self.w_))
    
    def net_output(self, p):
        """Calculate net output based on probability"""
        if p > 0.5:
            return 1
        elif p <= 0.5:
            return 0
    
    def multiclass_confidence(self, X):
        """Return confidence measures for a multi-class classifier"""
        h = self.net_input(X)
        confidence_measures = []
        for input in h:
            confidence_measures.append(self.activation(input))
        return confidence_measures

    def predict(self, X):
        """Return class labels after unit step"""
        h = self.net_input(X)
        labels = []
        for input in h:
            sigmoid = self.activation(input)
            labels.append(self.net_output(sigmoid))
        labels = np.array(labels)
        return labels
        
    def activation(self, x):
        """Sigmoid function"""
        return (1 / (1 + np.exp(-x)))

