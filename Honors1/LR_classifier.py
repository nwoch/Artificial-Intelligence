# Honor Code Upheld: Nicole Woch

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:07:10 2018

@author: Nicole
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
import copy

class LogisticRegressionClassifier:
    
    def __init__(self):
        # create 3 classifiers
        self.lr1 = LogisticRegression(n_iter=50, eta=3.923)
        self.lr2 = LogisticRegression(n_iter=50, eta=3.923)
        self.lr3 = LogisticRegression(n_iter=50, eta=3.923) 
        self.X = None
        self.y_labels = None
    
    def classify(self):
        confidence1 = self.lr1.multiclass_confidence(self.X)
        confidence2 = self.lr2.multiclass_confidence(self.X)
        confidence3 = self.lr3.multiclass_confidence(self.X)
        labels = []
        for i in range(len(confidence1)): 
            if confidence1[i] >= confidence2[i] and confidence1[i] >= confidence3[i]:
                label = 'Iris-setosa'
            elif confidence2[i] >= confidence1[i] and confidence2[i] >= confidence3[i]:
                label = 'Iris-versicolor'
            elif confidence3[i] >= confidence1[i] and confidence3[i] >= confidence2[i]:
                label = 'Iris-virginica'
            labels.append(label)
        labels = np.array(labels)
        return labels
        
    def calc_classification_error(self, predicted, y):
        errors = 0
        for i in range(len(y)):
            if y[i][0] != predicted[i]:
                errors += 1
        print("Errors: ", errors, "/", len(y))
        return errors / len(y)
             
    def train(self):
        # load Iris dataset from UCI repository
        df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                'machine-learning-databases/iris/iris.data', header=None)
        # extract sepal length and petal length from data vectors
        X_vals = df.iloc[0:150, 0:4].values
        self.X = np.insert(X_vals, 0, 1, axis=1)
        # extract flower labels from data vectors
        self.y_labels = df.iloc[0:150, [4]].values
        
        for i in range(3):
            y_num_labels = copy.deepcopy(self.y_labels)
            if i == 0:
                lr = self.lr1
                f_type = 'Iris-setosa'
            elif i == 1:
                lr = self.lr2
                f_type = 'Iris-versicolor'
            elif i == 2:
                lr = self.lr3
                f_type = 'Iris-virginica'
            for j in range(y_num_labels.size):
                if y_num_labels[j] == f_type:
                    y_num_labels[j] = 1
                else:
                    y_num_labels[j] = 0 
                   
            lr.fit(self.X, y_num_labels)
            self.plot_errors(lr)
            print("Classifier", i+1, "error rate: ", 
                  self.calc_classification_error(lr.predict(self.X), y_num_labels))
    
    def plot_errors(self, lr):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ax[0].plot(range(1, len(lr.errors_) + 1), np.log10(lr.errors_), marker='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log(Error)')
        ax[0].set_title('Logistic Regression - Learning rate 3.923')
        ax[1].plot(range(1, len(lr.errors_) + 1), lr.errors_, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Error')
        ax[1].set_title('Logistic Regression - Learning rate 3.923')
        
        
def main():
  multi_classifier = LogisticRegressionClassifier()
  multi_classifier.train()
  error_rate = multi_classifier.calc_classification_error(multi_classifier.classify(), multi_classifier.y_labels)
  print("Final classification error rate: ", error_rate)
  
if __name__== "__main__":
  main()
