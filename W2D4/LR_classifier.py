# Honor Code Upheld: Nicole Woch

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:07:10 2018

@author: Nicole
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from LogisticRegressionReg import LogisticRegression
import copy

class LogisticRegressionClassifier:
    
    def __init__(self, eta1 = 2.651, reg1 = 0.0033):
        # create 3 classifiers
        self.lr1 = LogisticRegression(eta=eta1, n_iter=50)
        self.lr2 = LogisticRegression(eta=eta1, n_iter=50)
        self.lr3 = LogisticRegression(eta=eta1, n_iter=50) 
        self.lr1_reg = LogisticRegression(eta=eta1, n_iter=50, reg=reg1)
        self.lr2_reg = LogisticRegression(eta=eta1, n_iter=50, reg=reg1)
        self.lr3_reg = LogisticRegression(eta=eta1, n_iter=50, reg=reg1)
        self.X = None
        self.X_test = None
        self.y_test = None
        self.y_labels = None
    
    def classify(self, lr1, lr2, lr3):
        confidence1 = lr1.multiclass_confidence(self.X_test)
        confidence2 = lr2.multiclass_confidence(self.X_test)
        confidence3 = lr3.multiclass_confidence(self.X_test)
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
        return errors / len(y)
    
    def load_data(self):
        df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                'machine-learning-databases/iris/iris.data', header=None)
        df = shuffle(df)
        self.X_test = df.iloc[99:150, 0:4].values
        self.y_test = df.iloc[99:150, [4]].values
        return df
             
    def train(self, df, start_index, end_index):
        X_vals = df.iloc[start_index:end_index, 0:4].values
        self.X = np.insert(X_vals, 0, 1, axis=1)
        self.y_labels = df.iloc[start_index:end_index, [4]].values
        
        for i in range(3):
            y_num_labels = copy.deepcopy(self.y_labels)
            if i == 0:
                lr = self.lr1
                lr_reg = self.lr1_reg
                f_type = 'Iris-setosa'
            elif i == 1:
                lr = self.lr2
                lr_reg = self.lr2_reg
                f_type = 'Iris-versicolor'
            elif i == 2:
                lr = self.lr3
                lr_reg = self.lr3_reg
                f_type = 'Iris-virginica'
            for j in range(y_num_labels.size):
                if y_num_labels[j] == f_type:
                    y_num_labels[j] = 1
                else:
                    y_num_labels[j] = 0 
                   
            lr.fit(self.X, y_num_labels)
            lr_reg.fit(self.X, y_num_labels)
    
    def plot_errors(self, lr):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ax[0].plot(range(1, len(lr.errors_) + 1), np.log10(lr.errors_), marker='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log(Error)')
        ax[0].set_title('Logistic Regression - Learning rate 2.651')
        ax[1].plot(range(1, len(lr.errors_) + 1), lr.errors_, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Error')
        ax[1].set_title('Logistic Regression - Learning rate 2.651')
        
        
def main():
    multi_classifier = LogisticRegressionClassifier(eta1=0.8, reg1=0.001)
    df = multi_classifier.load_data()
    sum_error_noreg = 0
    sum_error_reg = 0
    start_index = 0
    end_index = 33
    for i in range(3):
        multi_classifier.train(df, start_index, end_index)  
        predicted_labels = multi_classifier.classify(multi_classifier.lr1, multi_classifier.lr2, multi_classifier.lr3)
        error_rate = multi_classifier.calc_classification_error(predicted_labels, multi_classifier.y_test)
        sum_error_noreg += error_rate
        predicted_labels = multi_classifier.classify(multi_classifier.lr1_reg, multi_classifier.lr2_reg, multi_classifier.lr3_reg)
        error_rate = multi_classifier.calc_classification_error(predicted_labels, multi_classifier.y_test)
        sum_error_reg += error_rate
        start_index+=33
        end_index+=33
    print("Final average classification error rate (without reg term): ", (sum_error_noreg / 3))
    print("Final average classification error rate (with reg term): ", (sum_error_reg / 3))
      
if __name__== "__main__":
  main()
