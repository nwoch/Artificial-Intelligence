# Honor Code Upheld: Nicole Woch 11/5/18

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:29:29 2018

@author: Nicole
"""
from sklearn import datasets
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from sklearn import ensemble
import numpy as np

class TwentyNewsGroupsClassifiers:
    
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_labels = None
        
    def load_vector_data(self):
        train_data = datasets.fetch_20newsgroups_vectorized()
        test_data = datasets.fetch_20newsgroups_vectorized(subset='test')
        self.x_train = train_data.data
        self.y_train = train_data.target
        self.y_labels = train_data.target_names
        self.x_test = test_data.data
        self.y_test = test_data.target
        
    def perceptron(self):
        perceptron = linear_model.Perceptron(penalty='l2', alpha=0.0001, eta0=0.0004, random_state=1)
        perceptron.fit(self.x_train, self.y_train)
        y_predicted = perceptron.predict(self.x_test)
        score = metrics.f1_score(self.y_test, y_predicted, average='macro')
        print("Perceptron F Score:", score)
        
    def logistic_regression(self):
        logistic_regression = linear_model.LogisticRegression(penalty='l2', C=40)
        logistic_regression.fit(self.x_train, self.y_train)
        y_predicted = logistic_regression.predict(self.x_test)
        score = metrics.f1_score(self.y_test, y_predicted, average='macro')
        print("Logistic Regression F Score:", score)
    
    def svm(self):
        #not dependent on data dimensions, only requires dot products
        sv_machine = svm.SVC(gamma=0.4, C=8, random_state=1)
        sv_machine.fit(self.x_train, self.y_train)
        y_predicted = sv_machine.predict(self.x_test)
        score = metrics.f1_score(self.y_test, y_predicted, average='macro', labels=np.unique(y_predicted))
        print("Soft Margin SVM F Score:", score)
        
    def linear_svm(self):
        linear_svm = svm.LinearSVC(penalty='l2', C=2.2, random_state=1)
        linear_svm.fit(self.x_train, self.y_train)
        y_predicted = linear_svm.predict(self.x_test)
        score = metrics.f1_score(self.y_test, y_predicted, average='macro')
        print("Linear SVM F Score:", score)
        
    def decision_forest(self):
        decision_forest = ensemble.RandomForestClassifier(max_depth=92, max_features='sqrt')
        decision_forest.fit(self.x_train, self.y_train)
        y_predicted = decision_forest.predict(self.x_test)
        score = metrics.f1_score(self.y_test, y_predicted, average='macro', labels=np.unique(y_predicted))
        print("Decision Forest F Score:", score)

            
def main():
    twenty_ng = TwentyNewsGroupsClassifiers()
    twenty_ng.load_vector_data()
    
    twenty_ng.perceptron()
    twenty_ng.logistic_regression()
    twenty_ng.svm()
    twenty_ng.linear_svm()
    twenty_ng.decision_forest()

if __name__== "__main__":
  main()
