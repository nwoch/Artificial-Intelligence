# Multi-Class Support Vector Machine

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:11:58 2018

@author: Nicole
"""

import numpy as np
from sklearn.svm import SVC
import W2D2inclass
import pickle

class MultiClassSVM:
    
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.classifiers = []
        self.labels_per_classifier = []
        self.predicted_labels = []
        self.multiclass_predicted_labels = []
    
    def split_and_train(self, test_data):
        """Split data set specific to each classifier and then fit"""
        count = 0
        count2 = 1
        for i in range(45): 
            x = []
            y = []
            for j in range(self.y_train.size): 
                if self.y_train[j] == count or self.y_train[j] == (count+count2):
                    x.append(self.x_train[j])
                    if self.y_train[j] == count:
                        y.append(1)
                    else:
                        y.append(-1)
            x = np.array(x)
            y = np.array(y)
            self.labels_per_classifier.append((count, count+count2))
            if (count+count2) == 9:
                count += 1
                count2 = 1
            else:
                count2 += 1
            svm = SVC(kernel='rbf', random_state=1, gamma=0.005, C=5)
            self.classifiers.append(svm)
            svm.fit(x, y)
            self.predicted_labels.append(svm.predict(test_data))
    
    def multiclass_predict(self):
        for i in range(self.predicted_labels[0].size):
            label_counts = [0] * 10
            for j in range(len(self.predicted_labels)):
                positive_label = self.labels_per_classifier[j][0]
                negative_label = self.labels_per_classifier[j][1]
                if self.predicted_labels[j][i] == 1:
                    label_counts[positive_label] = label_counts[positive_label] + 1
                elif self.predicted_labels[j][i] == -1:
                    label_counts[negative_label] = label_counts[negative_label] + 1
            multiclass_label = label_counts.index(max(label_counts))
            self.multiclass_predicted_labels.append(multiclass_label)
        return self.multiclass_predicted_labels
    
    def calc_classification_error(self, predicted, y):
        errors = 0
        for i in range(len(y)):
            if y[i] != predicted[i]:
                errors += 1
        print("Errors: ", errors, "/", len(y))
        return errors / len(y) 
    
    def store_classifiers(self):
        with open("models.pckl", "wb") as f:
            for classifier in self.classifiers:
                pickle.dump(classifier, f)
        self.classifiers = []
        with open("models.pckl", "rb") as f:
            while True:
                try:
                    self.classifiers.append(pickle.load(f))
                except EOFError:
                    break
            
            
def main():
    multiclass_svm = MultiClassSVM(W2D2inclass.X_train, W2D2inclass.y_train)
    multiclass_svm.split_and_train(W2D2inclass.X_test)
    multiclass_svm.store_classifiers()
    error_rate = multiclass_svm.calc_classification_error(multiclass_svm.multiclass_predict(), 
                                                          W2D2inclass.y_test)
    print("Final classification error rate: ", error_rate)

if __name__== "__main__":
  main()