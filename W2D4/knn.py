# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 19:52:53 2018

@author: Nicole
"""
import numpy as np
import pandas as pd
from scipy.spatial import distance

class Knn:
    
    def __init__(self):
        self.x = None
        self.y = None
    
    def predict(self, k = 5):
        df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                'machine-learning-databases/iris/iris.data', header=None)
        self.x = df.iloc[0:150, 0:4].values
        self.y = df.iloc[0:150, [4]].values
        predicted_labels = []
        
        for x in self.x:
            # Find distance between x and each point in training set
            euclidean_distances = distance.cdist(self.x, x.reshape(1, -1))
            euclidean_distances[np.where(euclidean_distances == 0)] = np.nan
            k_labels = []
            # Find k closest neighbors to x
            for i in range(k):
                min_index = np.nanargmin(euclidean_distances, axis=0)
                euclidean_distances[min_index] = np.nan
                k_labels.append(self.y[min_index])
            # Choose label which appears the most among the k closest neighbors
            unique, counts = np.unique(k_labels, return_counts=True)
            k_label_counts = dict(zip(unique, counts))
            keys = list(k_label_counts.keys())
            values = list(k_label_counts.values())
            predicted_labels.append(keys[values.index(max(values))])
        return predicted_labels
        
    def calc_classification_error(self, predicted):
        errors = 0
        for i in range(len(self.y)):
            if self.y[i] != predicted[i]:
                errors += 1
        print("Errors: ", errors, "/", len(self.y))
        return errors / len(self.y)

def main():
    knn = Knn()
    error_rate = knn.calc_classification_error(knn.predict(k=20))
    print("Final classification error rate: ", error_rate)
    
if __name__== "__main__":
  main()