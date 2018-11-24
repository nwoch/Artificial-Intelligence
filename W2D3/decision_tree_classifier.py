# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:18:36 2018

@author: Nicole
"""
import pandas as pd
from decision_tree import DecisionTree

class DecisionTreeClassifier:
    
    def __init__(self):
        self.x = None
        self.y = None
    
    def predict(self, tree):
        df = pd.read_csv('tennis.csv', header=None)
        self.x = df.iloc[1:15, 1:5].values
        self.y = df.iloc[1:15, 5].values
        predicted_labels = []
        for i in range(len(self.x)):
            label = tree.traverse(self.x[i], tree.root)
            predicted_labels.append(label)
        return predicted_labels
    
    def calc_classification_error(self, predicted):
        errors = 0
        for i in range(len(self.y)):
            if self.y[i] != predicted[i]:
                errors += 1
        print("Errors: ", errors, "/", len(self.y))
        return errors / len(self.y)


def main():
    d_tree = DecisionTree()
    d_tree.process_data()
    d_tree.fit()
    dtree_classifier = DecisionTreeClassifier()
    predictions = dtree_classifier.predict(d_tree.tree)
    error_rate = dtree_classifier.calc_classification_error(predictions)
    print("Final classification error rate: ", error_rate)
    
if __name__== "__main__":
  main()