# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:10:23 2018

@author: Nicole
"""
import numpy as np
import pandas as pd
from binary_tree import Node
from binary_tree import BinaryTree

class DecisionTree:
    
    def __init__(self):
        self.data = None
        self.info_gains = {}
        self.tree = None
    
    def process_data(self):
        """Construct training data set"""
        df = pd.read_csv('tennis.csv', header=None)
        self.data = df.iloc[1:15, 1:6].values
    
    def calc_gini(self, node):
        """Calculate the Gini index for a node"""
        sum_of_squares = 0
        for key, value in node.attribute_counts.items():
            sum_of_squares += (np.square(value/len(node.data)))
        gini = 1 - sum_of_squares
        return gini
    
    def calc_information_gain(self, node):
        """Calculate the potential information gain for each of a node's attributes"""
        gini_sum = 0
        for key, value in node.attribute_counts.items():
            probability = value/len(node.data)
            positive_label_count = 0
            for i in range(len(node.data)):
                if node.data[i][node.feature] == key and node.data[i][-1] == "Yes":
                    positive_label_count+=1
            label_probability = positive_label_count/value
            gini_sum += ((probability)*(1-np.square(label_probability)))
        info_gain = self.calc_gini(node) - gini_sum
        self.info_gains[node] = info_gain
        return info_gain 
    
    def filter_data(self, data, feature, attribute):
        """Filter data for a attribute node of the feature chosen to split on 
            to only include entries containing that attribute"""
        filtered = []
        for i in range(len(data)):
            if data[i][feature] == attribute:
                filtered.append(data[i])
        return np.array(filtered)
        
    def decide(self, dictionary):
        """Decide which feature to split on based on greatest information gain"""
        keys = list(dictionary.keys())
        values = list(dictionary.values())
        return keys[values.index(max(values))]        
    
    def fit(self, node = None, labels = None):
        """Fit tree using training data set"""
        self.info_gains.clear()
        
        # Calculate information gain for all potential features to split on
        for i in range(len(self.data[0]) - 1):
            if self.tree is None:
                self.calc_information_gain(Node(i, self.data))
            elif i not in node.previous_features and node.feature != i: 
                self.calc_information_gain(Node(i, node.data))
                
        # If there are no more features, chose label with highest occurence
        if len(self.info_gains) == 0:
            unique, counts = np.unique(labels, return_counts=True)
            label_counts = dict(zip(unique, counts))
            node.decision_label = self.decide(label_counts)
        
        # Choose to split on feature with greatest information gain
        split_feature = self.decide(self.info_gains)
        
        # Add chosen feature's attribute nodes to tree 
        # and then choose which remaining feature to split each of those nodes on
        for key, value in split_feature.attribute_counts.items():
            split_data = self.filter_data(split_feature.data, split_feature.feature, key)
            # If parent is root
            if node is None:
                if self.tree is None:
                    self.tree = BinaryTree(split_feature)
                parent = split_feature
            else:
                parent = node
            child = Node(split_feature.feature, split_data)
            self.tree.add_node(parent, child)
            child_labels = child.data[:, -1]
            # If child's labels are pure, choose that label
            if np.unique(child_labels).size == 1:
                child.decision_label = child_labels[0]
            else:
                self.fit(node = child, labels = child_labels)
