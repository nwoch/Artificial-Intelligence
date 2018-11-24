# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:13:26 2018

@author: Nicole
"""
import numpy as np
import copy 

class Node:
    
    def __init__(self, feature, data):
        self.feature = feature
        self.data = data
        self.parent = None
        self.previous_features = []
        self.attribute_counts = {}
        self.children = []
        self.decision_label = None
        self.count_attributes()
        
    def count_attributes(self):
        attributes = self.data[:, self.feature]
        unique, counts = np.unique(attributes, return_counts=True)
        self.attribute_counts = dict(zip(unique, counts))       


class BinaryTree:
    
    def __init__(self, root):
        self.root = root
    
    def add_node(self, parent, child):
        parent.children.append(child)
        child.parent = parent
        new_previous = copy.copy(parent.previous_features)
        new_previous.append(parent.feature)
        child.previous_features = new_previous
        return child
    
    def traverse(self, x, current):
        if (len(current.children)) == 0:
            return current.decision_label
        for i in range(len(current.children)):
            if x[current.children[i].feature] in current.children[i].attribute_counts:
                return self.traverse(x, current.children[i])
