# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:52:40 2018

@author: Nicole
@author: Emma
"""

import numpy as np
from numpy import linalg
from scipy.spatial import distance

class DataManipulation:
# Part C

    def read_file(self, file_name):
        data_file = open(file_name)
        data = np.genfromtxt(data_file, delimiter=',')
        return data
    
    def calc_mean(self, data):
        # C1
        mean = np.mean(data, axis=0)
        return mean

    def calc_variance(self, data):
        # C2
        variance = np.var(data, axis=0)
        return variance
        
    def calc_min_distance_to_plane(self, data):
        # C3 
        min_to_plane = np.amin(np.absolute(data), axis=0)
        return min_to_plane
        
    def calc_min_euclidean_distance(self, data, point):
        # C4
        euclidean_distances = distance.cdist(data, point)
        min_distance = np.amin(euclidean_distances, axis=0)
        return min_distance
    
    def calc_gramm_matrix(self, data):
        # C5
        transposed = np.transpose(data)
        gramm_matrix = np.dot(data, transposed)
        return gramm_matrix
    
    def center_and_normalize(self, data, mean):
        # C6
        centered = data - mean
        norm = linalg.norm(data, axis=1)
        max_magnitude = np.amax(norm)
        cen_and_norm = np.divide(centered, max_magnitude)
        return cen_and_norm


dm = DataManipulation()
print("Data:")
data = dm.read_file("W1D1data.txt")
print(data, '\n')
print("(1) Compute the mean of the data.")
mean = dm.calc_mean(data)
print(mean, '\n')
print("(2) Compute the variance of the data")
print(dm.calc_variance(data), '\n')
print("(3) Compute minimum distances from the data to the planes x1 = 0, x2 = 0,...,xn = 0.")
print(dm.calc_min_distance_to_plane(data), '\n')
print("(4) Compute minimum distance from data the point (1, 1, 1...) ∈ R^n")
print(dm.calc_min_euclidean_distance(data, np.array([[1, 1, 1, 1, 1]])), '\n')
print("(5) Compute the ’Gramm Matrix’: the d × d matrix of inner products")
print(dm.calc_gramm_matrix(data), '\n')
print("(6) Center and normalize the data.")
print(dm.center_and_normalize(data, mean))
