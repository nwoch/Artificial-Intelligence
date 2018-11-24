# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:16:34 2018

@author: Nicole
"""
from sklearn.datasets import make_blobs
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import copy

class KMeans:
    
    def __init__(self, k, init_type, n_iter=10):
        self.k = k
        self.init_type = init_type
        self.n_iter = n_iter
        self.M = [] # centroids
        self.x = None
        self.clusters = None
        self.distortions = []
        self.calc_k_means = False
    
    def load_data(self, num_clusters):
        blobs = make_blobs(n_samples=200, cluster_std=0.2, centers=num_clusters, random_state=0)
        self.x = blobs[0]
    
    def calculate_k_means(self, k):
        self.clusters = np.empty((k, 0)).tolist()
        for x in self.x:
            closest_centroid = self.find_closest_centroid(x)
            self.clusters[np.where(self.M == closest_centroid)[0][0]].append(x)
        for i in range(len(self.clusters)):
            if len(self.clusters[i]) != 0:
                new_centroid = np.mean(self.clusters[i], axis=0)
                self.M[i] = new_centroid
        return self.M
    
    def find_closest_centroid(self, point):
        distances_to_centroids = distance.cdist(self.M, np.reshape(point, (1,-1)))
        min_distance = np.amin(distances_to_centroids, axis=0)
        if self.init_type == "random" or self.calc_k_means == True:    
            min_distance_index = np.where(distances_to_centroids == min_distance)
            return self.M[min_distance_index[0][0]]
        elif self.init_type == "++":
            return min_distance
        
    def initialize_k_means(self, k):
        if self.init_type == "random":    
            self.M = np.random.rand(k, 2)
        elif self.init_type == "++":
            self.M = []
            x_copy = copy.copy(self.x)
            self.M.append(x_copy[0])
            x_copy = np.delete(x_copy, 0, axis=0)
            for i in range(k - 1):
                sum_distances_squared = 0
                min_distances_squared = []
                for x in x_copy:
                    min_distance_squared = np.square(self.find_closest_centroid(x))
                    min_distances_squared.append(min_distance_squared)
                    sum_distances_squared += min_distance_squared
                #use x you plugged in? what's the point of the sum or dividing by the sum? ***
                next_centroid_index = min_distances_squared.index(max(min_distances_squared))
                next_centroid = x_copy[next_centroid_index]
                self.M.append(next_centroid)
                x_copy = np.delete(x_copy, next_centroid_index, axis=0)
            self.M = np.array(self.M)
        
    def calc_distortion(self):
        distortion = 0
        for i in range(len(self.M)):
            if len(self.clusters[i]) != 0:
                distances_to_centroid = distance.cdist(self.clusters[i], np.reshape(self.M[i], (1,-1)))
                distortion += np.sum(np.square(distances_to_centroid))
        self.distortions.append(distortion)
        return distortion
        
    def plot(self):
        # Data points
        plt.scatter(self.x[:,0], self.x[:,1], c='white', marker='o', edgecolor='black', s=50)
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        # Distortion
        plt.plot(range(1, self.k + 1), self.distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()
        plt.show()
    
    def cluster_and_plot(self, num_clusters):
        self.load_data(num_clusters)
        for i in range(1, self.k + 1):
            self.calc_k_means = False
            self.initialize_k_means(i)
            self.calc_k_means = True
            for j in range(self.n_iter):
                self.calculate_k_means(i)
            self.calc_distortion()
        self.plot()
    
    
def main():
    k_means = KMeans(10, "random")
    k_means_plus = KMeans(10, "++")
    k_means.cluster_and_plot(3)
    k_means_plus.cluster_and_plot(3)

if __name__== "__main__":
  main()