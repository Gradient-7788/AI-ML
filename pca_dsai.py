#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:17:43 2024

@author: yeshnavya
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.array([[2.5, 2.4],
                 [0.5, 0.7],
                 [2.2, 2.9],
                 [1.9, 2.2],
                 [3.1, 3.0],
                 [2.3, 2.7],
                 [2.0, 1.6],
                 [1.0, 1.1],
                 [1.5, 1.6],
                 [1.1, 0.9]])

#Plotting data
plt.figure(figsize=(10,10))
plt.scatter(data[:,0], data[:,1], label="Original Data")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()

#Standardised data
stand_data= (data- np.mean(data, axis=0))/np.std(data, axis=0)

#Covariance matrix
n= stand_data.shape[0]
m = stand_data.shape[1]

cov_matrix= stand_data.T @ stand_data
cov_matrix/= (n)

#Eigenvalues and Eigenvectors
eigenvalues, eigenvectors= np.linalg.eig(cov_matrix)

#Sort the eigenvectors and eigenvalues
sorted_indices= np.argsort(eigenvalues)[::-1]
eigenvectors_sorted= eigenvectors[:, sorted_indices]

#input for number of principal components
k= 1
      
#Select the largest k eigenvectors
principal_components= eigenvectors_sorted[:, :k]

#Projecting the data
transformed_data= stand_data.dot(principal_components)

#Plotting original and the principal component data together
plt.figure(figsize=(10,10))
plt.scatter(data[:,0], data[:,1], label="Original data")
plt.scatter(transformed_data, np.zeros_like(transformed_data), color="cyan", label="Transformed Data")

#Plot the vectors with their direction
for i in range(k):
    plt.quiver(np.mean(data[:,0]), np.mean(data[:,1]), principal_components[0,i]*2, principal_components[1,i]*2, angles="xy", scale_units="xy", scale=1, color="green", label=f"Principal Components {i+1}")
    
plt.title("PCA: Original Data Vs. Transformed Data with Vector Direction")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()

