# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 17:08:43 2021

@author: mohamed saeed
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 #read data
dataset = pd.read_csv('Wuzzuf_Jobs.csv')

# columns factorize
from sklearn.cluster import KMeans
dataset['factorized_YearExp'] = pd.factorize(dataset['YearsExp'])[0]
dataset['Title'] = pd.factorize(dataset['Title'])[0]
dataset['Company'] = pd.factorize(dataset['Company'])[0]

X = dataset.iloc[:, [0, 1]].values
x=pd.DataFrame(X)

# Using the elbow method to find the optimal number of clusters
wcss =[]
for i in range(1 , 11):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)         
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#visulization
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of titles and companies ')
plt.xlabel('The Title')
plt.ylabel('The Company')
plt.legend()
plt.show()