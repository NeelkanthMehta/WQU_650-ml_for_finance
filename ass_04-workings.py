# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:24:15 2018

@author: neelkanth mehta
"""
#!reset

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

from skimage import io
from ipywidgets import interact

"""Problem 01"""
## Loading image
#im = io.imread('./cartoon.png')
#plt.imshow(im/np.max(im), interpolation= 'nearest')
#plt.show()
#
#m, n = im.shape[:2]
#data = im.reshape(m*n, 3)
#data = np.array(data, dtype= float)
#data = data.astype('unit8')
#
#
#wcss = []
#for i in range(1,11):
#    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#    kmeans.fit(data)
#    wcss.append(kmeans.inertia_)
#
#
## Elbow plot revealing No. of optimal clusters:
#plt.plot(range(1,11),wcss)
#plt.title('The Elbow method')
#plt.xlabel('No. of clusters')
#plt.ylabel('WCSS')
#plt.grid()
#plt.xticks(np.arange(1,11, step=1))
#plt.show()
#
## Fitting K-means algorithm using optimal clusters
#kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
#y_kmeans = kmeans.fit_predict(data)
#
##new_image = y_kmeans.reshape(m, n, 3)
##new_image = np.asarray(new_image, dtype='unit8')
##plt.imshow(new_image, interpolation= 'nearest')
##plt.show()
#
#plt.scatter(data[y_kmeans == 0, 0], data[y_kmeans == 0, 1], s=10, c='red', label='cluster 1')
#plt.scatter(data[y_kmeans == 1, 0], data[y_kmeans == 1, 1], s=10, c='green', label='cluster 2')
#plt.scatter(data[y_kmeans == 2, 0], data[y_kmeans == 2, 1], s=10, c='blue', label='cluster 3')
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:1], s=30, c='yellow', label='cluster centers')
#plt.title('K-means cartoon')
#plt.xlabel('m')
#plt.ylabel('n')
#plt.legend(loc='best')
#plt.show()

"""Problem 02"""
#from sklearn.datasets import load_digits
#
## Loading data
#digits = load_digits()
#
## finding number of unique labels
#number_digits = len(np.unique(digits.target))
#
## Inspecting different images
#def show_digits(k=0):
#    """
#    Show the digits in the training set
#    """
#    plt.imshow(digits.images[k], cmap=cm.binary)
#    plt.show
#
#w_show_digits = interact(show_digits, k=(0, 1796))
#
#
## Creating a regular PCA model
#pca = PCA(n_components=2)
#
#reduced_data_pca = pca.fit_transform(digits.data)
#
#colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
#
#for i in range(len(colors)):
#    x = reduced_data_pca[:,0][digits.target == i]
#    y = reduced_data_pca[:,1][digits.target == i]
#    plt.scatter(x, y, marker='o', s=10, facecolors=colors[i], edgecolors='k')
#plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.xlabel('First Principal Component')
#plt.ylabel('Second Principal COmponent')
#plt.title('Regular PCA Scatter Plot')
#plt.show()

"""Problem 03"""
