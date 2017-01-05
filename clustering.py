from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cluster
import random
from images_200 import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_clusters_centroids():
    n = 10
    clustering = cluster.KMeans(n_clusters = n)
    clustering.fit(images)
    i = 0 
    for centroid in clustering.cluster_centers_:
        image = []
        line = []
        for elem in centroid:       
            if(elem > 0.5):
                line.append(1)
            else:
                line.append(0)
            i = i + 1
            if(i == 28):
                i = 0
                image.append(line)
                line = []
        plt.imshow(image, interpolation='nearest', cmap='Greys')
        plt.show()

def show_samples(n):
    for i in range(n):
        image = random.sample(images,1)[0]
        image = np.array(image)
        image = image.reshape(28,28)
        plt.imshow(image, interpolation='nearest', cmap='Greys')
        plt.show()

if __name__ == '__main__':
    plot_clusters_centroids()