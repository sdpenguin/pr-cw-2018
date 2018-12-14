from methods.k_means import KMeans
from matplotlib import pyplot as plt
import numpy as np
import os

from dataset.dataset_loader import DatasetLoader

data = DatasetLoader()

""" This will plot a series of kmeans plots over possible_clusters and possible_iterations.]
    The default action is to plot in the folder k_means/images and k_means/data will contain
    corresponding numpy arrays. """

possible_clusters = [1, 10, 100, 200, 300, 500, 700, 1000, 2500, 5000]
possible_iterations = [300]
rank = 50  # Rank to plot to

# Note that the script will skip producing images which it considers to already exist (if it find an image with the same name)
# It will notify you of this.

for num_clusters in possible_clusters:
    for iterations in possible_iterations:
        if os.path.exists('./k_means/data/{}_{}.npy'.format(num_clusters, iterations)):
            print('Skipping for {} clusters {} iterations.'.format(
                num_clusters, iterations))
            continue
        worker = KMeans(rank=rank, clusters=num_clusters,
                        iterations=iterations)
        worker.fit_predict(data.gallery_idx, data)
        data2 = worker.ranked_acc
        np.save('./k_means/data/{}_{}.npy'.format(num_clusters, iterations), data2)
        plt.plot(range(1, len(data2) + 1), data2)
        plt.xlabel('Rank')
        plt.ylabel('Accuracy')
        plt.title(
            'CMC Curve for {}-Means Clustering for {} Iterations'.format(num_clusters, iterations))
        plt.savefig(
            './k_means/images/{}_{}.png'.format(num_clusters, iterations))
        plt.close()
        print('Done for {} clusters {} iterations.'.format(
            num_clusters, iterations))
        del worker
