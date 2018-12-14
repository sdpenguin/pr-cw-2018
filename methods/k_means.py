import logging
import argparse

import numpy as np

from sklearn.cluster import KMeans as sk_kmeans
from sklearn.cluster import MiniBatchKMeans as minibatch
from methods.util_models import *
from sklearn.metrics import pairwise_distances


class KMeans(sk_kmeans):
    """A KMeans class that will attempt to perform clustering and then pairwise distance calculations
        and ranking. The query items supplied will be compared to cluster means and then all cluster items
        belonging to that mean, before the next cluster. It can therefore reduce the number of distances
        we need to calculate."""
    def __init__(self, rank=10, clusters=1, iterations=3, metric='euclidean'):
        """ Iterations is the max iterations """

        sk_kmeans.__init__(self, n_clusters=clusters, max_iter=iterations)
        # Cluster ranks is a list of lists of knn sorted elements for each cluster w.r.t. the cluster mean
        self.rank = rank
        self.metric = metric

    def fit_predict(self, indexes, dataset_obj, sample_weight=None, sort_by_distance_to_mean=False):
        """Assign clusters.
        If sort_by_distance_to_mean is True, then we will sort gallery items according to the distance to the cluster mean rather
        than to the cluster mean rather than to the sample. This is not ideal."""

        # Query data
        query_data = dataset_obj.data_matx[dataset_obj.query_idx]
        query_ids = dataset_obj.query_idx
        # Gallery data
        gallery_data = dataset_obj.data_matx[indexes]
        gallery_ids = indexes

        logging.info('Finding cluster mean positions.')
        # Fitted is the gallery id cluster labels in order
        fitted = sk_kmeans.fit_predict(
            self, dataset_obj.data_matx[indexes], None, sample_weight=sample_weight)
        logging.info('Done')
        cluster_means = self.cluster_centers_
        # Cluster ids for each different class
        cluster_ids = [[x for x in range(len(cluster_means))] for i in range(len(query_ids))]

        # Measure distances to cluster centres
        cluster_distance_matrix = pairwise_distances(query_data, cluster_means, metric=self.metric)

        cluster_ids_swapped = swap_indices(cluster_ids)

        cluster_gallery_ids = []
        cluster_gallery_data = []
        for cluster in range(len(cluster_ids_swapped)):
            valid_cluster_gallery_ids = gallery_ids[fitted == cluster]
            valid_cluster_gallery_data = dataset_obj.data_matx[valid_cluster_gallery_ids]
            cluster_gallery_ids.append(valid_cluster_gallery_ids)
            cluster_gallery_data.append(valid_cluster_gallery_data)

        gallery_distances_per_cluster = []
        for cluster in cluster_gallery_data:
            # Take only the gallery ids in the cluster
            gallery_distance_for_cluster = pairwise_distances(query_data, cluster, metric=self.metric)
            gallery_distances_per_cluster.append(gallery_distance_for_cluster)

        gallery_distances_per_cluster_swapped = swap_indices(gallery_distances_per_cluster) 

        cluster_gallery_ids_stacked = [cluster_gallery_ids for i in range(len(gallery_distances_per_cluster_swapped))]

        sorted_gallery_distances_per_query = []
        sorted_gallery_ids_per_query = []
        for cluster_distances, gallery_distances, gallery_ids, index in zip(cluster_distance_matrix, gallery_distances_per_cluster_swapped, cluster_gallery_ids_stacked, range(len(cluster_distance_matrix))):
            sorted_gallery_distances_per_query.append(sort_by_another(gallery_distances, cluster_distances))
            sorted_gallery_ids_per_query.append(sort_by_another(gallery_ids, cluster_distances))

        num_query_items = len(sorted_gallery_distances_per_query)
        num_clusters = len(gallery_ids)
        num_gallery_items = len(gallery_data)

        double_sorted_gallery_distances_per_query = [[] for i in range(num_query_items)]
        double_sorted_gallery_ids_per_query = [[] for i in range(num_query_items)]
        for query_item, query_item_id, index1 in zip(sorted_gallery_distances_per_query, sorted_gallery_ids_per_query, range(len(sorted_gallery_distances_per_query))):
            for cluster, cluster_id, index2 in zip(query_item, query_item_id, range(len(query_item))):
                sorted_gallery_distances = sort_by_another(cluster, cluster)
                sorted_gallery_ids = sort_by_another(cluster_id, cluster)
                double_sorted_gallery_distances_per_query[index1].append(sorted_gallery_distances)
                double_sorted_gallery_ids_per_query[index1].append(sorted_gallery_ids)

        final_distance_array = []
        final_ids_array = []
        for distances, indexes in zip(double_sorted_gallery_distances_per_query, double_sorted_gallery_ids_per_query):
            final_distance_array.append([item for sublist in distances for item in sublist])
            final_ids_array.append([item for sublist in indexes for item in sublist])

        final_distance_array = np.array(final_distance_array)
        final_ids_array = np.array(final_ids_array)

        final_updated_distance_array = []
        final_updated_ids_array = []
        for distances, indexes, query_id in zip(final_distance_array, final_ids_array, range(num_query_items)):
            mask = [id_is_valid(gal_id, query_id, dataset_obj) for gal_id in indexes]
            redone_distances = np.append(distances[mask], ([-1] * 20))[:num_gallery_items]
            redone_indexes = np.append(indexes[mask], ([-1] * 20))[:num_gallery_items]
            final_updated_distance_array.append(redone_distances)
            final_updated_ids_array.append(redone_indexes)

        final_updated_distance_array = np.array(final_updated_distance_array)
        final_updated_ids_array = np.array(final_updated_ids_array)

        def gal_to_label(row_of_ids):
            return dataset_obj.labels[row_of_ids]

        final_updated_labels_array = np.stack([gal_to_label(row) for row in final_updated_ids_array])
        tensor_array = torch.tensor(np.array(final_updated_labels_array, dtype=np.int32))

        ranks = torch.stack([get_rank(row, i, dataset_obj) for i, row in enumerate(tensor_array)]).numpy()
        ranked_count = np.bincount(ranks.flatten())[1:-1]
        # CMC curve (percentage of query items which were in any particular rank or below)
        self.ranked_acc = np.cumsum(ranked_count / dataset_obj.query_idx.shape[0])

        return self


def sort_by_another(to_sort, basis):
    """Sorts to_sort based on the values in basis"""
    return [x for (y, x) in sorted(zip(basis, to_sort), key=lambda pair: pair[0])]


def swap_indices(a_list):
    """Converts [[1,2,3], [4,5,6]] to [[1,4],[2,5],[3,6]]"""
    new_list = []
    for i in range(len(a_list[0])):
        new_list.append([a_list[j][i] for j in range(len(a_list))])
    return new_list


def id_is_valid(gal_id, query_id, data):
    """ Returns False if the gal_id is invalid according to camera and label values """
    return not ((data.cam_idx[query_id] == data.cam_idx[gal_id]) and (data.labels[query_id] == data.labels[gal_id]))


def extend_array(l, n):
    """Extend an array from the back. Useful to fill arrays that have been shortened and need to conform to a certain size."""
    l.extend([-1] * n)
    l = l[:n]
    return l


def k_means(prev_args, data_set_obj):
    """The k_means entry function"""
    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument('--clusters', required=True,
                        help='The number of clusters to use for kmeans.', type=int)
    parser.add_argument('--iterations', default=300,
                        help='The maximum number of iterations for the algorithm.', type=int)
    parser.add_argument('--metric', default='euclidean',
                        help='The distance metric to use.')
    args, unknown = parser.parse_known_args()
    kmeans = KMeans(prev_args.rank, args.clusters, args.iterations, args.metric)
    kmeans.fit_predict(data_set_obj.gallery_idx, data_set_obj)
    return kmeans.ranked_acc
