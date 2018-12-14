import argparse
import logging

import torch
from metric_learn.lmnn import LMNN
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import *

from methods.util_models import PCATorch, PairwiseDist

def kernel_methods(prev_args, data_set_obj):
    """The distance methods entry function."""
    parser = argparse.ArgumentParser(description='distmethods')
    parser.add_argument('--kernel', required=True,
                        help='The kernel to use for the kernel methods.', type=str)
    args, unknown = parser.parse_known_args()
    train_data  =    data_set_obj.data_matx[data_set_obj.train_idx]
    gal_data    =    data_set_obj.data_matx[data_set_obj.gallery_idx]
    qry_data    =    data_set_obj.data_matx[data_set_obj.query_idx]

    pca = PCATorch(data_set_obj).fit(dim=1024)
    kernel = args.kernel
    gram_matx = None

    kernels = {
    'cosine': (lambda x, y: cosine_similarity(x,y)),
    'poly':  (lambda x, y: polynomial_kernel(x, y, degree=3)),
    'laplacian': (lambda x, y: laplacian_kernel(x, y, gamma=None)),
    'chi2': (lambda x, y: additive_chi2_kernel(x, y))
    }

    if not args.kernel in kernels:
        raise Exception('Kernels must be one of: {}'.format(kernels.keys()))

    gram_matx = kernels[args.kernel](qry_data, gal_data)

    max_sim = np.max(gram_matx)
    print("Max Similarity Measure for %s: " % kernel, max_sim)

    # for dist_calc
    dist_matx = max_sim - gram_matx
    knn = PairwiseDist(data_set_obj)
    knn.dist_matx_torch = torch.tensor(dist_matx)
    result = knn.score(max_rank=prev_args.rank)

    return knn.ranked_acc
