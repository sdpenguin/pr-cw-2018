import logging
import argparse

from sklearn.metrics.pairwise import *
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.random_projection import *
from sklearn.kernel_approximation import RBFSampler
from metric_learn.lmnn import LMNN

from methods.util_models import *


# TODO: implement api
def transformer_methods(prev_args, data_set_obj):
    """The transformer_methods entry function"""

    parser = argparse.ArgumentParser(description='distmethods')
    parser.add_argument('--transformer', required=True,
                        help='The transformer to use. One of [pca, quantile, normalizer, standard_scaler, gauss_rand].', type=str)
    parser.add_argument('--components', default=1024,
                        help='The kernel to use for the kernel methods.', type=str)
    args, unknown = parser.parse_known_args()

    n_components = args.components
    max_rank = prev_args.rank

    train_data  =    data_set_obj.data_matx[data_set_obj.train_idx]
    gal_data    =    data_set_obj.data_matx[data_set_obj.gallery_idx]
    qry_data    =    data_set_obj.data_matx[data_set_obj.query_idx]

    transformers = {
        'pca': PCA(n_components=n_components),
        'quantile': QuantileTransformer(n_quantiles=n_components,output_distribution='normal'),
        'normalizer': Normalizer(),
        'standard_scaler': StandardScaler(),
        'gauss_rand': GaussianRandomProjection(n_components=n_components),
    }

    knn = PairwiseDist(data_set_obj)

    if args.transformer not in transformers:
        raise ValueError('Transformer must be one of: {}'.format(transformers.keys()))

    transformers[args.transformer].fit(train_data)

    qry_transform = transformers[args.transformer].transform(qry_data)
    gal_transform = transformers[args.transformer].transform(gal_data)

    knn.fit_predict(qry_data=qry_transform, gal_dat=gal_transform).score(max_rank=max_rank)

    return knn.ranked_acc