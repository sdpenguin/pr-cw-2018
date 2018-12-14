import logging
import argparse

from metric_learn.nca import NCA
from metric_learn.lfda import LFDA
from metric_learn.mlkr import MLKR
from metric_learn.lmnn import LMNN
from metric_learn.rca import RCA
from metric_learn.mmc import MMC_Supervised

from methods.util_models import *


# TODO: implement api
def mahalanobis(prev_args, data_set_obj):
    """The mahalanobis entry function."""
    parser = argparse.ArgumentParser(description='distmethods')
    parser.add_argument('--model', default='mmc',
                        help='The model used to obtain M matrix. Default: dynamic mcc. Possible: [lmnn, nca, lfda, mlkr, mmc]', type=str)
    args, unknown = parser.parse_known_args()

    pca = PCATorch(data_set_obj).fit(dim=1024)

    models = {
    'lmnn': LMNN(k=5, verbose=1),
    'nca':  NCA(num_dims=512),
    'lfda': LFDA(num_dims=1024, k=7),
    'mlkr': MLKR(num_dims=1024, verbose=True),
    'mmc': MMC_Supervised(verbose=True)
    }

    if args.model not in models:
        raise ValueError('Your model must be one of: {}'.format(models.keys()))

    model = models[args.model]

    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    train_data  =    torch.tensor(data_set_obj.data_matx[data_set_obj.train_idx], device=dev)
    gal_data    =    torch.tensor(data_set_obj.data_matx[data_set_obj.gallery_idx], device=dev)
    qry_data    =    torch.tensor(data_set_obj.data_matx[data_set_obj.query_idx], device=dev)

    train_data = pca.transform(train_data)
    qry_data, gal_data  = pca.transform(qry_data), pca.transform(gal_data)
    model.fit(train_data.cpu(), data_set_obj.train_labels)
    qry_transform = model.transform(X=qry_data.cpu())
    gal_transform = model.transform(X=gal_data.cpu())

    results = \
        PairWiseDistTorch(data_set_obj).fit_predict(qry_transform, gal_transform).score(max_rank=50).ranked_acc
    return results