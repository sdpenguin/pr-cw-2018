import argparse
import logging

import numpy as np

from methods.util_models import PairWiseDistTorch


def euclid(prev_args, data):
    """The euclidean baseline entry function"""
    parser = argparse.ArgumentParser(description='baseline')
    args, unknown = parser.parse_known_args()
    knn = PairWiseDistTorch(data)
    result = knn.fit_predict().score(max_rank=prev_args.rank).ranked_acc
    logging.info(result)
    return result
