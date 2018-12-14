from methods.util_models import *
import argparse
import logging
from matplotlib import pyplot as plt

from dataset.dataset_loader import DatasetLoader

logging.getLogger().setLevel(logging.DEBUG)

valid_methods = ['kmeans', 'mahala', 'euclid', 'kernel', 'dist', 'transformer', 'neuralnet']

# Parse cmd args
parser = argparse.ArgumentParser(description='Main callpoint')
parser.add_argument(
    'method', choices=valid_methods, metavar='METHOD', help=str(valid_methods), default='none')
parser.add_argument('--rank', default=10, type=int,
                    help='The maximum rank to measure to.')
parser.add_argument('-cmc', action='store_true')    # plot cmc
parser.add_argument('--title', default=None,
                    help='Specify a title for the cmc curve')
args, unknown = parser.parse_known_args()

assert args.method in valid_methods

try:
    data_set_obj = DatasetLoader('dataset/CUHK03/cuhk03_new_protocol_config_labeled.mat',
                                 'dataset/CUHK03/feature_data.npy', do_val_split=False)
    data_set_val_obj = DatasetLoader('dataset/CUHK03/cuhk03_new_protocol_config_labeled.mat',
                                 'dataset/CUHK03/feature_data.npy', do_val_split=True)
except FileNotFoundError:
    raise FileNotFoundError(
        'Please supply the cuhk03_new_protocol_config_labeled and feature_data.npy files.')

if args.method == 'mahala':
    from methods.mahalanobis import mahalanobis
    result = mahalanobis(args, data_set_obj)
elif args.method == 'kmeans':
    from methods.k_means import k_means
    result = k_means(args, data_set_obj)
elif args.method == 'euclid':
    from methods.euclid import euclid
    result = euclid(args, data_set_obj)
elif args.method == 'kernel':
    from methods.kernel_methods import kernel_methods
    result = kernel_methods(args, data_set_obj)
elif args.method == 'dist':
    from methods.dist_methods import dist_methods
    result = dist_methods(args, data_set_obj)
elif args.method == 'transformer':
    from methods.transformer_methods import transformer_methods
    result = transformer_methods(args, data_set_obj)
elif args.method == 'neuralnet':
    from methods.neural_net import neural_net
    result = neural_net(args, data_set_obj, data_set_val_obj)
else:
    raise NotImplementedError('This method does not exist yet.')


logging.info("RESULTS: " + str(result))

# Plot graph using returned result
if args.cmc:
    plt.plot(range(1, args.rank+1), result[:args.rank])
    plt.xlabel('Rank')
    plt.ylabel('Accuracy')
    if not args.title:
        plt.title('Cumulative Match Curve: {}   '.format(args.method))
    else:
        plt.title(args.title)
    plt.show()
