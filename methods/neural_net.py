import argparse
import logging

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from methods.util_models import *

from methods.nn_methods.nn_models import *
from methods.nn_methods.losses import *
from methods.nn_methods.utils import *
from methods.nn_methods.metrics import *
from methods.nn_methods.trainer import *



# TODO: your entrypoint is as follows, so you should use this to call all your neural net stuff

def neural_net(prev_args, data_set_obj, data_set_val_obj):
    """
        The neural_net entry point.
        Creates and trains a simple neural network to learn embeddings of inputs using the TripletLoss.
        The two available model architectures are SimpleEmbeddingNetV1 and SimpleEmbeddingNetV1
        See nn_methods/nn_models.py for more details.
        
        Code adapted from https://github.com/adambielski/siamese-triplet
        Ref Paper: https://arxiv.org/pdf/1703.07737.pdf
        Triplet Loss: sum(m + D_a_p - D_n_p) where
                        m = margin
                        D_a_p = distance between anchor and positive class samples
                        D_a_n = distance between anchor and negative class samples 

    """

    parser = argparse.ArgumentParser(description='NN')
    parser.add_argument('-t', '--train', action='store_true')       # plot cmc
    parser.add_argument('-nc', '--no-cuda', action='store_true')    # force no cuda
    parser.add_argument('-dv', '--do-val', action='store_true')     # split train into train+val
    parser.add_argument('-fp', '--file-path', type=str, default='checkpoint/simple_mlp.gold')   # load weights
    parser.add_argument('-m', '--model', choices=['mlp', 'mlp-conv'], type=str, default='mlp-conv') #MLP or CONV+MLP
    parser.add_argument('-rn', '--random-negative', action='store_true')  # don't use hard-negative loss based on batch samples
    parser.add_argument('-ep', '--epochs', type=int, default=30)    # num of epochs
    parser.add_argument('-dm', '--diff-margin', type=int, default=2.0) # difference margin for triplet loss
    args, unknown = parser.parse_known_args()


    do_train = args.train
    save_loc = args.file_path
    do_val = args.do_val
    do_hard_neg = not args.random_negative
    cuda = torch.cuda.is_available() if not args.no_cuda else False
    ep = args.epochs

    # choose either (train+val)+test dataset or train+test dataset based on passed cmd args
    val_train_dataset = data_set_val_obj if do_val else data_set_obj
                                
    train_dataset = BasicDataset(   
                                    True, 
                                    train_labels=val_train_dataset.train_labels,
                                    train_data=val_train_dataset.data_matx[val_train_dataset.train_idx]
                                )


    # for mini-batch training
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=5, n_samples=4)
    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs) # Returns triplets of images


    if do_val:
        test_dataset = BasicDataset(False, test_labels=val_train_dataset.val_labels, test_data=val_train_dataset.data_matx[val_train_dataset.val_idx])
        test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=5, n_samples=30)
        online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
    else:
        online_test_loader = None


    # margin diff between dist_pos and dist_neg for loss_fn (triplet loss)
    # this is a hyper-param
    margin = 2.0 #[1.0, 2.0, 3.0, 5.0]

    #model = CustomEmbeddingNet().double()
    if args.model == 'mlp':
        model = SimpleEmbeddingNetV1(input_size=2048).double()
    elif args.model == 'mlp-conv':
        model = SimpleEmbeddingNetV2(input_size=2048, first_conv_channels=16, out_features=512).double()
    else:
        raise NotImplementedError("Invalid model keyword.")

    if cuda:
        model.cuda()
    
    sample_selector = RandomNegativeTripletSelector(margin) \
                            if not do_hard_neg else HardestNegativeTripletSelector(margin)

    loss_fn = OnlineTripletLoss(margin, sample_selector)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = ep #[5, 15, 20, 50]
    log_interval = 500

    print("Model Params:\n", model.parameters)


    if do_train:
        fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])
    else:
        if not cuda:
            model = torch.load(save_loc, map_location='cpu')
        else:
            model = torch.load(save_loc)
        model.eval() # Useful to set Dropout / Batch Norm to deterministic i.e. give consistent results
        logging.info("Loaded model from file")

    logging.info("Testing on test data gal+qry...")
    gal_data = torch.from_numpy(val_train_dataset.data_matx[val_train_dataset.gallery_idx])
    qry_data = torch.from_numpy(val_train_dataset.data_matx[val_train_dataset.query_idx])


    # copy-back to cpu for query/gallery evaluation as GPU mem might not be enough
    if cuda:
        gal_data = gal_data.cpu()
        qry_data = qry_data.cpu()

    final_model = model.cpu()
    gal_transformed = final_model.forward(gal_data)
    qry_transformed = final_model.forward(qry_data)

    logging.info(" Gallery_transformed: ", gal_transformed.shape, "Query_transformed: ", qry_transformed.shape)

    knn = PairWiseDistTorch(val_train_dataset)
    result = knn.fit_predict(qry_data=qry_transformed, gal_data=gal_transformed).score(max_rank=10).ranked_acc


    return result