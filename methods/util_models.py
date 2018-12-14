import numpy as np
import torch as torch
import logging
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import pairwise_distances 

from numpy.linalg import inv, cholesky
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from scipy.spatial.distance import cdist



def dist2label(dist_vect_torch, query_i, dataset_obj, k=10, allow_duplicates=True):
        '''Find the first k labels in order of distance.
        k should only be less than the number of gallery items if you are not going
        to measure anything above k, since the get_rank function will give an invalid rank for the data.
        It is purely for efficiency reasons.'''
        # Select only distances corresponding to valid gallery items
        valid_gallery_mask = torch.from_numpy(dataset_obj.get_valid_test_gallery_mask(dataset_obj.query_labels[query_i],
            dataset_obj.query_cam_idx[query_i], dtype=np.uint8))
        
        new_sel = torch.masked_select(dist_vect_torch, valid_gallery_mask)

        if allow_duplicates:
            top_k_ids = torch.sort(new_sel)[1][:k]
            # must be long/int64 as torch can only use that for indexing
            new_labels = dataset_obj.labels_torch[ torch.masked_select(dataset_obj.gallery_idx_torch, valid_gallery_mask)[top_k_ids].long() ]
        else:
            top_k_ids = torch.sort(new_sel)[1]
            gallery_idx = torch.masked_select(dataset_obj.gallery_idx_torch, valid_gallery_mask)[top_k_ids]
            x = []
            for item in gallery_idx:
                if item.numpy() not in x:
                    x.append(item.numpy())
                if len(x) == k:
                    break
            new_labels = dataset_obj.labels_torch[ torch.tensor([int(x[i]) for i in range(len(x))]).long() ]

        return new_labels      # i.e. each unique label can only take one position in the ranking

 ## remember -- this is 1-based!
def get_rank(labels, query_i, dataset_obj):
    '''
        Labels are sorted gallery ids.
        query_i is the index of the query array.
        Therefore this runs for just one query item.
        Returns the rank starting from 1.
        If the label is not found. It will return the rank as the length of the labels + 1.
    '''
    
    val = (labels == dataset_obj.labels_torch[dataset_obj.query_idx[query_i]]).nonzero()
    return torch.tensor([labels.shape[0]+1]) if len(val) == 0 else val[0]+1


def get_AP(labels, query_i, dataset_obj):
    '''
        Labels are sorted gallery ids.
        query_i is the index of the query array.
        Therefore this runs for just one query item.

        Returns AP for query_i calculated on theoretical
        maximum samples for same label in gallery after same cam_id&class deletion
    '''
    true_label = dataset_obj.labels_torch[dataset_obj.query_idx[query_i]].numpy()

    valid_gallery_mask = torch.from_numpy(dataset_obj.get_valid_test_gallery_mask(dataset_obj.query_labels[query_i],
            dataset_obj.query_cam_idx[query_i], dtype=np.uint8))
    
    valid_gal = torch.masked_select(dataset_obj.gallery_labels_torch, valid_gallery_mask)
    
    tp = 0
    all_ = np.count_nonzero(np.isin(valid_gal.numpy(), true_label)) # max positives
    score = 0  # running sum of precisions
    true_label = torch.tensor(true_label)
    for i in range(all_):
        if labels[i] == true_label:
            tp +=1  # true positive
        score += tp/(i+1)
    
    # return AP: Average Precision
    return torch.tensor([score/all_])


class BaseModel(object):

    # The cumulative sum of the ranked accuracies.
    ranked_acc = None
    ranked_labels_torch = None

    def __init__(self):
        raise NotImplementedError('init has not been implemented. Are you calling the base class directly?')

    def fit_predict(self, qry_data=None, gal_data=None):
        raise NotImplementedError('fit_predict has not been implemented for this class.')

    def score(self, max_rank=10):
        '''
            Perform kNN scoring on a distance metric
            use self.ranked_axx for array of rank-1 -> rank-p accuracies
            rank-p = max_rank
        '''
        if self.dist_matx_torch is None:
            raise ValueError("Please call fit_predict or predict before score!")

        label_sorted_tensor = torch.stack([dist2label(row, i, self.dataset_obj, k=max_rank) for i,row in enumerate(self.dist_matx_torch)])
        ranks = torch.stack([get_rank(row, i, self.dataset_obj) for i,row in enumerate(label_sorted_tensor)])


        APs = torch.stack([get_AP(row, i, self.dataset_obj) for i,row in enumerate(label_sorted_tensor)])

        meanAP = np.mean(APs.numpy())
        print("mAP: ", meanAP)

        # Bins of the different counts with 0 and rank+1 bins not included
        ranked_count = np.bincount(ranks.numpy().flatten())[1:-1]
        # CMC curve (percentage of query items which were in any particular rank or below)
        ranked_acc = np.cumsum(ranked_count / self.dataset_obj.query_idx.shape[0])

        self.mean_AP = meanAP
        self.ranked_acc = ranked_acc
        self.ranked_labels_torch = label_sorted_tensor

        return self

class PairWiseDistTorch(BaseModel):
    '''
        A Simple euclidean pair-wise distance calculator and k-rank scorer using pytorch
    '''
    def __init__(self, dataset_obj=None, **kwargs):
        # super().__init__()
        self.dataset_obj = dataset_obj

        ## filled on fit
        self.ranked_acc = None
        self.ranked_labels_torch = None
        self.dist_matx_torch = None

    def fit_predict(self, qry_data=None, gal_data=None):

        eps = 1e-5  # for stability

        if qry_data is None and gal_data is None:
            dataQ = torch.from_numpy((self.dataset_obj.data_matx[self.dataset_obj.query_idx]))                
            dataG = torch.from_numpy((self.dataset_obj.data_matx[self.dataset_obj.gallery_idx]))
        elif type(qry_data) == np.ndarray and type(gal_data) == np.ndarray:
            dataQ, dataG = torch.from_numpy(qry_data), torch.from_numpy(gal_data)
        elif type(qry_data) == torch.Tensor and type(gal_data) == torch.Tensor:
            dataQ, dataG = qry_data, gal_data
        else:
            raise NotImplementedError("Found Types: %a %a. Both types must match." % (type(qry_data), type(gal_data)))

        #https://stackoverflow.com/questions/51986758/calculating-euclidian-norm-in-pytorch-trouble-understanding-an-implementation
        n_1, n_2 = dataQ.size(0), dataG.size(0)
        norms_1 = torch.sum(dataQ**2, dim=1, keepdim=True)
        norms_2 = torch.sum(dataG**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
             norms_2.transpose(0, 1).expand(n_1, n_2))
        self.dist_matx_torch = torch.sqrt(eps + torch.abs(norms - 2 * dataQ.mm(dataG.t())))

        logging.info("kNN Done")
        return self

    def fit_predict_slow(self, data_loader, rank=10):
        ### only for comparision and testing, use fast method instead
        # Query and Gallery
        dataQ = data_loader.data_matx[data_loader.query_idx]
        dataG = data_loader.data_matx[data_loader.gallery_idx]

        ## we do 15 because upto 4 maybe invalid samples (same id & cam) and so on safe side for 10 nn this is ok
        knn = NearestNeighbors(n_neighbors=15,metric='euclidean', algorithm = 'ball_tree')\
                                .fit(dataG)
        
        neighDist, neighIdx = knn.kneighbors(dataQ)

        logging.info("kNN Done")
        #print("kNN Done, Result:\n" , neighDist, "\nIdx\n", neighIdx)



        def get_knn_slow(idx_of_gal, query_i, k=10):
            #idx_of_gal -> relative to gal_labels_idx so 0 for 1dt elem in gal_labels_idx
            gal_idx = data_loader.gallery_idx[idx_of_gal]
            # dist vect is R^{5328} which is as big as gallery this is a torch vect!
            #
            validGalleryIdx = data_loader.gallery_idx[ data_loader.get_valid_test_gallery_mask(data_loader.query_labels[query_i],
                data_loader.query_cam_idx[query_i]) ]
            
            ## only get those that are valid, out of that get top k
            gal_idx =  (gal_idx[ (np.isin(gal_idx, validGalleryIdx, assume_unique=True)) ])[:k]
            #print(gal_idx)
            
            #kth_sort_idx = torch.sort(new_sel)[1][:k]
            new_labels = np.array(data_loader.labels[ gal_idx ]) #, dtype=np.int32
            
            #print(new_labels)


            return new_labels

        self.ranked_labels = np.stack([get_knn_slow(row, i, k=rank) for i,row in enumerate(neighIdx)])

        logging.info(self.ranked_labels)
        logging.info("Shape: ", self.ranked_labels.shape)
        #print("Different: ", np.count_nonzero(np.equal(self.ranked_labels_torch.numpy(), self.ranked_labels) == 0))




class PairwiseDist(BaseModel):
    '''
        A Simple euclidean pair-wise distance calculator and k-rank scorer.
    '''
    def __init__(self, dataset_obj):
        self.dataset_obj = dataset_obj
        
        ## filled on fit
        self.dist_matx_torch = None

    
    def fit_predict(self, qry_data=None, gal_dat=None, metric='l2'):
        

        ## test data
        qry_data = self.dataset_obj.data_matx[self.dataset_obj.query_idx] \
                        if qry_data is None else qry_data
        gal_data = self.dataset_obj.data_matx[self.dataset_obj.gallery_idx] \
                        if gal_dat is None else gal_dat

        if metric == 'mahalanobis':
            ### for this we use train data only to calc the cov matrix
            logging.info("Inverting matrix please wait...")
            train_cov_inv = inv(np.cov(
                        self.dataset_obj.data_matx[self.dataset_obj.train_idx], 
                        rowvar = False))
            
            logging.info("Now calc dist...")
            ## do it directly using scipy also scipy is backend of most sklearn
            self.dist_matx = cdist(qry_data, gal_data, metric=metric, VI=train_cov_inv)
        elif metric == 'seuclidean':
            cov = np.cov(
                        self.dataset_obj.data_matx[self.dataset_obj.train_idx], 
                        rowvar = False)
            var = np.linalg.eigvals(cov)
            self.dist_matx = cdist(qry_data, gal_data, metric=metric, V=var)

        else:
            # generic
            self.dist_matx = pairwise_distances(qry_data, gal_data, metric=metric)

        #print("Converting to torch tensor...")
        self.dist_matx_torch = torch.from_numpy(self.dist_matx)
        
        
        return self




class PCATorch(BaseModel):
    def __init__(self, dataset_obj=None, **kwargs):
        if torch.cuda.is_available():
            self.dev = torch.device('cuda')
        else:
            self.dev = torch.device('cpu')
        # super().__init__()
        self.dataset_obj = dataset_obj
        
        ## filled on fit
        self.transform_matrix = None
        
        ## filled on predict
        self.dist_matx_torch = None

    
    def fit(self, train_data=None, dim=1024):
        
        if train_data is None:
            X = torch.tensor(self.dataset_obj.data_matx[self.dataset_obj.train_idx], device=self.dev)
        else:
            X = torch.tensor(train_data, device=self.dev)

        # mean normalisation
        X_mean = torch.mean(X,0)
        X = X - X_mean.expand_as(X)

       

        # svd, need to transpose x so each data is in one col now
        U,S,_ = torch.svd(torch.t(X))

        self.transform_matrix = U[:,:dim]
        self.mean_vect = X_mean
        
        return self
    

    def transform(self, data):
        if self.transform_matrix is None: raise RuntimeError("Please call fit first before calling transform.")
        data = torch.tensor(data, device=self.dev) - self.mean_vect.expand_as(data)
        return torch.mm(data, self.transform_matrix)


    ## find distances in the transformed and reduced dim pca subspace
    def predict(self, qry_data=None, gal_data=None):
        if qry_data is None and gal_data is None:
            qry_data = torch.tensor(self.dataset_obj.data_matx[self.dataset_obj.query_idx], device=self.dev)                
            gal_data = torch.tensor(self.dataset_obj.data_matx[self.dataset_obj.gallery_idx], device=self.dev)
        elif (type(qry_data) == np.ndarray and type(gal_data) == np.ndarray) or \
             (type(qry_data) == torch.Tensor and type(gal_data) == torch.Tensor):
            qry_data, gal_data = torch.tensor(qry_data, device=self.dev), torch.tensor(gal_data, device=dev)
        else:
            raise NotImplementedError("Found Types: %a %a. Both types must match." % (type(qry_data), type(gal_data)))
        

        qry_data, gal_data = self.transform(qry_data), self.transform(gal_data)

        # only works with cpu
        self.dist_matx_torch = PairWiseDistTorch(self.dataset_obj).fit_predict(qry_data.cpu(), gal_data.cpu()).dist_matx_torch

        return self

