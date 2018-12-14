from scipy.io import loadmat
from urllib.request import urlopen
from os.path import isfile
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import matplotlib.pyplot as plt

import numpy as np
import torch


class DatasetLoader(object):
    ''' An object containing the data and associated methods.'''

    def __init__(self, spec_file='dataset/cuhk03_new_protocol_config_labeled.mat', data_file='dataset/feature_data.npy',
                do_val_split=False):
        
        self.transform = None # for compatibility with pytorch friendly datasets

        ## for compatibility
        self.val_idx = np.empty(0)

        matContents = loadmat(spec_file)
        if isfile(data_file):
            self.data_matx = np.load(data_file)
        else:
            print("slow loading")
            import json
            with open(data_file[:-4] + '.json', 'r') as f:
                self.data_matx = np.array(json.load(f))
                np.save(data_file, self.data_matx)

        self.cam_idx = matContents['camId'].flatten()  # 1d array
        self.filelist = matContents['filelist'].flatten()
        self.labels = matContents['labels'].flatten()

        # Matlab indexes from 1, so we take off 1 to start it from 0
        self.query_idx = matContents['query_idx'].flatten() - 1
        self.gallery_idx = matContents['gallery_idx'].flatten() - 1
        self.train_idx = matContents['train_idx'].flatten() - 1

        # Use the indexes to get the different data parts
        self.query_labels = self.labels[self.query_idx]
        self.gallery_labels = self.labels[self.gallery_idx]
        self.train_labels = self.labels[self.train_idx]

        # Camera Ids
        self.query_cam_idx = self.cam_idx[self.query_idx]
        self.gallery_cam_idx = self.cam_idx[self.gallery_idx]
        self.train_cam_idx = self.cam_idx[self.train_idx]

        # Tensor forms for Pytorch
        self.labels_torch = torch.from_numpy(
            np.array(self.labels, dtype=np.int32))
        self.gallery_idx_torch = torch.from_numpy(
            np.array(self.gallery_idx, dtype=np.int32))
        self.query_labels_torch = torch.from_numpy(
            np.array(self.query_labels, dtype=np.int32))
        self.gallery_labels_torch = torch.from_numpy(
            np.array(self.gallery_labels, dtype=np.int32))

        if do_val_split:
            # keep seed=1 for consistency
            # +1 as class labels use 1-based indexing
            # take only from trian_labels
            val_labels = np.random.choice(np.unique(self.train_labels), size=100, replace=False)
            
            # boolean array with True = sample now in validation set
            # get all samples of validation classes

            val_mask = np.in1d(self.train_labels, val_labels)

            # took all occurrences of 100 unique identities for validation and removed all occurences of them from training
            self.val_labels = self.train_labels[val_mask]
            self.train_labels = self.train_labels[~val_mask]

            # find the ids corresponding to these labels
            self.val_idx    = self.train_idx[val_mask]  
            self.train_idx  = self.train_idx[~val_mask]

            # find the cam_ids corresponding to these labels
            self.val_cam_idx = self.train_cam_idx[val_mask]
            self.train_cam_idx = self.train_cam_idx[~val_mask]


    def get_valid_test_gallery_ids(self, query_label, query_cam_id, dtype=int):
        ''' Return the test gallery indices with invalid item indices removed.'''
        return self.gallery_idx[DatasetLoader._is_valid_id(query_label, query_cam_id,
                                                           self.gallery_labels, self.gallery_cam_idx)].astype(dtype)

    def get_valid_test_gallery_mask(self, query_label, query_cam_id, dtype=bool):
        '''Return the a mask for the test gallery, with invalid items being False.'''
        return np.array(DatasetLoader._is_valid_id(query_label, query_cam_id,
                                                   self.gallery_labels, self.gallery_cam_idx), dtype=dtype)

    @staticmethod
    def _is_valid_id(query_label, query_cam_idx, gal_lbl_arr, gal_cam_idx_arr):
        # Returns true if the id is valid
        return (gal_lbl_arr != query_label) | ((gal_cam_idx_arr != query_cam_idx) & (gal_lbl_arr == query_label))
        # TODO: assert verify this somewhere...

    #####
    # Properties: these are functions that can be accessed as properties.
    # Useful for quick stats about the object.
    #####

    @property
    def unique_labels(self):
        # The number of unique IDs in the gallery
        unique_labels = []
        for x in self.labels:
            if x not in unique_labels:
                unique_labels.append(x)
        return len(unique_labels)

    @property
    def unique_gallery_labels(self):
        # The number of unique IDs in the gallery
        unique_labels = []
        for x in self.gallery_labels:
            if x not in unique_labels:
                unique_labels.append(x)
        return len(unique_labels)

    @property
    def unique_train_labels(self):
        # The number of unique IDs in the gallery
        unique_labels = []
        for x in self.train_labels:
            if x not in unique_labels:
                unique_labels.append(x)
        return len(unique_labels)

    @property
    def unique_query_labels(self):
        # The number of unique IDs in the gallery
        unique_labels = []
        for x in self.query_labels:
            if x not in unique_labels:
                unique_labels.append(x)
        return len(unique_labels)

    @property
    def len_valset(self):
        # The number val samples
        # Return 0 if not splitting to train and val set
        # the return zero is done by setting this to np.empty at start
        return self.val_idx.shape[0]
    
    @property
    def len_trainset(self):
        # The number val samples
        # Return 0 if not splitting to train and val set
        return self.train_idx.shape[0]
    
    @property
    def len_dataset(self):
        # The number total datasamples
        return self.data_matx.shape[0]

