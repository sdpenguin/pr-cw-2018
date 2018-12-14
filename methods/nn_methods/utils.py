from itertools import combinations

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    loss values are calc as batch so essentially the hardest negative will really depend on the batch itself
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)


class BasicDataset(object):
    def __init__(self, train, train_labels=None, train_data=None, test_labels=None, test_data=None):
        self.train = train  # boolean
        
        if train: 
            self.train_labels = torch.tensor(np.array(train_labels, dtype=np.int32))
            self.train_data = torch.tensor(train_data)

        else:
            self.test_labels = torch.tensor(np.array(test_labels, dtype=np.int32)) if test_labels is not None else None
            self.test_data = torch.tensor(test_data)
        self.length = train_labels.shape[0] if train else test_labels.shape[0]
    
    def __getitem__(self, index):
        if self.train:
            return self.train_data[index], self.train_labels[index]
        else:
            return self.test_data[index], self.test_labels[index]
    
    def __len__(self):
        return self.length

 
class DatasetLoaderTorch(Dataset):
    ''' A Pytorch-based derived class for dataset handling. '''
    ''' Takes a DatasetLoader object as init param. '''
    '''
        Train: For each sample (anchor) randomly chooses a positive and negative samples
        Test: Creates fixed triplets for testing

        Code adapted from https://github.com/adambielski/siamese-triplet
    '''
    def __init__(self, dataset_obj, is_train=True, transform=None):
        self.is_train = is_train
        self.len_dataset = dataset_obj.len_dataset
        self.len_trainset = dataset_obj.len_trainset
        self.len_valset = dataset_obj.len_valset

        self.transform = transform

        if self.is_train:
            ## this data is automatically split to train+val if we call it with is_train=False
            self.train_labels = torch.from_numpy(np.array(dataset_obj.train_labels, dtype=np.int32))
            self.train_cam_idx = torch.from_numpy(np.array(dataset_obj.train_cam_idx, dtype=np.int32))
            self.train_data = torch.from_numpy(dataset_obj.data_matx[dataset_obj.train_idx])

            if transform is not None:
                ## do transform on entire dataset
                print("Pre-transform: ", self.train_data.shape, end='')
                self.train_data = transform(self.train_data)
                print("\tPost-transform: ", self.train_data.shape)

            ### get all unique class labels in train_labels
            self.labels_set = set(self.train_labels.numpy())

            ### get the list of train_label indices for each class
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            # generate fixed pairs for testing --- actually validation
            # we use val for this
            self.val_labels = torch.from_numpy(np.array(dataset_obj.val_labels, dtype=np.int32))
            self.val_cam_idx = torch.from_numpy(np.array(dataset_obj.val_cam_idx, dtype=np.int32))
            self.val_data = torch.from_numpy(dataset_obj.data_matx[dataset_obj.val_idx])

            if transform is not None:
                ## do transform on entire dataset
                print("Pre-transform: ", self.val_data.shape[0] , end='')
                self.val_data = transform(self.val_data)
                print("\tPost-transform: ", self.val_data.shape[0])
            
            self.labels_set = set(self.val_labels.numpy())
            self.label_to_indices = {label: np.where(self.val_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            ## do pre-calc of pairs
            random_state = np.random.RandomState(29)
            
            ## code from siamese entworks pytorch
            ## create a triplet of anchor, positive and negative samples
            ## 
            ## self.val_labels[i].item() gets the classlabel of item i
            ## self.label_to_indices[]...] gets all indices of that class label in val set
            ## 
            ## for neg sample the same happens but the valid classlabels are now all possible class
            ## labels except classlabel of item i
            ## in the end whats returned is i, i_pos, i_neg as a list
            ## which are indices relative to the validation set
            ## random choice chooses one index at random
            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.val_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.val_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.val_data))]
            self.val_triplets = triplets

        #self.train_labels = dataset_obj.train_labels


    def __getitem__(self, index):
        if self.is_train:
            sample1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index

            # while loop ensures same sample is not selected as positive
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            #print("Index, pos, neg: ", index, positive_index, negative_index)
            #print("Label Index, pos, neg: ", self.train_labels[index], self.train_labels[positive_index], self.train_labels[negative_index])
            

            sample2 = self.train_data[positive_index]
            sample3 = self.train_data[negative_index]
        else:
            sample1 = self.val_data[self.val_triplets[index][0]]
            sample2 = self.val_data[self.val_triplets[index][1]]
            sample3 = self.val_data[self.val_triplets[index][2]]

        return (sample1, sample2, sample3), []

    def __len__(self):
        return self.len_trainset if self.is_train else self.len_valset



class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.dataset.length:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
