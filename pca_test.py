from metric_learn.lmnn import LMNN

from methods.baseline_models import *
from util import *

from sklearn.kernel_approximation import RBFSampler

from sklearn.metrics.pairwise import *
from sklearn.decomposition import *

data_set_obj = DatasetLoader('dataset/cuhk03_new_protocol_config_labeled.mat', 
                            'dataset/feature_data.npy',  do_val_split=True)

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

#new
train_data  =    torch.tensor(data_set_obj.data_matx[data_set_obj.train_idx], device=dev)
val_data  =    torch.tensor(data_set_obj.data_matx[data_set_obj.val_idx], device=dev)


val_labels = list(data_set_obj.val_labels)
val_idx = data_set_obj.val_idx

val_label_red

valset = set(val_labels)

for item in valset:
    

for i in range(val_labels.shape)

val = list(val_data.numpy())
qry_val = []
for i in range val:
    if v[i] == 


train_data  =    torch.tensor(data_set_obj.data_matx[data_set_obj.train_idx], device=dev)
gal_data    =    torch.tensor(data_set_obj.data_matx[data_set_obj.gallery_idx], device=dev)
qry_data    =    torch.tensor(data_set_obj.data_matx[data_set_obj.query_idx], device=dev)


knn = PairwiseDist(data_set_obj)

scores = []
# for components in [2048, 1024, 512, 256, 128, 64, 32, 16]:
#     pca = PCA(n_components=components)
#     pca.fit(train_data)
    
#     knn.fit_predict(qry_data=pca.transform(qry_data), gal_dat=pca.transform(gal_data)).score(max_rank=1)
#     #print("Transformer: ", transformer, "\nAcc: ", knn.ranked_acc)
#     scores += [[components, knn.ranked_acc]]

# scores = np.array(scores)

# print("Scores: ", scores)

# import matplotlib.pyplot as plt
# simple2DLinePlot(scores[:, 1], scores[:, 0], title="Rank1 Accuracy VS PCA Components", yLbl="Accuracy@Rank-1", xLbl="PCA Components")

scores = []
pd = PairwiseDist(data_set_obj).fit_predict(metric='seuclidean').score(max_rank=10)
print(pd.ranked_acc)

# for metric in ['l2', 'l1', 'braycurtis', 'chebyshev', 'jaccard', 'correlation', 'seuclidean', 'mahalanobis']:
#     pd = PairwiseDist(data_set_obj).fit_predict(metric=metric).score(max_rank=40)
#     scores += ('Metric', pd.ranked_acc)
#     print("Metric: ", metric, "Score: ", pd.ranked_acc)


print("Scores: ", scores)


