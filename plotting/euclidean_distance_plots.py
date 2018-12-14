from dataset.dataset_loader import DatasetLoader
from matplotlib import pyplot as plt

import numpy as np


data = DatasetLoader()

from methods.baseline_models import PairWiseDistTorch

knn = PairWiseDistTorch(data)
result = knn.fit_predict().score(max_rank=100).ranked_acc
print(result[0], result[9])


fig, ax = plt.subplots()
ax.plot(result, '-.', label='Rank-n Accuracy')
#ax.axvline(mean, color='r', label=r'$\mu$')
ax.set_xlabel('n')
ax.set_ylabel(r'$P(Rank)\leq n$')
ax.set_title('CMC for Euclidean Distance')
ax.set_xlim(xmin=0)
ax.grid()

# textstr = r'$\mu=9.61$'

ax.annotate('Rank-1=', xy=(1350, 5), xytext=(1000, 5.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ax.axhline(result[0], color='r', label=r'Rank-1')
ax.axhline(result[9], color='g', label=r'Rank-10')
ax.axhline(result[49], color='b', label=r'Rank-50')
ax.legend(loc=(0.6,0.2))

x = (0.95699-0.448)

props = dict(boxstyle=None, facecolor='wheat', alpha=0.5)
ax.text(0.2, (result[0]-0.448)/x+0.07, '47%', transform=ax.transAxes, fontsize=14,
     verticalalignment='top')
ax.text(0.2, (result[9]-0.448)/x+0.06, '74.93%', transform=ax.transAxes, fontsize=14,
     verticalalignment='top')
ax.text(0.2, (result[99]-0.448)/x-0.04, '89.79%', transform=ax.transAxes, fontsize=14,
     verticalalignment='top')

plt.show()
