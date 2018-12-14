from dataset.dataset_loader import DatasetLoader
from matplotlib import pyplot as plt

import numpy as np


data = DatasetLoader()


label_count = np.bincount(np.bincount(data.labels)[1:])[6:]
mean = np.mean(np.bincount(data.labels)[1:])
# print(mean)
#mean_repeated = [mean for x in range(len(label_count))]
fig, ax = plt.subplots()
ax.bar(range(6, 11), label_count, label='Number of Labels', width=0.7)
ax.axvline(mean, color='r', label=r'$\mu$')
ax.legend(loc='bottom right')
ax.set_xlabel('Count')
ax.set_ylabel('Number of Labels')
ax.set_title('Number of Occurrences of Separate Identities')
ax.set_xlim(xmin=5)
ax.grid()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = r'$\mu=9.61$'
ax.text(0.5, 0.3, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.annotate('minimum=5', xy=(1350, 5), xytext=(1000, 5.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.show()

# Composition of the data

labels = 'Gallery', 'Query', 'Training'
sizes = [5328, 1400, 7368]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Composition of the HUK03 Dataset')
ax1.text(0.27, 0.05, '7368 Identities')
ax1.text(-0.86, 0.32, '5328 Identities')

ax1.annotate('1400 Identities', xy=(-0.354, -0.759), xytext=(-1.483, -0.843),
             arrowprops=dict(facecolor='black'),
             )

plt.show()
