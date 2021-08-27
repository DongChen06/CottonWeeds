import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


# function to add value labels
def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] + 5, y[i], ha='center')


species = ['Morningglory', 'Carpetweed', 'Palmer', 'Amaranth', 'Waterhemp', 'Purslane', 'Nutsedge', 'Eclipta',
           'Spotted', 'Spurge', 'Sicklepod', 'Goosegrass', 'Prickly',
           'Sida', 'Ragweed', 'Crabgrass', 'Swinecress', 'Spurred', 'Anoda']

N = 15

data = (1115, 763, 689, 451, 450, 273, 254, 234, 240, 216, 129, 129, 111, 72, 61)

ind = np.arange(N)
width = 0.45

fig = plt.subplots(figsize=(10, 7))
p1 = plt.bar(ind, data, width)

plt.ylabel("Number of Species", fontsize=20)
plt.xticks(ind, ('Morningglory', 'Carpetweed', 'Palmer', 'Amaranth', 'Waterhemp', 'Purslane', 'Nutsedge', 'Eclipta',
           'Spotted', 'Spurge', 'Sicklepod', 'Goosegrass', 'Prickly',
           'Sida', 'Ragweed', 'Crabgrass', 'Swinecress', 'Spurred', 'Anoda'))
# plt.yticks(np.arange(0, 81, 10))
# plt.legend(('train', 'val', 'test'), fontsize=18)
plt.xticks(rotation=60, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
addlabels(ind, data)
plt.savefig('dataset.png')
plt.show()
