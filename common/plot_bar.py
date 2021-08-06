
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')


species = ['Carpetweed', 'Crabgrass', 'Goosegrass', 'Morningglory', 'Nutsedge', 'PalmerAmaranth',
               'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress']

N = 12

train = (524, 35, 126, 385, 156, 388, 56, 90, 100, 52, 41, 49)
val = (151, 10, 37, 111, 45, 112, 16, 26, 29, 15, 13, 15)
bottom = (675,  45, 163, 496, 201, 500,  72, 116, 129,  67,  54,  64)
test = (76, 5, 19, 56, 23, 56, 8, 13, 15, 8, 7, 8)
nums = [751, 50, 182, 552, 224, 556, 80, 129, 144, 75, 61, 72]
ind = np.arange(N)
width = 0.45

fig = plt.subplots(figsize=(10, 7))
p1 = plt.bar(ind, train, width)
p2 = plt.bar(ind, val, width, bottom=train)
p3 = plt.bar(ind, test, width, bottom=bottom)

plt.ylabel("number of Species", fontsize=20)
plt.xticks(ind, ('Carpetweed', 'Crabgrass', 'Goosegrass', 'Morningglory', 'Nutsedge', 'PalmerAmaranth',
               'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress'))
# plt.yticks(np.arange(0, 81, 10))
plt.legend(('train', 'val', 'test'), fontsize=18)
plt.xticks(rotation=60, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
addlabels(ind, nums)
plt.savefig('dataset.png')
plt.show()
