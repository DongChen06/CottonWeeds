import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})


# function to add value labels
def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] + 5, y[i], ha='center')


species = ['Morningglory', 'Carpetweed', 'Palmer Amaranth', 'Waterhemp', 'Purslane', 'Nutsedge', 'Eclipta',
           'Spotted Spurge', 'Sicklepod', 'Goosegrass', 'Prickly Sida', 'Ragweed', 'Crabgrass', 'Swinecress', 'Spurred Anoda']

N = 15

nums = (1115, 763, 689, 451, 450, 273, 254, 234, 240, 216, 129, 129, 111, 72, 61)
train = (724, 495, 447, 292, 292, 177, 164, 151, 156, 139, 83, 83, 71, 46, 38)
val = (223, 153, 138, 91, 90, 55, 51, 47, 48, 44, 26, 26, 23, 15, 13)
bottom = (947, 648, 585, 383, 382, 232, 213, 195, 204, 183, 109, 109, 94, 61, 51)
test = (168, 115, 104, 68, 68, 41, 39, 36, 36, 33, 20, 20, 17, 11, 10)

ind = np.arange(N)
width = 0.45

fig = plt.subplots(figsize=(10, 7))
p1 = plt.bar(ind, train, width)
p2 = plt.bar(ind, val, width, bottom=train)
p3 = plt.bar(ind, test, width, bottom=bottom)

plt.ylabel("Number of Species", fontsize=16)
plt.xticks(ind, ('Morningglory', 'Carpetweed', 'Palmer Amaranth', 'Waterhemp', 'Purslane', 'Nutsedge', 'Eclipta',
           'Spotted Spurge', 'Sicklepod', 'Goosegrass', 'Prickly Sida', 'Ragweed', 'Crabgrass', 'Swinecress', 'Spurred Anoda'))
# plt.yticks(np.arange(0, 81, 10))
plt.legend(('train', 'val', 'test'), fontsize=18)
plt.xticks(rotation=60, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
addlabels(ind, nums)
plt.savefig('dataset.png')
plt.show()
