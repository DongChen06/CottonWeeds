import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

conf_mat0 = np.genfromtxt('dpn68_cm_0.csv', delimiter=',')[1:, 1:]

conf_mat1 = np.genfromtxt('dpn68_cm_1.csv', delimiter=',')[1:, 1:]

conf_mat2 = np.genfromtxt('dpn68_cm_2.csv', delimiter=',')[1:, 1:]

conf_mat3 = np.genfromtxt('dpn68_cm_3.csv', delimiter=',')[1:, 1:]

conf_mat4 = np.genfromtxt('dpn68_cm_4.csv', delimiter=',')[1:, 1:]

conf_mat = (conf_mat0 + conf_mat1 + conf_mat2 + conf_mat3 + conf_mat4) / 5 / np.sum(conf_mat0, 1).reshape(15, 1)

# Class label names
class_names = ['Carpetweeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory', 'Nutsedge',
               'PalmerAmaranth', 'PricklySida', 'Purslane', 'Ragweed', 'Sicklepod',
                'SpottedSpurge', 'SpurredAnoda', 'Swinecress', 'Waterhemp']


plt.figure(figsize=(12, 6))
df_cm = pd.DataFrame(conf_mat, index=class_names,
                     columns=class_names)
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap='Greens',  fmt='0.2f')
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.savefig('dpn68_cm.pdf')
plt.savefig('dpn68_cm.png')
plt.show()
