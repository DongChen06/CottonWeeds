import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

conf_mat = np.genfromtxt('dpn68_cm_4.csv', delimiter=',')[1:, 1:]
conf_mat = conf_mat / np.sum(conf_mat, 1).reshape(15, 1)

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
plt.savefig('dpn68_cm_4.pdf')
plt.show()
