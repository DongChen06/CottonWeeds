import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


conf_mat = np.genfromtxt('../mnasnet_w.csv', delimiter=',')

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
plt.savefig('mnasnet_w.png')
plt.show()
