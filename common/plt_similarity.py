"""ref: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"


vegetables = ['Carpetweed', 'Crabgrass', 'Goosegrass', 'Morningglory', 'Nutsedge', 'PalmerAmaranth',
              'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress']
farmers = ['Carpetweed', 'Crabgrass', 'Goosegrass', 'Morningglory', 'Nutsedge', 'PalmerAmaranth',
           'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress']

harvest = np.array([[1, 0.5513525, 0.54049176, 0.54774076, 0.4881417, 0.5461825, 0.5543379, 0.513075,
                     0.5383335, 0.55092794, 0.5299297, 0.53675777],
                    [0.5513525, 1, 0.67786133, 0.5272321, 0.6121038, 0.50892854, 0.53197664, 0.5115813, 0.48685187,
                     0.5660697, 0.502544, 0.47039053],
                    [0.54049176, 0.67786133, 1, 0.49323675, 0.60704565, 0.49297047, 0.5044459, 0.50126606, 0.4660865,
                     0.53984284, 0.47153124, 0.49544746],
                    [0.54774076, 0.5272321, 0.49323675, 1, 0.4610599, 0.61521256, 0.5687529, 0.4880359, 0.58376735,
                     0.5567023, 0.64139354, 0.5162961],
                    [0.4881417, 0.6121038, 0.60704565, 0.4610599, 1, 0.43753168, 0.46606866, 0.46402296, 0.4358546,
                     0.49089375, 0.4410367, 0.44206482],
                    [0.5461825, 0.50892854, 0.49297047, 0.61521256, 0.43753168, 1, 0.55106497, 0.51951146,
                     0.5946175, 0.5435123, 0.6206482, 0.5511815],
                    [0.5543379, 0.53197664, 0.5044459, 0.5687529, 0.46606866, 0.55106497, 1, 0.5257276, 0.5675584,
                     0.5718493, 0.55295295, 0.5529324],
                    [0.513075, 0.5115813, 0.50126606, 0.4880359, 0.46402296, 0.51951146, 0.5257276, 1, 0.5113702,
                     0.5170228, 0.5088189, 0.5648251],
                    [0.5383335, 0.48685187, 0.4660865, 0.58376735, 0.4358546, 0.5946175, 0.5675584, 0.5113702, 1,
                     0.54159397, 0.57934535, 0.5471926],
                    [0.55092794, 0.5660697, 0.53984284, 0.5567023, 0.49089375, 0.5435123, 0.5718493, 0.5170228,
                     0.54159397, 1, 0.5241013, 0.5340295],
                    [0.5299297, 0.502544, 0.47153124, 0.64139354, 0.4410367, 0.6206482, 0.55295295, 0.5088189,
                     0.57934535, 0.5241013, 1, 0.51286024],
                    [0.53675777, 0.47039053, 0.49544746, 0.5162961, 0.44206482, 0.5511815, 0.5529324, 0.5648251,
                     0.5471926, 0.5340295, 0.51286024, 1]])

# harvest = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
#                      np.nan, np.nan, np.nan, np.nan],
#                     [0.5513525, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
#                      np.nan, np.nan, np.nan, np.nan],
#                     [0.54049176, 0.67786133, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
#                      np.nan, np.nan, np.nan, np.nan],
#                     [0.54774076, 0.5272321, 0.49323675, np.nan, np.nan, np.nan, np.nan, np.nan,
#                      np.nan, np.nan, np.nan, np.nan],
#                     [0.4881417, 0.6121038, 0.60704565, 0.4610599, np.nan, np.nan, np.nan, np.nan,
#                      np.nan, np.nan, np.nan, np.nan],
#                     [0.5461825, 0.50892854, 0.49297047, 0.61521256, 0.43753168, np.nan, np.nan, np.nan,
#                      np.nan, np.nan, np.nan, np.nan],
#                     [0.5543379, 0.53197664, 0.5044459, 0.5687529, 0.46606866, 0.55106497, np.nan, np.nan,
#                      np.nan, np.nan, np.nan, np.nan],
#                     [0.513075, 0.5115813, 0.50126606, 0.4880359, 0.46402296, 0.51951146, 0.5257276, np.nan,
#                      np.nan, np.nan, np.nan, np.nan],
#                     [0.5383335, 0.48685187, 0.4660865, 0.58376735, 0.4358546, 0.5946175, 0.5675584, 0.5113702, np.nan,
#                      np.nan, np.nan, np.nan],
#                     [0.55092794, 0.5660697, 0.53984284, 0.5567023, 0.49089375, 0.5435123, 0.5718493, 0.5170228,
#                      0.54159397, np.nan, np.nan, np.nan],
#                     [0.5299297, 0.502544, 0.47153124, 0.64139354, 0.4410367, 0.6206482, 0.55295295, 0.5088189,
#                      0.57934535, 0.5241013, np.nan, np.nan],
#                     [0.53675777, 0.47039053, 0.49544746, 0.5162961, 0.44206482, 0.5511815, 0.5529324, 0.5648251,
#                      0.5471926, 0.5340295, 0.51286024, np.nan]])

harvest = np.around(harvest, decimals=2)

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(harvest)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('', rotation=-90, va="bottom")

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Similarity Matrix over Different Species")
fig.tight_layout()
plt.savefig('similarity.pdf')
plt.show()
