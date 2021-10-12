# https://medium.com/the-owl/magnifying-dense-regions-in-matplotlib-plots-c765db7ba431

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

datapts = np.array([[22.51, 27.99, 27.3, 39.55],
                    [28.23, 28.01, 35.34, 54.78],
                    [36.87, 36.87, 42.57, 55.54],
                    [38.47, 43.13, 49.08, 55.76]])

x = [5, 10, 15, 20]
xlabs = ["5", "10", "15", "20"]
y = [10, 15, 20, 25]
ylabs = ["10", "15", "20", "25"]

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(y, datapts.T)
ax.plot(y, datapts.T, 'yo')
ax.legend(xlabs, title='CURRENT (A)', loc='upper right')
ax.set_xlabel('RESISTANCE (OHM)')
ax.set_ylabel('VOLTAGE (V)')
ax.set_xticks(ticks=y)
ax.set_xticklabels(labels=ylabs)
ax.set_xlim([5, 30])

for dpts, ypt in zip(datapts.T, y):
    for dpt in dpts:
        ax.annotate("%.3f" % (dpt), xy=(ypt, dpt), textcoords='data')

# loc values
# 2---8---1
# |   |   |
# 6---10--5/7
# |   |   |
# 3---9---4

axins1 = zoomed_inset_axes(ax, zoom=5, loc=2)
axins1.plot(y, datapts.T)
axins1.plot(y, datapts.T, 'mo')

# SPECIFY THE LIMITS
x1, x2, y1, y2 = 24, 26, 54.5, 56
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)

# FOR ANNOTATING THE 3 POINTS WE ARE ZOOMING ON
for dpt, ypt in zip(datapts.T[3], [25] * 4):
    if dpt > y1 and dpt < y2:
        axins1.annotate("%.2f" % (dpt), xy=(ypt, dpt), textcoords='data')

# IF SET TO TRUE, TICKS ALONG
# THE TWO AXIS WILL BE VISIBLE
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins1, loc1=1, loc2=4, fc="none", ec="0.5")



axins2 = zoomed_inset_axes(ax, zoom=50, loc=4)
axins2.plot(y, datapts.T)
axins2.plot(y, datapts.T, 'mo')
# SPECIFY THE LIMITS
x1, x2, y1, y2 = 14.95, 15.05, 27.95, 28.05
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)

# ANNOTATE THE MIDDLE TWO POINTS
for dpt, ypt in zip(datapts.T[1], [15] * 4):
    if ypt > y1 and ypt < y2:
        axins2.annotate("%.2f" % (dpt), xy=(ypt, dpt), textcoords='data')

plt.xticks(visible=False)
plt.yticks(visible=False)

mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()