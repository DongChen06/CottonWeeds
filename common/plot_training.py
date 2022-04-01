import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.rcParams["font.family"] = "Times New Roman"


def smooth(x, timestamps=3):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
color_cycle = sns.color_palette("husl", 10)
sns.set_color_codes()

colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

alpha = 0.3
legend_size = 16.7
line_size_others = 1.5
line_size_ours = 1.5
tick_size = 20
label_size = 20


def cal_collision_rate(l):
    l = list(l)
    output = [l[0]]
    for i in range(100, len(l) - 1):
        output.append(1 - sum(np.array(l[i - 100:i]) == 100) / len(l[i - 100:i]))
    return np.array(output)


X = np.arange(50)
dpn68_0 = np.genfromtxt('plots/acc_csv/run-dpn68_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
dpn68_1 = np.genfromtxt('plots/acc_csv/run-dpn68_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
dpn68_2 = np.genfromtxt('plots/acc_csv/run-dpn68_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
dpn68_3 = np.genfromtxt('plots/acc_csv/run-dpn68_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
dpn68_4 = np.genfromtxt('plots/acc_csv/run-dpn68_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
dpn68 = (dpn68_0 + dpn68_1 + dpn68_2 + dpn68_3 + dpn68_4) / 5

ResNext50_0 = np.genfromtxt('plots/acc_csv/run-resnext50_32x4d_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext50_1 = np.genfromtxt('plots/acc_csv/run-resnext50_32x4d_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext50_2 = np.genfromtxt('plots/acc_csv/run-resnext50_32x4d_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext50_3 = np.genfromtxt('plots/acc_csv/run-resnext50_32x4d_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext50_4 = np.genfromtxt('plots/acc_csv/run-resnext50_32x4d_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext50 = (ResNext50_0 + ResNext50_1 + ResNext50_2 + ResNext50_3 + ResNext50_4) / 5

ResNext101_0 = np.genfromtxt('plots/acc_csv/run-resnext101_32x8d_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext101_1 = np.genfromtxt('plots/acc_csv/run-resnext101_32x8d_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext101_2 = np.genfromtxt('plots/acc_csv/run-resnext101_32x8d_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext101_3 = np.genfromtxt('plots/acc_csv/run-resnext101_32x8d_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext101_4 = np.genfromtxt('plots/acc_csv/run-resnext101_32x8d_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
ResNext101 = (ResNext101_0 + ResNext101_1 + ResNext101_2 + ResNext101_3 + ResNext101_4) / 5


resnet101_0 = np.genfromtxt('plots/acc_csv/run-resnet101_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_1 = np.genfromtxt('plots/acc_csv/run-resnet101_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_2 = np.genfromtxt('plots/acc_csv/run-resnet101_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_3 = np.genfromtxt('plots/acc_csv/run-resnet101_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_4 = np.genfromtxt('plots/acc_csv/run-resnet101_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101 = (resnet101_0 + resnet101_1 + resnet101_2 + resnet101_3 + resnet101_4) / 5

RepVGG_A0_0 = np.genfromtxt('plots/acc_csv/run-RepVGG-A0_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A0_0_1 = np.genfromtxt('plots/acc_csv/run-RepVGG-A0_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A0_0_2 = np.genfromtxt('plots/acc_csv/run-RepVGG-A0_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A0_0_3 = np.genfromtxt('plots/acc_csv/run-RepVGG-A0_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A0_0_4 = np.genfromtxt('plots/acc_csv/run-RepVGG-A0_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A0_0 = (RepVGG_A0_0 + RepVGG_A0_0_1 +RepVGG_A0_0_2 + RepVGG_A0_0_3 + RepVGG_A0_0_4) / 5

RepVGG_A1_0 = np.genfromtxt('plots/acc_csv/run-RepVGG-A1_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A1_1 = np.genfromtxt('plots/acc_csv/run-RepVGG-A1_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A1_2 = np.genfromtxt('plots/acc_csv/run-RepVGG-A1_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A1_3 = np.genfromtxt('plots/acc_csv/run-RepVGG-A1_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A1_4 = np.genfromtxt('plots/acc_csv/run-RepVGG-A1_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A1 = (RepVGG_A1_0 + RepVGG_A1_1 + RepVGG_A1_2 + RepVGG_A1_3 + RepVGG_A1_4) / 5

RepVGG_A2_0 = np.genfromtxt('plots/acc_csv/run-RepVGG-A2_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A2_1 = np.genfromtxt('plots/acc_csv/run-RepVGG-A2_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A2_2 = np.genfromtxt('plots/acc_csv/run-RepVGG-A2_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A2_3 = np.genfromtxt('plots/acc_csv/run-RepVGG-A2_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A2_4 = np.genfromtxt('plots/acc_csv/run-RepVGG-A2_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_A2 = (RepVGG_A2_0 + RepVGG_A2_1 + RepVGG_A2_2 + RepVGG_A2_3 + RepVGG_A2_4) / 5

RepVGG_B0_0 = np.genfromtxt('plots/acc_csv/run-RepVGG-B0_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B0_1 = np.genfromtxt('plots/acc_csv/run-RepVGG-B0_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B0_2 = np.genfromtxt('plots/acc_csv/run-RepVGG-B0_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B0_3 = np.genfromtxt('plots/acc_csv/run-RepVGG-B0_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B0_4 = np.genfromtxt('plots/acc_csv/run-RepVGG-B0_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B0 = (RepVGG_B0_0 + RepVGG_B0_1 + RepVGG_B0_2 + RepVGG_B0_3 + RepVGG_B0_4) / 5

RepVGG_B1_0 = np.genfromtxt('plots/acc_csv/run-RepVGG-B1_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B1_1 = np.genfromtxt('plots/acc_csv/run-RepVGG-B1_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B1_2 = np.genfromtxt('plots/acc_csv/run-RepVGG-B1_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B1_3 = np.genfromtxt('plots/acc_csv/run-RepVGG-B1_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B1_4 = np.genfromtxt('plots/acc_csv/run-RepVGG-B1_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B1 = (RepVGG_B1_0 + RepVGG_B1_1 + RepVGG_B1_2 + RepVGG_B1_3 + RepVGG_B1_4) / 5

RepVGG_B2_0 = np.genfromtxt('plots/acc_csv/run-RepVGG-B2_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B2_1 = np.genfromtxt('plots/acc_csv/run-RepVGG-B2_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B2_2 = np.genfromtxt('plots/acc_csv/run-RepVGG-B2_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B2_3 = np.genfromtxt('plots/acc_csv/run-RepVGG-B2_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B2_4 = np.genfromtxt('plots/acc_csv/run-RepVGG-B2_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
RepVGG_B2 = (RepVGG_B2_0 + RepVGG_B2_1 + RepVGG_B2_2 + RepVGG_B2_3 + RepVGG_B2_4) / 5

Y = np.array([smooth(dpn68), smooth(resnet101), smooth(RepVGG_A1), smooth(ResNext101),  smooth(RepVGG_A0_0), smooth(ResNext50),
              smooth(RepVGG_B1), smooth(RepVGG_B2), smooth(RepVGG_A2), smooth(RepVGG_B0)])

######################################
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.plot(X, Y.T * 100, lw=line_size_others)
ax.set_xlim(1, 50)
ax.set_ylim(70, 100)
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
ax.set_xlabel('Training epochs', fontsize=20)
ax.set_ylabel('F1-score  (%)', fontsize=20)
ax.ticklabel_format(axis="x")
ax.grid()

axins2 = zoomed_inset_axes(ax, zoom=6, loc=7)
axins2.plot(X, Y.T * 100)
# SPECIFY THE LIMITS
x1, x2, y1, y2 = 46, 49.2, 96.7, 98.3
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
axins2.axes.xaxis.set_visible(False)
# axins2.axes.yaxis.set_visible(False)
axins2.tick_params(axis='y', labelsize=tick_size)
mark_inset(ax, axins2, loc1=2, loc2=4, fc="None", lw=1.5, ec="k")

ax.legend(
    ['ResNext101', 'RepVGG-B1', 'RepVGG-B2', 'ResNext50', 'RepVGG-A2', 'RepVGG-B0', 'RepVGG-A1', 'RepVGG-A0',
     'DPN68', 'ResNet101'], ncol=3, fontsize=legend_size)

plt.tight_layout()
plt.savefig('train_accv2.pdf')
plt.savefig('train_accv2.png')
plt.show()
