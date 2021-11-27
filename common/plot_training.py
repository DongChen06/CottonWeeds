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
legend_size = 17.7
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

densenet121_0 = np.genfromtxt('plots/acc_csv/run-densenet121_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet121_1 = np.genfromtxt('plots/acc_csv/run-densenet121_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet121_2 = np.genfromtxt('plots/acc_csv/run-densenet121_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet121_3 = np.genfromtxt('plots/acc_csv/run-densenet121_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet121_4 = np.genfromtxt('plots/acc_csv/run-densenet121_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet121 = (densenet121_0 + densenet121_1 + densenet121_2 + densenet121_3 + densenet121_4) / 5

densenet161_0 = np.genfromtxt('plots/acc_csv/run-densenet161_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet161_1 = np.genfromtxt('plots/acc_csv/run-densenet161_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet161_2 = np.genfromtxt('plots/acc_csv/run-densenet161_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet161_3 = np.genfromtxt('plots/acc_csv/run-densenet161_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet161_4 = np.genfromtxt('plots/acc_csv/run-densenet161_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet161 = (densenet161_0 + densenet161_1 + densenet161_2 + densenet161_3 + densenet161_4) / 5


resnet101_0 = np.genfromtxt('plots/acc_csv/run-resnet101_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_1 = np.genfromtxt('plots/acc_csv/run-resnet101_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_2 = np.genfromtxt('plots/acc_csv/run-resnet101_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_3 = np.genfromtxt('plots/acc_csv/run-resnet101_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_4 = np.genfromtxt('plots/acc_csv/run-resnet101_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101 = (resnet101_0 + resnet101_1 + resnet101_2 + resnet101_3 + resnet101_4) / 5

densenet169_0 = np.genfromtxt('plots/acc_csv/run-densenet169_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet169_1 = np.genfromtxt('plots/acc_csv/run-densenet169_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet169_2 = np.genfromtxt('plots/acc_csv/run-densenet169_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet169_3 = np.genfromtxt('plots/acc_csv/run-densenet169_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet169_4 = np.genfromtxt('plots/acc_csv/run-densenet169_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet169 = (densenet169_0 + densenet169_1 +densenet169_2 + densenet169_3 + densenet169_4) / 5

resnet50_0 = np.genfromtxt('plots/acc_csv/run-resnet50_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50_1 = np.genfromtxt('plots/acc_csv/run-resnet50_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50_2 = np.genfromtxt('plots/acc_csv/run-resnet50_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50_3 = np.genfromtxt('plots/acc_csv/run-resnet50_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50_4 = np.genfromtxt('plots/acc_csv/run-resnet50_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50 = (resnet50_0 + resnet50_1 + resnet50_2 + resnet50_3 + resnet50_4) / 5

resnet18_0 = np.genfromtxt('plots/acc_csv/run-resnet18_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18_1 = np.genfromtxt('plots/acc_csv/run-resnet18_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18_2 = np.genfromtxt('plots/acc_csv/run-resnet18_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18_3 = np.genfromtxt('plots/acc_csv/run-resnet18_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18_4 = np.genfromtxt('plots/acc_csv/run-resnet18_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18 = (resnet18_0 + resnet18_1 + resnet18_2 + resnet18_3 + resnet18_4) / 5

vgg19_0 = np.genfromtxt('plots/acc_csv/run-vgg19_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
vgg19_1 = np.genfromtxt('plots/acc_csv/run-vgg19_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
vgg19_2 = np.genfromtxt('plots/acc_csv/run-vgg19_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
vgg19_3 = np.genfromtxt('plots/acc_csv/run-vgg19_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
vgg19_4 = np.genfromtxt('plots/acc_csv/run-vgg19_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
vgg19 = (vgg19_0 + vgg19_1 + vgg19_2 + vgg19_3 + vgg19_4) / 5

inception_0 = np.genfromtxt('plots/acc_csv/run-inception_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
inception_1 = np.genfromtxt('plots/acc_csv/run-inception_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
inception_2 = np.genfromtxt('plots/acc_csv/run-inception_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
inception_3 = np.genfromtxt('plots/acc_csv/run-inception_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
inception_4 = np.genfromtxt('plots/acc_csv/run-inception_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
inception = (inception_0 + inception_1 + inception_2 + inception_3 + inception_4) / 5

mobilenet_v2_0 = np.genfromtxt('plots/acc_csv/run-mobilenet_v2_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
mobilenet_v2_1 = np.genfromtxt('plots/acc_csv/run-mobilenet_v2_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
mobilenet_v2_2 = np.genfromtxt('plots/acc_csv/run-mobilenet_v2_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
mobilenet_v2_3 = np.genfromtxt('plots/acc_csv/run-mobilenet_v2_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
mobilenet_v2_4 = np.genfromtxt('plots/acc_csv/run-mobilenet_v2_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
mobilenet_v2 = (mobilenet_v2_0 + mobilenet_v2_1 + mobilenet_v2_2 + mobilenet_v2_3 + mobilenet_v2_4) / 5


Y = np.array([smooth(dpn68), smooth(resnet101), smooth(resnet50), smooth(densenet161),  smooth(densenet169), smooth(densenet121),
              smooth(inception), smooth(mobilenet_v2), smooth(resnet18), smooth(vgg19)])

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
    ['DPN68', 'ResNet101', 'ResNet50', 'DenseNet161', 'DenseNet169', 'DenseNet121', 'Inception-v3', 'MobileNet-v2',
     'ResNet18', 'VGG16'], ncol=3, fontsize=legend_size)

plt.tight_layout()
plt.savefig('train_accv2.pdf')
plt.savefig('train_accv2.png')
plt.show()
