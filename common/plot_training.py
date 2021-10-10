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
legend_size = 15
line_size_others = 1.5
line_size_ours = 1.5
tick_size = 18
label_size = 18


def cal_collision_rate(l):
    l = list(l)
    output = [l[0]]
    for i in range(100, len(l) - 1):
        output.append(1 - sum(np.array(l[i - 100:i]) == 100) / len(l[i - 100:i]))
    return np.array(output)


X = np.arange(50)
alexnet = np.genfromtxt('plots/acc_csv/run-alexnet_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
squeezenet = np.genfromtxt('plots/acc_csv/run-squeezenet_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
googlenet = np.genfromtxt('plots/acc_csv/run-googlenet_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
xception = np.genfromtxt('plots/acc_csv/run-xception_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
dpn68 = np.genfromtxt('plots/acc_csv/run-dpn68_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet121 = np.genfromtxt('plots/acc_csv/run-densenet121_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet161 = np.genfromtxt('plots/acc_csv/run-densenet161_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101 = np.genfromtxt('plots/acc_csv/run-resnet101_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
densenet169 = np.genfromtxt('plots/acc_csv/run-densenet169_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50 = np.genfromtxt('plots/acc_csv/run-resnet50_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18 = np.genfromtxt('plots/acc_csv/run-resnet18_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
vgg11 = np.genfromtxt('plots/acc_csv/run-vgg11_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
vgg16 = np.genfromtxt('plots/acc_csv/run-vgg16_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
vgg19 = np.genfromtxt('plots/acc_csv/run-vgg19_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
inception = np.genfromtxt('plots/acc_csv/run-inception_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
inceptionv4 = np.genfromtxt('plots/acc_csv/run-inceptionv4_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
inceptionresnetv2 = np.genfromtxt('plots/acc_csv/run-inceptionresnetv2_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
mobilenet_v2 = np.genfromtxt('plots/acc_csv/run-mobilenet_v2_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
mobilenet_v3_small = np.genfromtxt('plots/acc_csv/run-mobilenet_v3_small_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
mobilenet_v3_large = np.genfromtxt('plots/acc_csv/run-mobilenet_v3_large_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']

mnasnet1_0 = np.genfromtxt('plots/acc_csv/run-mnasnet1_0_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
efficientnet_b0 = np.genfromtxt('plots/acc_csv/run-efficientnet-b0_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
efficientnet_b1 = np.genfromtxt('plots/acc_csv/run-efficientnet-b1_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
efficientnet_b2 = np.genfromtxt('plots/acc_csv/run-efficientnet-b2_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
efficientnet_b3 = np.genfromtxt('plots/acc_csv/run-efficientnet-b3_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
efficientnet_b4 = np.genfromtxt('plots/acc_csv/run-efficientnet-b4_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
efficientnet_b5 = np.genfromtxt('plots/acc_csv/run-efficientnet-b5_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']

Y = np.array([smooth(dpn68), smooth(densenet161), smooth(resnet101), smooth(densenet169), smooth(resnet50),
                smooth(vgg19), smooth(densenet121), smooth(vgg16), smooth(mnasnet1_0),
              smooth(alexnet), smooth(squeezenet), smooth(googlenet), smooth(xception), smooth(dpn68),  smooth(resnet18), smooth(vgg11),
             smooth(inception), smooth(inceptionv4), smooth(inceptionresnetv2),
            smooth(mobilenet_v2), smooth(mobilenet_v3_small), smooth(mobilenet_v3_large), smooth(efficientnet_b3), smooth(efficientnet_b4),
              smooth(efficientnet_b0), smooth(efficientnet_b1), smooth(efficientnet_b2), smooth(efficientnet_b5)])

######################################
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X, Y.T * 100, lw=line_size_others)
ax.set_xlim(1, 50)
ax.set_ylim(65, 100)
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
ax.set_xlabel('Training epochs', fontsize=20)
ax.set_ylabel('F1-score  (%)', fontsize=20)
ax.ticklabel_format(axis="x")
ax.grid()

axins2 = zoomed_inset_axes(ax, zoom=6, loc=7)
axins2.plot(X, Y.T*100)
# SPECIFY THE LIMITS
x1, x2, y1, y2 = 46, 49.2, 96.5, 98.4
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
axins2.axes.xaxis.set_visible(False)
# axins2.axes.yaxis.set_visible(False)
axins2.tick_params(axis='y', labelsize=tick_size)
mark_inset(ax, axins2, loc1=2, loc2=4, fc="None", lw=1.5, ec="k")

ax.legend(['DPN68', 'DenseNet161', 'ResNet101', 'DenseNet169', 'ResNet50', 'VGG19', 'DenseNet121', 'VGG16', 'MobileNetv2', 'VGG11'],
          ncol=3, fontsize=legend_size)
plt.tight_layout()
plt.savefig('train_acc.pdf')
plt.show()

