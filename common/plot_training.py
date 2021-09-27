import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def smooth(x, timestamps=2):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
color_cycle = sns.color_palette()
sns.set_color_codes()

colors = [0, 5, 2, 6, 1, 3]

alpha = 0.3
legend_size = 12
line_size_others = 1.5
line_size_ours = 1.5
tick_size = 12
label_size = 14


def cal_collision_rate(l):
    l = list(l)
    output = [l[0]]
    for i in range(100, len(l) - 1):
        output.append(1 - sum(np.array(l[i - 100:i]) == 100) / len(l[i - 100:i]))
    return np.array(output)


X = np.arange(50)
"""alexnet"""
alexnet_0 = np.genfromtxt('../acc_csv/run-alexnet_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
alexnet_1 = np.genfromtxt('../acc_csv/run-alexnet_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
alexnet_2 = np.genfromtxt('../acc_csv/run-alexnet_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
alexnet_3 = np.genfromtxt('../acc_csv/run-alexnet_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
alexnet_4 = np.genfromtxt('../acc_csv/run-alexnet_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']


alexnet = np.vstack((smooth(alexnet_0), smooth(alexnet_1), smooth(alexnet_2), smooth(alexnet_3), smooth(alexnet_4)))

alexnet_mean = np.mean(alexnet, axis=0)
alexnet_std = np.std(alexnet, axis=0)
alexnet_lower_bound = alexnet_mean - alexnet_std
alexnet_upper_bound = alexnet_mean + alexnet_std


"""googlenet"""
googlenet_0 = np.genfromtxt('../acc_csv/run-googlenet_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
googlenet_1 = np.genfromtxt('../acc_csv/run-googlenet_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
googlenet_2 = np.genfromtxt('../acc_csv/run-googlenet_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
googlenet_3 = np.genfromtxt('../acc_csv/run-googlenet_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
googlenet_4 = np.genfromtxt('../acc_csv/run-googlenet_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']


googlenet = np.vstack((smooth(googlenet_0), smooth(googlenet_1), smooth(googlenet_2), smooth(googlenet_3), smooth(googlenet_4)))

googlenet_mean = np.mean(googlenet, axis=0)
googlenet_std = np.std(googlenet, axis=0)
googlenet_lower_bound = googlenet_mean - googlenet_std
googlenet_upper_bound = googlenet_mean + googlenet_std


"""resnet18"""
resnet18_0 = np.genfromtxt('../acc_csv/run-resnet18_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18_1 = np.genfromtxt('../acc_csv/run-resnet18_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18_2 = np.genfromtxt('../acc_csv/run-resnet18_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18_3 = np.genfromtxt('../acc_csv/run-resnet18_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet18_4 = np.genfromtxt('../acc_csv/run-resnet18_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']


resnet18 = np.vstack((smooth(resnet18_0), smooth(resnet18_1), smooth(resnet18_2), smooth(resnet18_3), smooth(resnet18_4)))

resnet18_mean = np.mean(resnet18, axis=0)
resnet18_std = np.std(resnet18, axis=0)
resnet18_lower_bound = resnet18_mean - resnet18_std
resnet18_upper_bound = resnet18_mean + resnet18_std


"""resnet50"""
resnet50_0 = np.genfromtxt('../acc_csv/run-resnet50_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50_1 = np.genfromtxt('../acc_csv/run-resnet50_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50_2 = np.genfromtxt('../acc_csv/run-resnet50_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50_3 = np.genfromtxt('../acc_csv/run-resnet50_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet50_4 = np.genfromtxt('../acc_csv/run-resnet50_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']

resnet50 = np.vstack((smooth(resnet50_0), smooth(resnet50_1), smooth(resnet50_2), smooth(resnet50_3), smooth(resnet50_4)))

resnet50_mean = np.mean(resnet50, axis=0)
resnet50_std = np.std(resnet50, axis=0)
resnet50_lower_bound = resnet50_mean - resnet50_std
resnet50_upper_bound = resnet50_mean + resnet50_std


"""resnet101"""
resnet101_0 = np.genfromtxt('../acc_csv/run-resnet101_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_1 = np.genfromtxt('../acc_csv/run-resnet101_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_2 = np.genfromtxt('../acc_csv/run-resnet101_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_3 = np.genfromtxt('../acc_csv/run-resnet101_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
resnet101_4 = np.genfromtxt('../acc_csv/run-resnet101_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']

resnet101 = np.vstack((smooth(resnet101_0), smooth(resnet101_1), smooth(resnet101_2), smooth(resnet101_3), smooth(resnet101_4)))

resnet101_mean = np.mean(resnet101, axis=0)
resnet101_std = np.std(resnet101, axis=0)
resnet101_lower_bound = resnet101_mean - resnet101_std
resnet101_upper_bound = resnet101_mean + resnet101_std


"""squeezenet"""
squeezenet_0 = np.genfromtxt('../acc_csv/run-squeezenet_0-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
squeezenet_1 = np.genfromtxt('../acc_csv/run-squeezenet_1-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
squeezenet_2 = np.genfromtxt('../acc_csv/run-squeezenet_2-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
squeezenet_3 = np.genfromtxt('../acc_csv/run-squeezenet_3-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']
squeezenet_4 = np.genfromtxt('../acc_csv/run-squeezenet_4-tag-Train_Accuracy.csv', dtype=None, delimiter=',', names=True)['Value']

squeezenet = np.vstack((smooth(squeezenet_0), smooth(squeezenet_1), smooth(squeezenet_2), smooth(squeezenet_3), smooth(squeezenet_4)))

squeezenet_mean = np.mean(squeezenet, axis=0)
squeezenet_std = np.std(squeezenet, axis=0)
squeezenet_lower_bound = squeezenet_mean - squeezenet_std
squeezenet_upper_bound = squeezenet_mean + squeezenet_std

######################################
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(X, alexnet_mean, lw=line_size_others, label='AlexNet', color=color_cycle[colors[0]])
ax.fill_between(X, alexnet_lower_bound, alexnet_upper_bound, facecolor=color_cycle[colors[0]],
                 edgecolor='none', alpha=alpha)

ax.plot(X, googlenet_mean, lw=line_size_others, label='GoogleNet', color=color_cycle[colors[1]])
ax.fill_between(X, googlenet_lower_bound, googlenet_upper_bound, facecolor=color_cycle[colors[1]],
                 edgecolor='none', alpha=alpha)

ax.plot(X, squeezenet_mean, lw=line_size_others, label='SqueezeNet', color=color_cycle[colors[2]])
ax.fill_between(X, squeezenet_lower_bound, squeezenet_upper_bound, facecolor=color_cycle[colors[2]],
                 edgecolor='none', alpha=alpha)

ax.plot(X, resnet18_mean, lw=line_size_others, label='ResNet18', color=color_cycle[colors[3]])
ax.fill_between(X, resnet18_lower_bound, resnet18_upper_bound, facecolor=color_cycle[colors[3]],
                 edgecolor='none', alpha=alpha)

ax.plot(X, resnet50_mean, lw=line_size_others, label='ResNet50', color=color_cycle[colors[4]])
ax.fill_between(X, resnet50_lower_bound, resnet50_upper_bound, facecolor=color_cycle[colors[4]],
                 edgecolor='none', alpha=alpha)

ax.plot(X, resnet101_mean, lw=line_size_ours, label='ResNet101',  color=color_cycle[colors[-1]])
ax.fill_between(X, resnet101_lower_bound, resnet101_upper_bound, facecolor=color_cycle[colors[-1]],
                 edgecolor='none', alpha=alpha)
leg2 = ax.legend(fontsize=legend_size, loc='lower right', ncol=2)

ax.set_xlim(1, 50)
ax.set_ylim(0.6, 1)
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
ax.set_xlabel('Training epochs', fontsize=label_size)
ax.set_ylabel('F1-score', fontsize=label_size)
ax.ticklabel_format(axis="x")

ax.grid()
# set the linewidth of each legend object
for legobj in leg2.legendHandles:
    legobj.set_linewidth(2.0)

plt.tight_layout()
plt.show()
