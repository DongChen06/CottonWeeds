import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

size = [57.1, 0.743, 5.6, 20.8, 11.8,
        3.1, 11.2, 23.5, 42.5, 128.8, 134.3,
        139.6, 7.0, 26.5, 12.5, 24.4,
        41.2, 54.3, 2.2, 1.5, 4.2,
        4.0, 6.5, 7.7, 10.7, 17.6, 28.4]
time = [37, 46, 52, 89, 79,
        51, 47, 73, 92, 67,
        99, 112, 75, 133, 85,
        73, 120, 124, 53, 41,
        49, 63, 77, 78, 92,
        113, 144]
model = ['AlexNet', 'SqueezeNet', 'GoogleNet', 'Xception', 'DPN68',
         'MnasNet', 'ResNet18', 'ResNet50', 'ResNet101', 'VGG11',
         'VGG16', 'VGG19', 'Densenet121', 'Densenet161', 'Densenet169',
         'Inceptionv3', 'Inceptionv4', 'Inception-ResNetv2', 'MobilenetV2',
         'MobilenetV3-small', 'MobilenetV3-large', 'EfficientNet-b0',
         'EfficientNet-b1', 'EfficientNet-b2', 'EfficientNet-b3', 'EfficientNet-b4',
         'EfficientNet-b5']

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(size, time, marker='^')

ax.set_xlabel('Model size (million)', fontsize=12)
ax.set_ylabel('Training time', fontsize=12)

for i, txt in enumerate(model):
    ax.annotate(txt, (size[i], time[i]))

plt.tight_layout()
plt.savefig('size_time.pdf')
plt.show()
