import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os, csv

# for reproducing
torch.manual_seed(66)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='Test CottonWeed Classifier')
    # Load a pretrained model - resnet18, resnet50, resnet101, alexnet, squeezenet, vgg11, vgg16, vgg19,
    # densenet121, densenet169,  densenet161, inception, inceptionv4, googlenet, xception, mobilenet_v2,
    # mobilenet_v3_small, mobilenet_v3_large, inceptionresnetv2, dpn68, mnasnet1_0, efficientnet-b0
    # efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5
    parser.add_argument('--model_name', type=str, required=False, default='alexnet',
                        help="choose a deep learning model")
    parser.add_argument('--EVAL_DIR', type=str, required=False,
                        default='/home/dong9/Downloads/DATA_0820/CottonWeedDataset/test',
                        help="dir for the testing image")
    parser.add_argument('--batch_size', type=int, required=False, default=8, help="Training batch size")
    parser.add_argument('--img_size', type=int, required=False, default=512, help="Image Size")
    args = parser.parse_args()
    return args


args = parse_args()
EVAL_DIR = args.EVAL_DIR
model_name = args.model_name
EVAL_MODEL = './models/' + model_name + ".pth"
img_size = args.img_size
bs = args.batch_size

if not os.path.isfile('eval_performance.csv'):
    with open('eval_performance.csv', mode='w') as csv_file:
        fieldnames = ['Model', 'Evaluating Acc']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and number of cpu's
num_cpu = multiprocessing.cpu_count()
# Prepare the eval data loader
eval_transform = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.CenterCrop(size=img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

eval_dataset = datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader = data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                              num_workers=num_cpu, pin_memory=True)

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
num_classes = len(eval_dataset.classes)
dsize = len(eval_dataset)

# Class label names
class_names = ['Carpetweeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory', 'Nutsedge',
               'PalmerAmaranth', 'PricklySida', 'Purslane', 'Ragweed', 'Sicklepod',
                'SpottedSpurge', 'SpurredAnoda', 'Swinecress', 'Waterhemp']

# Initialize the prediction and label lists
predlist = torch.zeros(0, dtype=torch.long, device='cpu')
lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

# Evaluate the model accuracy on the dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predlist = torch.cat([predlist, predicted.view(-1).cpu()])
        lbllist = torch.cat([lbllist, labels.view(-1).cpu()])

# Overall accuracy
overall_accuracy = 100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize, overall_accuracy))

# Confusion matrix
conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-' * 16)
print(conf_mat, '\n')

plt.figure(figsize=(10, 7))
df_cm = pd.DataFrame(conf_mat, index=class_names,
                     columns=class_names)
sn.set(font_scale=1.2)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.tight_layout()
plt.xticks(rotation=60, fontsize=16)
plt.savefig('Confusing_Matrices/' + model_name + '_cm.png')
# plt.show()

# Per-class accuracy
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
print('Per class accuracy')
print('-' * 18)
for label, accuracy in zip(eval_dataset.classes, class_accuracy):
    print('Accuracy of class %8s : %0.2f %%' % (label, accuracy))

with open('eval_performance.csv', 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([model_name, overall_accuracy])
