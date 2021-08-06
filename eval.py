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

# Paths for image directory and model
EVAL_DIR = 'DATASET/test'
# Load a pretrained model - resnet18, resnet50, resnet101, alexnet, squeezenet, vgg11, vgg16, densenet121, densenet161, inception
# googlenet
name = 'densenet161'
EVAL_MODEL = './models/' + name + ".pth"
img_size = 512
# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and number of cpu's
num_cpu = multiprocessing.cpu_count()
bs = 3

# Prepare the eval data loader
eval_transform=transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True)

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
num_classes = len(eval_dataset.classes)
dsize = len(eval_dataset)

# Class label names
class_names = ['Carpetweed', 'Crabgrass', 'Goosegrass', 'Morningglory', 'Nutsedge', 'PalmerAmaranth',
               'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress']

# Initialize the prediction and label lists
predlist = torch.zeros(0,dtype=torch.long, device='cpu')
lbllist = torch.zeros(0,dtype=torch.long, device='cpu')

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

        predlist = torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist = torch.cat([lbllist,labels.view(-1).cpu()])

# Overall accuracy
overall_accuracy = 100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize, 
    overall_accuracy))

# Confusion matrix
conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-'*16)
print(conf_mat, '\n')


plt.figure(figsize=(10, 7))
df_cm = pd.DataFrame(conf_mat, index=class_names,
                     columns=class_names)
sn.set(font_scale=1.2)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.tight_layout()
plt.savefig('Confusing_Matrices/' + name + '_cm.png')
plt.xticks(rotation=60, fontsize=16)
plt.show()


# Per-class accuracy
class_accuracy = 100*conf_mat.diagonal()/conf_mat.sum(1)
print('Per class accuracy')
print('-'*18)
for label, accuracy in zip(eval_dataset.classes, class_accuracy):
     print('Accuracy of class %8s : %0.2f %%'%(label, accuracy))

'''
Sample run: python eval.py eval_ds
'''
