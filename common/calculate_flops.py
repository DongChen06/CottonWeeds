import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
"""no random rotation"""
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train CottonWeed Classifier')
    parser.add_argument('--train_directory', type=str, required=False,
                        default='/home/dong9/Downloads/DATA_0820/CottonWeedDataset/train',
                        help="training directory")
    parser.add_argument('--valid_directory', type=str, required=False,
                        default='/home/dong9/Downloads/DATA_0820/CottonWeedDataset/val',
                        help="validation directory")
    parser.add_argument('--model_name', type=str, required=False, default='mobilenet_v2',
                        help="choose a deep learning model")
    parser.add_argument('--train_mode', type=str, required=False, default='finetune',
                        help="Set training mode: finetune, transfer, scratch")
    parser.add_argument('--num_classes', type=int, required=False, default=15, help="Number of Classes")
    parser.add_argument('--seeds', type=int, required=False, default=0,
                        help="random seed")
    parser.add_argument('--epochs', type=int, required=False, default=50, help="Training Epochs")
    parser.add_argument('--batch_size', type=int, required=False, default=12, help="Training batch size")
    parser.add_argument('--img_size', type=int, required=False, default=512, help="Image Size")
    parser.add_argument('--use_weighting', type=bool, required=False, default=False, help="use weighted cross entropy or not")
    args = parser.parse_args()
    return args


args = parse_args()
import torch, os
import random
import numpy as np

# for reproducing
torch.manual_seed(args.seeds)
torch.cuda.manual_seed(args.seeds)
torch.cuda.manual_seed_all(args.seeds) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seeds)
random.seed(args.seeds)
np.random.seed(args.seeds)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(args.seeds)


from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchsummary import summary
import time, copy
import multiprocessing
import pretrainedmodels  # for inception-v4 and xception
from efficientnet_pytorch import EfficientNet


num_classes = args.num_classes
model_name = args.model_name
train_mode = args.train_mode
num_epochs = args.epochs
bs = args.batch_size
img_size = args.img_size
train_directory = args.train_directory
valid_directory = args.valid_directory

if not os.path.isfile('flops.csv'):
    with open('flops.csv', mode='w') as csv_file:
        fieldnames = ['Index', 'Model', 'Trainable Parameters', 'MACs']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

# Set the model save path
if args.use_weighting:
    print(True)
    PATH = model_name + "_" + str(args.seeds) + "_w" + ".pth"
else:
    PATH = model_name + "_" + str(args.seeds) + ".pth"
# Number of workers
num_cpu = 32  # multiprocessing.cpu_count()

# Applying transforms to the data
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

# Size of train and validation data
dataset_sizes = {
    'train': len(dataset['train']),
    'valid': len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                             num_workers=num_cpu, pin_memory=True, drop_last=True,
                             worker_init_fn=seed_worker, generator = g),
    'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                             num_workers=num_cpu, pin_memory=True, drop_last=True,
                             worker_init_fn=seed_worker, generator=g)}

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)

# Print the train and validation data sizes
print("Training-set size:", dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

print("\nLoading pretrained-model for finetuning ...\n")
model_ft = None


if model_name == 'resnet18':
    # Modify fc layers to match num_classes
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'resnet50':
    # Modify fc layers to match num_classes
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'resnet101':
    # Modify fc layers to match num_classes
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'alexnet':
    model_ft = models.alexnet(pretrained=True)
    model_ft.classifier[6] = nn.Linear(4096, num_classes)
elif model_name == 'vgg11':
    model_ft = models.vgg11(pretrained=True)
    model_ft.classifier[6] = nn.Linear(4096, num_classes)
elif model_name == 'vgg16':
    model_ft = models.vgg16(pretrained=True)
    model_ft.classifier[6] = nn.Linear(4096, num_classes)
elif model_name == 'vgg19':
    model_ft = models.vgg19(pretrained=True)
    model_ft.classifier[6] = nn.Linear(4096, num_classes)
elif model_name == 'squeezenet':
    model_ft = models.squeezenet1_0(pretrained=True)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
elif model_name == 'densenet121':
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
elif model_name == 'densenet169':
    model_ft = models.densenet169(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
elif model_name == 'densenet161':
    model_ft = models.densenet161(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
elif model_name == 'inception':
    model_ft = models.inception_v3(pretrained=True)
    model_ft.aux_logits = False
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'inceptionv4':
    model_ft = pretrainedmodels.inceptionv4(pretrained='imagenet')
    num_ftrs = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
elif model_name == 'googlenet':
    model_ft = models.googlenet(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'xception':
    model_ft = pretrainedmodels.xception(pretrained='imagenet')
    num_ftrs = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
elif model_name == 'mobilenet_v2':
    model_ft = models.mobilenet_v2(pretrained=True)
    model_ft.classifier[1] = nn.Linear(model_ft.last_channel, num_classes)
elif model_name == 'mobilenet_v3_small':
    model_ft = models.mobilenet_v3_small(pretrained=True)
    model_ft.classifier[3] = nn.Linear(model_ft.classifier[3].in_features, num_classes)
elif model_name == 'mobilenet_v3_large':
    model_ft = models.mobilenet_v3_large(pretrained=True)
    model_ft.classifier[3] = nn.Linear(model_ft.classifier[3].in_features, num_classes)
elif model_name == 'shufflenet_v2_x0_5':
    model_ft = models.shufflenet_v2_x0_5(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'shufflenet_v2_x1_0':
    model_ft = models.shufflenet_v2_x1_0(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'inceptionresnetv2':
    model_ft = pretrainedmodels.inceptionresnetv2(pretrained='imagenet')
    num_ftrs = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
elif model_name == 'nasnetamobile':
    model_ft = pretrainedmodels.nasnetamobile(num_classes=1000, pretrained='imagenet')
    num_ftrs = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
elif model_name == 'dpn68':
    model_ft = pretrainedmodels.dpn68(pretrained='imagenet')
    model_ft.last_linear = nn.Conv2d(832, num_classes, kernel_size=(1, 1), stride=(1, 1))
elif model_name == 'polynet':
    model_ft = pretrainedmodels.polynet(num_classes=1000, pretrained='imagenet')
    num_ftrs = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
elif model_name == 'mnasnet1_0':
    model_ft = models.mnasnet1_0(pretrained=True)
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
elif model_name == 'efficientnet-b0':
    model_ft = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
elif model_name == 'efficientnet-b1':
    model_ft = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
elif model_name == 'efficientnet-b2':
    model_ft = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
elif model_name == 'efficientnet-b3':
    model_ft = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
elif model_name == 'efficientnet-b4':
    model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
elif model_name == 'efficientnet-b5':
    model_ft = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
elif model_name == 'efficientnet-b6':
    model_ft = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
else:
    print("Invalid model name, exiting...")
    exit()

# Transfer the model to GPU
# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

with torch.cuda.device(0):
  macs, params = get_model_complexity_info(model_ft, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


with open('flops.csv', 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([args.seeds, model_name, params, macs])