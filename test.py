import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import time

since = time.time()
# Paths for image directory and model
IMDIR = './test'
# Load a pretrained model - resnet18, resnet50, resnet101, alexnet, squeezenet, vgg11, vgg16, densenet121, densenet161, inception,
# googlenet
name = 'resnet18'
MODEL = './models/' + name + ".pth"

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Retreive 9 random images from directory
files = Path(IMDIR).resolve().glob('*.*')
images = list(files)

# Configure plots
rows, cols = 3, 3

# Preprocessing transformations
preprocess = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Perform prediction and plot results
with torch.no_grad():
    for num, img in enumerate(images):
        img_name = str(img).split('/')[-1]
        img = Image.open(img).convert('RGB')
        inputs = preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        # _, preds = torch.max(outputs, 1)
        # label = class_names[preds]
        # print(img_name, preds.item() + 1)

time_elapsed = time.time() - since
print('Testing time per image in {}s'.format(
        time_elapsed / len(images)))