import argparse
import os, csv
from pathlib import Path
from PIL import Image
from numpy import savetxt
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='Test CottonWeed Classifier')
    parser.add_argument('--model_name', type=str, required=False, default='mobilenet_v2',  # vgg11
                        help="choose a deep learning model")
    parser.add_argument('--EVAL_DIR', type=str, required=False,
                        default='/home/orange/Downloads/CottonWeedDataset/similarity_dataset/',
                        help="dir for the testing image")
    parser.add_argument('--seeds', type=int, required=False, default=0,
                        help="random seed")
    parser.add_argument('--img_size', type=int, required=False, default=512, help="Image Size")
    args = parser.parse_args()
    return args


def cosine_distance(input1, input2):
    '''Calculating the distance of two inputs.

    The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
    `1` denotes they are the most similar.

    Args:
        input1, input2: two input numpy arrays.

    Returns:
        Element-wise cosine distances of two inputs.
    '''
    # return np.dot(input1, input2) / (np.linalg.norm(input1) * np.linalg.norm(input2))
    return np.dot(input1, input2.T) / \
           np.dot(np.linalg.norm(input1, axis=1, keepdims=True), \
                  np.linalg.norm(input2.T, axis=0, keepdims=True))


args = parse_args()

import numpy as np
import torch
import random
from torchvision import datasets, transforms


# for reproducing
torch.manual_seed(args.seeds)
torch.cuda.manual_seed(args.seeds)
torch.cuda.manual_seed_all(args.seeds) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seeds)
random.seed(args.seeds)
np.random.seed(args.seeds)


IMDIR = args.EVAL_DIR
model_name = args.model_name
EVAL_MODEL = '../../models/' + model_name + '_' + str(args.seeds) + ".pth"
img_size = args.img_size

# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()


# Prepare the eval data loader
# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.CenterCrop(size=img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

# Class label names
class_names = ['Carpetweeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory', 'Nutsedge',
               'PalmerAmaranth', 'PricklySida', 'Purslane', 'Ragweed', 'Sicklepod',
                'SpottedSpurge', 'SpurredAnoda', 'Swinecress', 'Waterhemp']


for class_name in class_names:
    files = Path(IMDIR + class_name).resolve().glob('*.*')
    images = list(files)

    # Enable gpu mode, if cuda available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Evaluate the model accuracy on the dataset
    correct = 0
    total = 0
    image_list = []

    for img in images:
        image = Image.open(img).convert('RGB')
        inputs = preprocess(image)
        image_list.append(inputs)

    image_list = torch.stack(image_list).to(device)
    with torch.no_grad():
        # vgg11
        # x = model.features(image_list)
        # x = model.avgpool(x)
        # x = torch.flatten(x, 1)
        # features = model.classifier[0](x)

        # mobilenet-v2
        x = model.features(image_list)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)

    np.save(class_name, features.cpu().numpy())

similarity_matrix = np.zeros((len(class_names), len(class_names)))
for i, ci in enumerate(class_names):
    for j, cj in enumerate(class_names):
        c1 = np.load(ci + '.npy')
        c2 = np.load(cj + '.npy')
        similarity_matrix[i][j] = np.mean(np.diagonal(cosine_distance(c1, c2)))

# print(similarity_matrix)
savetxt('similarity_matrix.csv', similarity_matrix, delimiter=',')