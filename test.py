import argparse
import os
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='Test CottonWeed Classifier')
    parser.add_argument('--model_name', type=str, required=False, default='alexnet',
                        help="choose a deep learning model")
    parser.add_argument('--IMDIR', type=str, required=False, default='./test',
                        help="dir for the testing image")
    parser.add_argument('--seeds', type=int, required=False, default=0,
                        help="dir for the testing image")
    parser.add_argument('--img_size', type=int, required=False, default=224, help="Image Size")
    args = parser.parse_args()
    return args


args = parse_args()
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import time
import random
import numpy as np

# for reproducing
torch.manual_seed(args.seeds)
torch.cuda.manual_seed(args.seeds)
torch.cuda.manual_seed_all(args.seeds)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seeds)
random.seed(args.seeds)
np.random.seed(args.seeds)

IMDIR = args.IMDIR
model_name = args.model_name
img_size = args.img_size
since = time.time()
MODEL = './models/' + model_name + '_' + str(args.seeds) + ".pth"

if not os.path.isfile('test_performance512.csv'):
    with open('test_performance512.csv', mode='w') as csv_file:
        fieldnames = ['Index', 'Model', 'Testing Time (s)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Retreive 9 random images from directory
files = Path(IMDIR).resolve().glob('*.*')
images = list(files)

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.CenterCrop(size=img_size),
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

time_elapsed = time.time() - since
# print('Testing time per image in {}s'.format(
#     time_elapsed / len(images)))

with open('test_performance512.csv', 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([args.seeds, model_name, '{}'.format(time_elapsed / len(images))])
