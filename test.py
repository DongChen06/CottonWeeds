import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import time
import argparse
import os, csv


def parse_args():
    parser = argparse.ArgumentParser(description='Test CottonWeed Classifier')
    # Load a pretrained model - resnet18, resnet50, resnet101, alexnet, squeezenet, vgg11, vgg16, vgg19,
    # densenet121, densenet169,  densenet161, inception, inceptionv4, googlenet, xception, mobilenet_v2,
    # mobilenet_v3_small, mobilenet_v3_large, inceptionresnetv2, dpn68, mnasnet1_0, efficientnet-b0
    # efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5
    parser.add_argument('--model_name', type=str, required=False, default='alexnet',
                        help="choose a deep learning model")
    parser.add_argument('--IMDIR', type=str, required=False, default='./test',
                        help="dir for the testing image")
    parser.add_argument('--IMDIR', type=str, required=False, default='./test',
                        help="dir for the testing image")
    parser.add_argument('--img_size', type=int, required=False, default=512, help="Image Size")
    args = parser.parse_args()
    return args


args = parse_args()


# for reproducing
torch.manual_seed(args.seeds)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

IMDIR = args.IMDIR
model_name = args.model_name
img_size = args.img_size
since = time.time()
MODEL = './models/' + model_name + ".pth"

if not os.path.isfile('test_performance.csv'):
    with open('test_performance.csv', mode='w') as csv_file:
        fieldnames = ['Model', 'Testing Time (s)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

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
        # _, preds = torch.max(outputs, 1)
        # label = class_names[preds]
        # print(img_name, preds.item() + 1)

time_elapsed = time.time() - since
print('Testing time per image in {}s'.format(
    time_elapsed / len(images)))

with open('test_performance.csv', 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([model_name, '{}'.format(time_elapsed / len(images))])
