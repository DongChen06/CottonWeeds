"""no random rotation"""
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Eval CottonWeed Classifier')
    parser.add_argument('--train_directory', type=str, required=False,
                        default='/home/dong9/Downloads/DATA_1012',
                        help="training directory")
    parser.add_argument('--model_name', type=str, required=False, default='alexnet',
                        help="choose a deep learning model")
    parser.add_argument('--num_classes', type=int, required=False, default=15, help="Number of Classes")
    parser.add_argument('--seeds', type=int, required=False, default=0,
                        help="random seed")
    parser.add_argument('--batch_size', type=int, required=False, default=12, help="Training batch size")
    parser.add_argument('--img_size', type=int, required=False, default=512, help="Image Size")
    parser.add_argument('--use_weighting', type=bool, required=False, default=False, help="use weighted cross entropy or not")
    args = parser.parse_args()
    return args


args = parse_args()
import torch, os
import random
import numpy as np
from sklearn.model_selection import KFold

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


from torchvision import datasets, transforms
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


num_classes = args.num_classes
bs = args.batch_size
img_size = args.img_size
train_directory = args.train_directory

if not os.path.isfile('eval_cross_val.csv'):
    with open('eval_cross_val.csv', mode='w') as csv_file:
        fieldnames = ['Index', 'Model', 'Best Val Acc']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

# Set the model save path
model_name = args.model_name
if args.use_weighting:
    EVAL_MODEL = './models/' + model_name + '_' + str(args.seeds) + '_w' + ".pth"
else:
    EVAL_MODEL = './models/' + model_name + '_' + str(args.seeds) + ".pth"
# Number of workers
num_cpu = 32  # multiprocessing.cpu_count()

print(model_name)
# Applying transforms to the data
image_transforms = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.CenterCrop(size=img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

# Load data from folders
dataset = datasets.ImageFolder(root=train_directory, transform=image_transforms)

# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Size of train and validation data
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    if fold != 4:
        continue
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    # Create iterators for data loading
    eval_loader = data.DataLoader(dataset, batch_size=bs, num_workers=num_cpu, pin_memory=True, drop_last=True,
                                 worker_init_fn=seed_worker, generator=g, sampler=test_subsampler)

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
    print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(len(test_subsampler), overall_accuracy))

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    # print('Confusion Matrix')
    # print('-' * 16)
    # print(conf_mat, '\n')

    if not os.path.exists('Confusing_Matrices_cv'):
        os.mkdir('Confusing_Matrices_cv')
    if not os.path.exists('Confusing_Matrices_cv/plots/'):
        os.mkdir('Confusing_Matrices_cv/plots/')
    if not os.path.exists('Confusing_Matrices_cv/csv/'):
        os.mkdir('Confusing_Matrices_cv/csv/')

    plt.figure(figsize=(10, 6))
    df_cm = pd.DataFrame(conf_mat / conf_mat.sum(1), index=class_names, columns=class_names)
    sn.set(font_scale=1.0)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap='Greens', fmt='0.3f')
    plt.xticks(rotation=75, fontsize=14)
    plt.tight_layout()
    if args.use_weighting:
        plt.savefig('Confusing_Matrices_cv/plots/' + model_name + '_cm_' + str(args.seeds) + '_w.png')
    else:
        plt.savefig('Confusing_Matrices_cv/plots/' + model_name + '_cm_' + str(args.seeds) + '.png')
    # plt.show()

    if args.use_weighting:
        df_cm.to_csv('Confusing_Matrices_cv/csv/' + model_name + '_cm_' + str(args.seeds) + '_w.csv')
    else:
        df_cm.to_csv('Confusing_Matrices_cv/csv/' + model_name + '_cm_' + str(args.seeds) + '.csv')

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    print('Per class accuracy')
    print('-' * 18)
    for label, accuracy in zip(class_names, class_accuracy):
        print('Accuracy of class %8s : %0.2f %%' % (label, accuracy))

    with open('eval_cross_val.csv', 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow([model_name, overall_accuracy])