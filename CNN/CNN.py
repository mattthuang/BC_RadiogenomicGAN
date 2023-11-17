<<<<<<< HEAD
import os
from collections import OrderedDict

import torch
from xarray.plot.utils import plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchsummary import summary
from helper import *
from tqdm.notebook import trange,tqdm
from ignite.engine import *
from ignite.contrib.metrics import *
import copy
import seaborn as sns; sns.set(style='whitegrid')
from sklearn.metrics import precision_recall_curve, auc
from decimal import *
import argparse



parser = argparse.ArgumentParser(description='CNN Training and Inference')
parser.add_argument('--train-data', type=str, help='Path to the training data CSV file')
parser.add_argument('--test-data', type=str, help='Path to the testing data CSV file')
parser.add_argument('--train_image', type=str, help = 'Path to the directory of training MRIs')
parser.add_argument('--test_image', type=str, help = 'Path to the directory of testing MRIs')
parser.add_argument('--gene', type=str, help='Name of the gene for analysis')
parser.add_argument('--epochs', type=int, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--weight-path', type=str, default='weights.pt', help='Path to save the trained model weights')
args = parser.parse_args()


"""
load real MRI csv
"""
# train_data = pd.read_csv("./mutation/real_CNN.csv")
# test_data = pd.read_csv("./mutation/real_CNN_test.csv")

"""
load real CDH1 MRI csv
"""

# train_data = pd.read_csv("./mutation/CDH1_train.csv")
# test_data = pd.read_csv("./mutation/CDH1_test.csv")

"""
load fake; 20 percent
"""

# train_data = pd.read_csv("./mutation/MutationStatus_Train.csv")
# test_data = pd.read_csv("./mutation/MutationStatus_Test.csv")

"""
load fake; 10 percent
"""

# train_data = pd.read_csv("./mutation/training_10percent.csv")
# test_data = pd.read_csv("./mutation/testing_10percent.csv")

"""
load fake; 30 percent
"""

#
# train_data = pd.read_csv("./mutation/training_30percent.csv")
# test_data = pd.read_csv("./mutation/testing_30percent.csv")

"""
load real + fake; 20 percent
"""

# train_data = pd.read_csv("./mutation/MutationStatus_Train_combined.csv")
# test_data = pd.read_csv("./mutation/MutationStatus_Test_combine.csv")

train_data = pd.read_csv(args.train_data)
test_data = pd.read_csv(args.test_data)


'''
load in data -> non stratified 
'''

class ImageDataset(Dataset):
    def __init__(self, data, transform=None, img_dir = None, has_label=True):
        self.data = data
        self.transform = transform
        self.img_dir=img_dir
        self.has_label = has_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = os.path.join(self.img_dir, self.data.iloc[idx]['ID']) + '.npy'
        label = self.data.iloc[idx][args.gene]
        image = np.load(filename)
        # print(image.shape)
        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        return image, label

transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])

'''
Real patient MRIs ( 20 percent testing set) 
'''
# train_dataset = ImageDataset(train_data, transform=transform, img_dir='./training_image')
# test_dataset = ImageDataset(test_data, transform=transform, img_dir='./testing_image')

'''
cGAN generated MRI (10 percent testing set) 
'''
# train_dataset = ImageDataset(train_data, transform=transform, img_dir='./training_image_10percent')
# test_dataset = ImageDataset(test_data, transform=transform, img_dir='./testing_image_10percent')

'''
cGAN generated MRIs (20 percent testing set) 
'''
# train_dataset = ImageDataset(train_data, transform=transform, img_dir='./training_image_cgan')
# test_dataset = ImageDataset(test_data, transform=transform, img_dir='./testing_image_cgan')

'''
cGAN generated MRI (30 percent testing set) 
'''

# train_dataset = ImageDataset(train_data, transform=transform, img_dir='./training_image_30percent')
# test_dataset = ImageDataset(test_data, transform=transform, img_dir='./testing_image_30percent')

'''
Real patient MRI + cGAN generated MRI (20 percent testing set) 
'''

# train_dataset = ImageDataset(train_data, transform=transform, img_dir='./training_image_combine')
# test_dataset = ImageDataset(test_data, transform=transform, img_dir='./testing_image_combine')

'''
CDH1 real MRI
'''
# train_dataset = ImageDataset(train_data, transform=transform, img_dir='./training_image_CDH1')
# test_dataset = ImageDataset(test_data, transform=transform, img_dir='./testing_image_CDH1')

train_dataset = ImageDataset(train_data, transform=transform, img_dir=args.train_image)
test_dataset = ImageDataset(test_data, transform=transform, img_dir=args.test_image)

print(len(test_dataset))
print(len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


'''
CNN MODEL 

'''
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 4 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        print(x.shape)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 4 * 16)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x


cnn = CNN3D()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn.to(device)

summary(cnn,input_size=(1,128,32,128),device=device.type)

loss = nn.BCELoss()
optimiser = optim.Adam(cnn.parameters(), lr = 3e-5)
lr_scheduler = ReduceLROnPlateau(optimiser, mode='min',factor=0.5, patience=20,verbose=1)


def train_val(model, params, verbose=True):

    #retrieve parameters

    epochs = params["epochs"]
    loss_function = params["f_loss"]
    opt = params["optimiser"]
    print(type(opt))
    train_data = params["train"]
    test_data = params["test"]
    lr_scheduler = params["lr_change"]
    weight_path = params ["weight_path"]

    # history of loss values in each epoch
    loss_history = {"train":[], "test": []}


    #deep copy of weigts for best performign model
    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float("inf")

    '''TRAIN MODEL FOR __ EPOCHS '''

    for epoch in tqdm(range(epochs)):

        current_lr = get_lr(opt)
        if(verbose):
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))

        '''TRAIN PROCESS'''

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_function, train_data, opt)
        loss_history["train"].append(train_loss)


        ''' TEST PROCESS '''

        model.eval()
        with torch.no_grad():

            test_loss, test_metric = loss_epoch(model, loss_function, test_data)

            if(test_loss < best_loss):
                best_loss = test_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), weight_path)
                if (verbose):
                    print("Copied best model weights")

            loss_history["test"].append(test_loss)

            lr_scheduler.step(test_loss)
            if current_lr != get_lr(opt):
                if(verbose):
                    print("Loading best model weights")
                model.load_stat_dict(best_model_wts)

            if(verbose):
                print(f"train loss: {train_loss:.6f}, dev loss: {test_loss:.6f}, accuracy: {100 * test_metric:.2f}")
                print("-" * 10)

    model.load_state_dict(best_model_wts)

    return model, loss_history

params_train={
 "train": train_loader,
 "test": test_loader,
 "epochs": args.epochs,
 "optimiser": optim.Adam(cnn.parameters(), lr=3e-5),
 "lr_change": ReduceLROnPlateau(optimiser,
                                mode='min',
                                factor=0.5,
                                patience=20,
                                verbose=True),
 "f_loss": nn.BCELoss(reduction="sum"),
 "weight_path": args.weight_path,
 "check": False,
}

cnn,loss_hist = train_val(cnn, params_train,verbose=True)
epochs= params_train["epochs"]

sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["train"],label='Training_Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig("Loss Curve")
plt.clf()


''' inference '''
cnn.load_state_dict(torch.load('weights.pt'))


def inference(model, dataset, device, num_classes=1):
    len_data = len(dataset)
    y_out = torch.zeros(len_data, num_classes)  # initialize output tensor on CPU
    y_gt = np.zeros((len_data,1), dtype="uint8")  # initialize ground truth on CPU
    model = model.to(device)  # move model to device

    with torch.no_grad():
        for i in tqdm(range(len_data)):
            x, y = dataset[i]
            y_gt[i][0] = y
            y_out[i] = model(x.unsqueeze(0).to(device))

    return y_out.numpy(), y_gt

# inference_outputs = inference(cnn, test_dataset, device)
predictions, true = inference(cnn, test_dataset, device)


''' AUC_ROC IMPLEMENTATION'''


def eval_step (engine, batch):

    return batch

evaluator = Engine(eval_step)

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

# manual_seed(666)

roc_auc = ROC_AUC()

roc_auc.attach(evaluator, 'roc_auc')

print(predictions)
print(true)
y_pred = torch.tensor(predictions)
y_true = torch.tensor(true)

state = evaluator.run([[y_pred, y_true]])
print(state.metrics['roc_auc'])


def plot_roc_curve(y_true, y_score):
    # Sort scores and labels by descending score
    scores, labels = zip(*sorted(zip(y_score, y_true), reverse=True))
    num_positives = sum(labels)
    num_negatives = len(labels) - num_positives

    # Calculate true positive rate (TPR) and false positive rate (FPR) at each threshold
    tpr_list = []
    fpr_list = []
    for i in range(len(scores)):
        tp = sum(labels[:i])
        fp = i - tp
        tpr_list.append(tp / num_positives)
        fpr_list.append(fp / num_negatives)

    # Plot ROC curve
    getcontext().prec = 1

    AUC = "{:.4f}".format(state.metrics['roc_auc'])

    plt.plot(fpr_list, tpr_list, label= 'ROC Curve' + " " + "AUC:" + AUC)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("ROC Curve")
    plt.clf()

plot_roc_curve(y_true, y_pred)


''' Precision AUC '''

from sklearn.metrics import precision_recall_curve, auc

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
auprc = auc(recall, precision)
print(f"AUPRC: {auprc:.4f}")

# Plot the precision-recall curve
import matplotlib.pyplot as plt

plt.plot(recall, precision, label = f"AUC: {auprc:.4f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim(0.0, 1.05)
plt.title(f'Precision-Recall Curve')
plt.legend()
plt.savefig("PR Curve")

