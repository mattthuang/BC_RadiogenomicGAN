import scipy
import scipy.ndimage as nd
import scipy.io as io
import matplotlib
from params import *
import params

if params.device.type != 'cpu':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
import csv
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
import pathlib


def SavePloat_Voxels(voxels, path, iteration):

    # save voxel data as numpy array
    np.save(path + '/{}.npy'.format(str(iteration).zfill(3)), voxels)
    
    #plot to visualize
    fig = plt.figure(figsize=(64, 32))
    gs = gridspec.GridSpec(4, 8)
    for j in range(32):
            #print(j)
        #print(voxels[j])
        ax = plt.Subplot(fig, gs[j])
        ax.imshow(voxels[j],cmap='gray')      
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)


    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


class ShapeNetDataset(data.Dataset):

    def __init__(self, root, args, train_or_val="train"):
        
        self.root_dir = data_dir
        self.volumn_list = [str(file) for file in pathlib.Path(self.root_dir).iterdir()]
        self.volumn_list.sort()
        data_size = len(self.volumn_list)
        self.args = args

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        volumn_path = self.volumn_list[index]
        volumn= np.load(volumn_path)
        #volumn = scipy.ndimage.zoom(volumn, (32/volumn.shape[0], 32/volumn.shape[1], 32/volumn.shape[2]), order=1)
        volumn = scipy.ndimage.zoom(volumn, (32/volumn.shape[0], 128/volumn.shape[1], 128/volumn.shape[2]), order=1)
        volumn = volumn[np.newaxis,:,:,:]
        volumn = torch.from_numpy(volumn)
        #print('volumn.shape',volumn.shape)

        rows = []
        with open(BTF_dir) as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                rows.append(row)
                #rows.append(row[1:])
        
        BTF = np.array(rows[index][1:]).astype(np.float)
        #BTF = np.array(rows[1:]).astype(np.float)

        return volumn, BTF

    def __len__(self):
        return len(self.volumn_list)


def generateZ(args, batch):
    #generate random Z vector

    if params.z_dis == "norm":
        Z = torch.Tensor(batch, params.z_dim).normal_(0, 0.33).to(params.device)
    elif params.z_dis == "uni":
        Z = torch.randn(batch, params.z_dim).to(params.device).to(params.device)
    else:
        print("z_dist is not normal or uniform")

    return Z

