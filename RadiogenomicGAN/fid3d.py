import os
import torch
import numpy as np
from pytorch_fid import fid_score
from Med3D.setting import parse_opts
from Med3D.model import generate_model
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_med3d_backbone():

    # establish settings

    sets = parse_opts()
    sets.resume_path = '../Med3D/pretrain/resnet_10_23dataset.pth'
    sets.target_type = "normal"
    sets.phase = 'test'
    sets.no_cuda = True
    sets.model_depth = 10
    sets.resnet_shortcut = 'B'

    # retrieving pre-trained model

    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net_dict = net.state_dict()

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = checkpoint['state_dict']

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    net_dict.update(new_state_dict)
    net.load_state_dict(net_dict)

    net_backbone = nn.Sequential(*[list(net.children())[i] for i in range(8)])

    return net_backbone


def normalise_image_med3d(in_batch):
    batch_size = in_batch.shape[0]
    channel_size = in_batch.shape[1]
    z = in_batch.shape[2]
    y = in_batch.shape[3]
    x = in_batch.shape[4]
    in_batch_flat = in_batch.reshape(batch_size, -1)
    mean = torch.mean(in_batch_flat, dim=1)
    std = torch.std(in_batch_flat, dim=1)

    mean = mean[:, None, None, None, None].repeat(1, 1, z, x, y)
    std = std[:, None, None, None, None].repeat(1, 1, z, x, y)

    #transforms mean and std to have batch_size, channel_size and z,x,y dimensions

    out = (in_batch - mean) / std

    return out

def stats_from_fake_folder_3d(fake_path, device):
    files = os.listdir(fake_path)
    med3d_net = load_med3d_backbone().to(device)
    act_arr = []
    with torch.no_grad():

        for ii, file in enumerate(files):
            curr_arr = np.load(fake_path + file)
            torch_arr = torch.from_numpy(curr_arr).float()

            # import numpy array as torch object
            x = torch.as_tensor(torch_arr).to(device)

            # reshape array

            x = x.reshape(1,1,x.shape[0],x.shape[1],x.shape[2])
            tmp_act = med3d_net(normalise_image_med3d(x))


            channels_to_use = tmp_act.shape[1]
            act_arr.append(tmp_act[:, 0:channels_to_use].mean(dim=(2, 3, 4)))


    act_arr = torch.cat(act_arr)
    num_images = act_arr.shape[0]
    act_arr = act_arr.reshape(num_images, -1)

    mu = np.mean(act_arr.cpu().numpy(), axis=0)

    sigma = np.cov(act_arr.cpu().numpy().astype('float16'), rowvar=False)
    std_dev = np.std(act_arr.cpu().numpy().astype('float16'), axis=0)

    return mu, sigma, std_dev



def stats_from_real_folder_3d(real_path, device):
    files = os.listdir(real_path)
    med3d_net = load_med3d_backbone().to(device)
    act_arr = []
    with torch.no_grad():

        for ii, file in enumerate(files):
            curr_arr = np.load(real_path + file)
            torch_arr = torch.from_numpy(curr_arr).float()

            x = torch.as_tensor(torch_arr).to(device)
            x = x.reshape(1, 1, x.shape[0], x.shape[1], x.shape[2])
            tmp_act = med3d_net(normalise_image_med3d(x))


            channels_to_use = tmp_act.shape[1]
            act_arr.append(tmp_act[:, 0:channels_to_use].mean(dim=(2, 3, 4)))

    act_arr = torch.cat(act_arr)
    num_images = act_arr.shape[0]
    act_arr = act_arr.reshape(num_images, -1)

    mu = np.mean(act_arr.cpu().numpy(), axis=0)

    sigma = np.cov(act_arr.cpu().numpy().astype('float16'), rowvar=False)

    std_dev = np.std(act_arr.cpu().numpy().astype('float16'), axis=0)

    return mu, sigma, std_dev


def fid3d(real_folder, fake_path, device):

    mu2, sigma2, std_dev_fake = stats_from_fake_folder_3d(fake_path, device)
    print("fake", mu2.shape, sigma2.shape, std_dev_fake.shape)


    mu1, sigma1, std_dev_real = stats_from_real_folder_3d(real_folder, device)
    print("real", mu1.shape, sigma1.shape, std_dev_real.shape)

    fid_val = fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    fid_sd = np.sqrt(np.sum((std_dev_real - std_dev_fake)**2))

    return fid_val, fid_sd


# script

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--real', default='', type=str, help="Path to real images")
parser.add_argument('--fake', default='', type=str, help="Path to fake images")
parser.add_argument('--cudaDevice', default='', help="cuda device number to use")
opt = parser.parse_known_args()[0]
args = parser.parse_args()

if opt.cudaDevice == '':
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + opt.cudaDevice)



real_path = args.real
fake_path = args.fake

fid_val, fid_sd = fid3d(real_path, fake_path, device)
print('#########################')
print("FID:",fid_val, " std:", fid_sd)
print('#########################')

