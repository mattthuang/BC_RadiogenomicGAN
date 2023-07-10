from utils import *
import os
from model import net_G, net_D
from tqdm import tqdm
import numpy as np
import params
import visdom

def tester(args):
    print('Evaluation Mode...')

    #Define directory to save generated images
    image_saved_path = params.output_dir + '/' + args.model_name + '/' + args.logs + '/test_outputs'
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    if args.use_visdom:
        vis = visdom.Visdom()

    #Define directory which holds the trained model
    save_file_path = params.output_dir + '/' + args.model_name
    pretrained_file_path_G = save_file_path + '/' + args.logs + '/models/G.pth'
    pretrained_file_path_D = save_file_path + '/' + args.logs + '/models/D.pth'

    print(pretrained_file_path_G)

    #Instantiate generator and discriminator models
    D = net_D(args)
    G = net_G(args)

    #Load in trained models
    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))

    print('visualizing model')

    G.to(params.device)
    D.to(params.device)
    G.eval()
    D.eval()

    rows = []
    with open(test_dir) as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    for i, sublist in enumerate (tqdm(rows)):

        BTF_np = np.array(sublist[1:]).astype(np.float)
        BTF = torch.tensor(BTF_np, dtype=torch.float32).unsqueeze(0)
        filename = sublist[0]
        BTF = BTF.to(params.device)
        z = generateZ(args, 1)
        fake = G(z, BTF)
        samples = fake.cpu().data[:1].squeeze().numpy()

        # Visualization and saving generated image
        if not args.use_visdom:
            SavePloat_Voxels(samples, image_saved_path, filename)  # norm_
        else:
            plotVoxelVisdom(samples[0, :], vis, "tester_" + str(i))