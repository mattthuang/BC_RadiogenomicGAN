import torch
torch.cuda.set_device(0)
epochs = 1500
batch_size = 2
soft_label = False
adv_weight = 0
d_thresh = 0.8
z_dim = 200
z_dis = "norm"
model_save_step = 1
g_lr = 0.000025
d_lr = 0.00001
beta = (0.5, 0.999)
cube_len = 128
leak_value = 0.2
bias = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
data_dir = '../matched_sideview/'
model_dir = 'chair/'
BTF_dir = '../BTF_side.csv'
test_dir= '../Mu_paired_test.csv'
output_dir = '../outputs'
images_dir = '../test_outputs'
visualized = '../visualization'


def print_params():
    l = 16
    print(l * '*' + 'hyper-parameters' + l * '*')

    print('epochs =', epochs)
    print('batch_size =', batch_size)
    print('soft_labels =', soft_label)
    print('adv_weight =', adv_weight)
    print('d_thresh =', d_thresh)
    print('z_dim =', z_dim)
    print('z_dis =', z_dis)
    print('model_images_save_step =', model_save_step)
    print('data =', model_dir)
    print('device =', device)
    print('g_lr =', g_lr)
    print('d_lr =', d_lr)
    print('cube_len =', cube_len)
    print('leak_value =', leak_value)
    print('bias =', bias)

    print(l * '*' + 'hyper-parameters' + l * '*')