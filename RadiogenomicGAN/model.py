import torch
import torch.nn as nn
import params
from torch.autograd import Variable
import math
import torch.nn.functional as F

# Convolution Operation with weight normalization technique
class WN_Conv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))

        self.train_scale = train_scale
        self.init_mode = False
        self.init_stdv = init_stdv
        self.kernel_size = kernel_size

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # normalize weight matrix and linear projection [out x in x h x w x z]
        # for each output dimension, normalize through (in, h, w, z) = (1, 2, 3, 4) dims

        # This is done to ensure padding as "SAME"
        #print(input.shape)
        pad_h = math.ceil((self.kernel_size[0] - input.shape[2] * (1 - self.stride[0]) - self.stride[0]) / 2)
        pad_w = math.ceil((self.kernel_size[1] - input.shape[3] * (1 - self.stride[1]) - self.stride[1]) / 2)
        pad_z = math.ceil((self.kernel_size[2] - input.shape[4] * (1 - self.stride[2]) - self.stride[2]) / 2)
        padding = (pad_h, pad_w, pad_z)

        norm_weight = self.weight * (weight_scale[:,None,None,None,None] / torch.sqrt((self.weight ** 2).sum(4).sum(3).sum(2).sum(1) + 1e-6).reshape([-1, 1, 1, 1, 1])).expand_as(self.weight)
        norm_weight = norm_weight.cuda()
        activation = F.conv3d(input, norm_weight, bias=None,
                              stride=self.stride, padding=padding,
                              dilation=self.dilation, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(4).mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(4).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None,None].expand_as(activation)

        return activation

class WN_ConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_ConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))

        self.train_scale = train_scale
        self.init_mode = False
        self.init_stdv = init_stdv
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input, output_size=None):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # normalize weight matrix and linear projection [in x out x h x w x z]
        # for each output dimension, normalize through (in, h, w, z)  = (0, 2, 3, 4) dims

        if self.in_channels == self.out_channels:
            norm_weight = self.weight * (weight_scale[None,:,None,None,None] / torch.sqrt((self.weight ** 2).sum(4).sum(3).sum(2).sum(0) + 1e-6).reshape([-1, 1, 1, 1, 1])).expand_as(self.weight)
        else:
            norm_weight = self.weight * (weight_scale[None,:,None,None,None] / torch.sqrt((self.weight ** 2).sum(4).sum(3).sum(2).sum(0) + 1e-6)).expand_as(self.weight)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        activation = F.conv_transpose3d(input, norm_weight, bias=None,
                                        stride=self.stride, padding=self.padding,
                                        output_padding=output_padding, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(4).mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(4).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None,None].expand_as(activation)

        return activation



class net_G(torch.nn.Module):
    def __init__(self, args):
        super(net_G, self).__init__()
        self.args = args
        self.cube_len = params.cube_len
        self.bias = params.bias
        self.z_dim = params.z_dim + 17
        self.f_dim = 32
        # self.btf_embedding = torch.nn.Linear(17,200,bias=self.bias)
        # self.btf_embedding.weight.requires_grad = False
        # self.btf_embedding = self.btf_embedding.double()


        padd = (0, 0, 0)
        padd128 = (1,1,1)
        padd32 = (2,1,1)
        if self.cube_len == 32:
            padd = (1,1,1)

###################  shape model G ######################

#3,2,0

        self.layer1 = self.conv_layer(self.z_dim, self.f_dim*8, kernel_size=(4,8,8), stride=2, padding=padd, bias=self.bias)
        self.layer111 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer112 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer121 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer122 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer131 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer132 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer141 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer142 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer151 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer152 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer161 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer162 = self.conv_layer(self.f_dim*8, self.f_dim*8, kernel_size=3, stride=1, padding=1, bias=self.bias)





        self.layer2 = self.conv_layer(self.f_dim*8+self.f_dim*8, self.f_dim*4, kernel_size=(2,4,4), stride=2, padding=padd128, bias=self.bias)
        self.layer211 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer212 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer221 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer222 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer231 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer232 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer241 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer242 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer251 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer252 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer261 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer262 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer271 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer272 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer281 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer282 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer291 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer292 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer2101 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer2102 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer2111 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer2112 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer2121 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer2122 = self.conv_layer(self.f_dim*4, self.f_dim*4, kernel_size=3, stride=1, padding=1, bias=self.bias)



        self.layer3 = self.conv_layer(self.f_dim*4+self.f_dim*4, self.f_dim*2, kernel_size=(2,4,4), stride=2, padding=padd128, bias=self.bias)
        self.layer311 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer312 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer321 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer322 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer331 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer332 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer341 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer342 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer351 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer352 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer361 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer362 = self.conv_layer(self.f_dim*2, self.f_dim*2, kernel_size=3, stride=1, padding=1, bias=self.bias)




        self.layer4 = self.conv_layer(self.f_dim*2+self.f_dim*2, self.f_dim, kernel_size=(2,4,4), stride=2, padding=padd128, bias=self.bias)
        self.layer411 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer412 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer421 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer422 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer431 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer432 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer441 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer442 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer451 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer452 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer461 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer462 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer471 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer472 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer481 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer482 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer491 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer492 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer4101 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer4102 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer4111 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer4112 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer4121 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.layer4122 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer5 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=(2,4,4), stride=(2,2,2), padding=padd32, bias=self.bias)
        self.layer6 = self.conv_layer(self.f_dim, self.f_dim, kernel_size=(2,4,4), stride=2, padding=padd, bias=self.bias)

        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.f_dim+self.f_dim, 1, kernel_size=(2,4,4), stride=(2,2,2), bias=self.bias, padding=padd32),
            torch.nn.Sigmoid()
            # torch.nn.Tanh()
        )

    def conv_layer(self, input_dim, output_dim, kernel_size, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.ReLU(True)
            # torch.nn.LeakyReLU(self.leak_value, True)
        )
        return layer


    def forward(self, x, btf):
        btf = btf.float()
        print(btf.size())
        print(x.size())
        x = torch.cat((btf, x),1)
        print('x+btf',x.size())
        out = x.view(-1, self.z_dim, 1, 1, 1)
        out = self.layer1(out)
        print(out.size())
        # torch.Size([32, 256, 2, 2, 2])
        out1 = self.layer111(out)
        print(out1.size())
        out = self.layer112(out1)
        out = self.layer121(out)
        out = self.layer122(out)
        out = self.layer131(out)
        out = self.layer132(out)

        out = self.layer141(out)
        out = self.layer142(out)
        out = self.layer151(out)
        out = self.layer152(out)
        out = self.layer161(out)
        out2 = self.layer162(out)
        print(out2.size())
        out = torch.cat((out1,out2),1)

        print(out.size())
        out = self.layer2(out)
        print(out.size())
        # torch.Size([32, 128, 4, 4, 4])
        out1 = self.layer211(out)
        out = self.layer212(out)
        out = self.layer221(out)
        out = self.layer222(out)
        out = self.layer231(out)
        out = self.layer232(out)

        out = self.layer241(out)
        out = self.layer242(out)
        out = self.layer251(out)
        out = self.layer252(out)
        out = self.layer261(out)
        out = self.layer262(out)

        out = self.layer271(out)
        out = self.layer272(out)
        out = self.layer281(out)
        out = self.layer282(out)
        out = self.layer291(out)
        out = self.layer292(out)

        out = self.layer2101(out)
        out = self.layer2102(out)
        out = self.layer2111(out)
        out = self.layer2112(out)
        out = self.layer2121(out)
        out2 = self.layer2122(out)

        out = torch.cat((out1,out2),1)

        out = self.layer3(out)
        print(out.size())
        # torch.Size([32, 64, 8, 8, 8])
        out1 = self.layer311(out)
        out = self.layer312(out)
        out = self.layer321(out)
        out = self.layer322(out)
        out = self.layer331(out)
        out = self.layer332(out)

        out = self.layer341(out)
        out = self.layer342(out)
        out = self.layer351(out)
        out = self.layer352(out)
        out = self.layer361(out)
        out2 = self.layer362(out)

        out = torch.cat((out1,out2),1)

        out = self.layer4(out)
        print(out.size())
        # torch.Size([32, 32, 16, 16, 16])
        out1 = self.layer411(out)
        out = self.layer412(out)
        out = self.layer421(out)
        out = self.layer422(out)
        out = self.layer431(out)
        out = self.layer432(out)

        out = self.layer441(out)
        out = self.layer442(out)
        out = self.layer451(out)
        out = self.layer452(out)
        out = self.layer461(out)
        out = self.layer462(out)

        out = self.layer471(out)
        out = self.layer472(out)
        out = self.layer481(out)
        out = self.layer482(out)
        out = self.layer491(out)
        out = self.layer492(out)

        out = self.layer4101(out)
        out = self.layer4102(out)
        out = self.layer4111(out)
        out = self.layer4112(out)
        out = self.layer4121(out)
        out2 = self.layer4122(out)

        out = torch.cat((out1,out2),1)
        out = self.layer7(out)

        print(out.size())

        return out


class net_D(torch.nn.Module):
    def __init__(self, args):
        super(net_D, self).__init__()
        self.args = args
        self.cube_len = params.cube_len
        self.leak_value = params.leak_value
        self.bias = params.bias

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)



        self.f_dim = 32
        self.btf_embedding = torch.nn.Linear(17,32*self.cube_len*self.cube_len,bias=self.bias)
        self.btf_embedding.weight.requires_grad = False
        self.btf_embedding = self.btf_embedding.double()


#####################  shape model_D #############################

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.f_dim*8, 1, kernel_size=(2,8,8), stride=2, bias=self.bias, padding=padd),
            torch.nn.Sigmoid()
        )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer



    def forward(self, x, btf):

        btf = btf.float()
        btf = self.btf_embedding(btf)
        btf = btf.view(btf.shape[0], 1, 32, self.cube_len,self.cube_len)
        x = x.view(x.shape[0],1,32,self.cube_len,self.cube_len)
        x = x + x*btf
        #print('x',x.size())
        out = x.view(-1, 1, 32, self.cube_len, self.cube_len)
        #print('out',out.size()) # torch.Size([32, 1, 32, 32, 32])

        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        # out = out.view(-1, 256*2*2*2)
        out = self.layer5(out)
        #print(out.size())  # torch.Size([32, 1, 1, 1, 1])
        out = torch.squeeze(out)
        return out

