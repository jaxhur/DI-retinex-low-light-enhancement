import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

class enhance_net_nopool(nn.Module):

    def __init__(self, ch=6, number_f=64, linear=False, linear_size=16):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.identity = nn.Identity()

        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,ch,3,1,1,bias=True) 
        self.linear = linear
        if self.linear:
            self.linear_size = linear_size
            self.linear1 = nn.Linear(3*linear_size**2, 256)
            self.linear2 = nn.Linear(256, ch)

    def brightness_contrast(self, x, c, b, mapfunc="tan"):
        if mapfunc == "tan":
            k = torch.tan((45 - 44.8 * c) / 180 * np.pi)
        elif mapfunc == "diff_div":
            k = (1. + x) / (1. - x + 1e-8)
        elif mapfunc == "arctanh":
            k = 3 * 1.5 * torch.log((1.5 + 0.5 * x) / (0.5 - 0.5 * x + 1e-8))
        elif mapfunc == 'logtan':
            k = 3 * torch.log(torch.tan((45 - 44.8 * c) / 180 * np.pi) + 1)
        mean = 0.5 # x.mean((2,3),keepdim=True) # 0.5
        y = k * (x + mean * (b - 1)) + mean * (b + 1)
        y = torch.clamp(y, 0, 1)
        return y, k
    
    def forward(self, x, ):
        if self.linear:
            n = x.shape[0]
            x_ = F.interpolate(x, size=(self.linear_size, self.linear_size))
            x_ = x_.reshape(n, -1)
            x_ = self.relu(self.linear1(x_))
            x_ = self.identity(self.linear2(x_))
            half_ch = x_.shape[1] // 2
            c0, b0 = x_[:,:half_ch].view(n,-1,1,1), x_[:,half_ch:].view(n,-1,1,1)
            # print(c0.item(), b0.item())
        else:
            c0, b0 = 0, 0
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))

        x3 = self.e_conv3(x2)
        half_ch = x3.shape[1] // 2
        c, b = c0 + x3[:,:half_ch], b0 + x3[:,half_ch:]
        # p2 = self.maxpool(x2)
        c = self.tanh(c)
        b = self.sigmoid(b)
        enhance_image, k = self.brightness_contrast(x, c, b)
        return enhance_image, c, b, k



