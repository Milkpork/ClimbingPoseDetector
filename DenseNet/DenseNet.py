import math
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization(dic):
    if len(dic) == 6:
        value = np.array(dic)
        M_m = np.max(value) - np.min(value)
        minNum = np.min(value)
        value = (value - minNum) / M_m
        return np.array(value[:])
    elif len(dic) >= 0:
        raise ValueError("not enough 15 parameters")


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.LeakyReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.LeakyReLU(inplace=False))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        # print(x.shape)
        # print(new_features.shape)
        return torch.cat([x, new_features], 1)


class Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""

    def __init__(self, num_input_feature, num_output_features):
        super(Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.LeakyReLU(inplace=False))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module("pool", nn.AvgPool2d(2, stride=2))
        # self.add_module("pool", nn.AdaptiveAvgPool2d((8, 3)))


class ResNet(nn.Module):
    def __init__(self, featureSize=4, outSize=1):
        super(ResNet, self).__init__()
        self.norm = nn.BatchNorm2d(featureSize)
        self.relu = nn.LeakyReLU(inplace=False)
        self.linear = nn.Linear(12, outSize)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = x.view(1, 12)
        x = self.linear(x)
        return x


class DenseNet(nn.Module):
    """DenseNet-BC model"""
    model_path = "model_data/model_data.pth"

    def __init__(self, path='./', growth_rate=32, num_init_features=2, bn_size=2, drop_rate=0, cuda=False):
        super(DenseNet, self).__init__()
        self.cuda = cuda

        self.model_path = path + self.model_path
        self.block1 = DenseLayer(num_init_features, growth_rate, bn_size, drop_rate)
        self.block2 = DenseLayer(num_init_features, growth_rate, bn_size, drop_rate)
        self.block3 = DenseLayer(num_init_features, growth_rate, bn_size, drop_rate)
        self.block4 = DenseLayer(num_init_features, growth_rate, bn_size, drop_rate)
        self.block5 = DenseLayer(num_init_features, growth_rate, bn_size, drop_rate)
        self.blockList = [self.block1, self.block2, self.block3, self.block4, self.block5]

        self.tran1 = Transition(34, 2)
        self.tran2 = Transition(68, 2)
        self.tran3 = Transition(102, 2)
        self.tran4 = Transition(136, 2)
        self.tran5 = Transition(170, 2)
        self.tranList = [self.tran1, self.tran2, self.tran3, self.tran4, self.tran5]

        self.resNet = ResNet(2, 1)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.load_state_dict(torch.load(self.model_path, map_location=device))
            print("load dict success")
        except FileNotFoundError:
            print("load dict2 failed")
            pass

    def forward(self, x):
        tx = normalization(x)
        x = tx - tx[0]
        x = torch.Tensor(x)
        if self.cuda:
            x = x.cuda()
        x = x.view(1, 2, 3, 2)
        nowOut = None
        nextInput = x
        for i in range(5):
            out = self.blockList[i](nextInput)
            out = out.view(1, -1, 3, 2)
            if nowOut is None:
                nowOut = out
            else:
                nowOut = torch.cat([nowOut, out], 1)
            nextInput = self.tranList[i](nowOut)
        nextInput = self.resNet(nextInput)
        return nextInput[0]


if __name__ == '__main__':
    clor = DenseNet('../')
    t = time.time()
    for i in range(1):
        nodes = [[-0.004026244088522285, -0.024157464531133738],
                 [0.0, -0.024157464531133738],
                 [0.004026244088522299, -0.028183708619656023],
                 [-0.012078732265566855, -0.020131220442611425],
                 [-0.012078732265566855, -0.024157464531133738],
                 [-0.012078732265566855, -0.024157464531133738]]

        res = clor(nodes)
        print(res)
    print(time.time() - t)
