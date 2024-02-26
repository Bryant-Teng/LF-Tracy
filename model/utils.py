import torch
import torch.nn as nn
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Parameter, Linear, Softmax, Dropout, Embedding
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

class IF(Module):
    """ Position attention module"""
    def __init__(self, in_dim,batch_size):
        super(IF, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gammas = []
        for i in range(0, 12):
            gamma = Parameter(torch.zeros(1)).to('cuda')
            self.gammas.append(gamma)
        self.softmax = Softmax(dim=-1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, stack_list, y):
        outs=[]
        for i in stack_list:
            m_batchsize, C,height, width = i.size()
            proj_query = self.query_conv(i).view(m_batchsize, -1, width*height).permute(0, 2, 1)
            proj_key = self.key_conv(i).view(m_batchsize, -1, width*height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
    
            proj_value = self.value_conv(y).view(m_batchsize, -1, width*height)

            out_ = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out_ = out_.view(m_batchsize, C, height, width)
            outs.append(out_)

        out = [a * b for a,b in zip(outs,self.gammas)]
        out = sum(out) + y
        return out
