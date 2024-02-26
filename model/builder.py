import torch.nn as nn
from bakebone.pvtv2 import pvt_v2_b2
import torch.nn.functional as F
from model.utils2 import Part2

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.focal_encoder = pvt_v2_b2()
        self.decoder = Part2()

    def forward(self, x,x_stack, y):
        image = x_stack
        image.append(y)
        y,out1,out2,out3,out4 = self.focal_encoder(image)  
        rgb_out = self.Part2(y)
        rgb_out = F.interpolate(rgb_out, size=(256, 256), mode='bilinear', align_corners=False)
        return rgb_out,out1,out2,out3,out4
