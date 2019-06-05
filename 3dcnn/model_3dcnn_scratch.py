# source: https://worksheets.codalab.org/rest/bundles/0x06b2964943264afd91355df46aa085aa/contents/blob/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv3d(3, 5, (5,5,5), stride=1, padding=2)
        self.pool1 = nn.MaxPool3d((2,2,2), stride=2)
        self.conv2 = nn.Conv3d(5, 7, (3,3,3), stride=2, padding=1)
        self.pool2 = nn.MaxPool3d((2,2,2), stride=2)
        self.conv3 = nn.Conv3d(7, 9, (3,3,3), stride=2, padding=1)
        self.fc1 = nn.Linear(16, 1)
        self.fc2 = nn.Linear(144, 1)

    def forward(self, x):
        
#         x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 144)
        x = self.fc2(x)
#         x = x.view(-1)
        
        return x

class TripleMRNet(nn.Module):
    def __init__(self):
        super().__init__()
#         self.model = models.alexnet(pretrained=True)
#         self.batchnorm2d = nn.BatchNorm2d(256)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(256*3, 3)
        self.use_gpu = True
        self.conv1 = nn.Conv3d(3, 5, (5,5,5), stride=1, padding=2)
        self.pool1 = nn.MaxPool3d((2,2,2), stride=2)
        self.conv2 = nn.Conv3d(5, 7, (3,3,3), stride=2, padding=1)
        self.pool2 = nn.MaxPool3d((2,2,2), stride=2)
        self.conv3 = nn.Conv3d(7, 9, (3,3,3), stride=2, padding=1)
        self.fc1 = nn.Linear(16, 1)
        self.classifier = nn.Linear(144*3, 3)
        
    def forward(self, vol_axial, vol_sagit, vol_coron):
#         vol_axial = torch.squeeze(vol_axial, dim=0)
#         vol_sagit = torch.squeeze(vol_sagit, dim=0)
#         vol_coron = torch.squeeze(vol_coron, dim=0)
        
        vol_axial = vol_axial.permute(0, 2, 1, 3, 4)
        vol_sagit = vol_sagit.permute(0, 2, 1, 3, 4)
        vol_coron = vol_coron.permute(0, 2, 1, 3, 4)
        
        vol_axial = self.conv1(vol_axial)
        vol_sagit = self.conv1(vol_sagit)
        vol_coron = self.conv1(vol_coron)
        
        vol_axial = self.pool1(vol_axial)
        vol_sagit = self.pool1(vol_sagit)
        vol_coron = self.pool1(vol_coron)
        
        vol_axial = self.conv2(vol_axial)
        vol_sagit = self.conv2(vol_sagit)
        vol_coron = self.conv2(vol_coron)

        vol_axial = self.pool2(vol_axial)
        vol_sagit = self.pool2(vol_sagit)
        vol_coron = self.pool2(vol_coron)

        vol_axial = self.conv3(vol_axial)
        vol_sagit = self.conv3(vol_sagit)
        vol_coron = self.conv3(vol_coron)

        vol_axial = F.relu(self.fc1(vol_axial))
        vol_sagit = F.relu(self.fc1(vol_sagit))
        vol_coron = F.relu(self.fc1(vol_coron))
        
        vol_axial = vol_axial.view(-1, 144)
        vol_sagit = vol_sagit.view(-1, 144)
        vol_coron = vol_coron.view(-1, 144)

#         vol_axial = self.gap(vol_axial).view(vol_axial.size(0), -1)
#         vol_axial = torch.max(vol_axial, 0, keepdim=True)[0]
        
#         vol_sagit = self.gap(vol_sagit).view(vol_sagit.size(0), -1)
#         vol_sagit = torch.max(vol_sagit, 0, keepdim=True)[0]
        
#         vol_coron = self.gap(vol_coron).view(vol_coron.size(0), -1)
#         vol_coron = torch.max(vol_coron, 0, keepdim=True)[0]

        vol = torch.cat((vol_axial, vol_sagit, vol_coron), 1)

        out = self.classifier(vol)
        return out  