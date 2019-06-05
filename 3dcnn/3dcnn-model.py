import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.batchnorm2d = nn.BatchNorm2d(256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 128)
        self.classifier = nn.Linear(256, 3)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        #x = self.batchnorm2d(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        #x = F.relu(self.fc(x))
        x = self.classifier(x)
        
        return x

class TripleMRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.conv1 = nn.Conv3d(16, 5, (3,3,3), stride=1, padding=2)
        self.conv2 = nn.Conv3d(5, 3, (3,3,3), stride=1, padding=2)  #channels 5, 3
        self.conv3 = nn.Conv3d(3, 3, (3,3,3), stride=1, padding=2)  # channels 3,3
        
        self.conv1b = nn.Conv3d(16, 5, (3,3,3), stride=1, padding=2)
        self.conv2b = nn.Conv3d(5, 3, (3,3,3), stride=1, padding=2)  #channels 5, 3
        self.conv3b = nn.Conv3d(3, 3, (3,3,3), stride=1, padding=2) 
        
        self.conv1c = nn.Conv3d(16, 5, (3,3,3), stride=1, padding=2)
        self.conv2c = nn.Conv3d(5, 3, (3,3,3), stride=1, padding=2)  #channels 5, 3
        self.conv3c = nn.Conv3d(3, 3, (3,3,3), stride=1, padding=2) 
#         self.batchnorm2d = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool3d((2,2,2), stride=2)
        self.pool2 = nn.MaxPool3d((2,2,2), stride=2)
        self.pool3 = nn.MaxPool3d((2,2,2), stride=2)
#         self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(396*3, 3)
        
    def forward(self, vol_axial, vol_sagit, vol_coron):
#         print("\n")
#         print("input:", vol_axial.size())
        vol_axial = torch.squeeze(vol_axial, dim=0)
        vol_sagit = torch.squeeze(vol_sagit, dim=0)
        vol_coron = torch.squeeze(vol_coron, dim=0)
       
#         print("after squeeze:", vol_axial.size())
        vol_axial = self.model.features(vol_axial)
        vol_sagit = self.model.features(vol_sagit)
        vol_coron = self.model.features(vol_coron)
        
#         print("alexnet output:", vol_axial.size())
        vol_axial = torch.unsqueeze(vol_axial, dim=0)
        vol_sagit = torch.unsqueeze(vol_sagit, dim=0)
        vol_coron = torch.unsqueeze(vol_coron, dim=0)

#         print("unsqueeze:", vol_axial.size())
        vol_axial = F.relu(self.conv1(vol_axial))#.view(vol_axial.size(0), -1)
#         print("conv1:", vol_axial.size())
        vol_axial = self.pool1(vol_axial)
#         print("pool1:", vol_axial.size())
#         vol_axial = torch.max(vol_axial, 0, keepdim=True)[0]
        vol_axial = F.relu(self.conv2(vol_axial))#.view(vol_axial.size(0), -1)
#         print("conv2:", vol_axial.size())
        vol_axial = self.pool2(vol_axial)
#         print("pool2:", vol_axial.size())
        vol_axial = F.relu(self.conv3(vol_axial))#.view(vol_axial.size(0), -1)
#         print("conv3:", vol_axial.size())
        vol_axial = self.pool3(vol_axial)
#         print("pool3:", vol_axial.size())
        vol_axial = vol_axial.view(vol_axial.size(0), -1)
#         print("reshape:", vol_axial.size())
        
        vol_sagit = F.relu(self.conv1b(vol_sagit))#.view(vol_sagit.size(0), -1)
        vol_sagit = self.pool1(vol_sagit)
#         vol_sagit = torch.max(vol_sagit, 0, keepdim=True)[0]
        vol_sagit = F.relu(self.conv2b(vol_sagit))#.view(vol_sagit.size(0), -1)
        vol_sagit = self.pool2(vol_sagit)
        vol_sagit = F.relu(self.conv3b(vol_sagit))#.view(vol_sagit.size(0), -1)
        vol_sagit = self.pool3(vol_sagit)
        vol_sagit = vol_sagit.view(vol_sagit.size(0), -1)
        
        vol_coron = F.relu(self.conv1c(vol_coron))#.view(vol_coron.size(0), -1)
        vol_coron = self.pool1(vol_coron)
#         vol_coron = torch.max(vol_coron, 0, keepdim=True)[0]
        vol_coron = F.relu(self.conv2c(vol_coron))#
        vol_coron = self.pool2(vol_coron)
        vol_coron = F.relu(self.conv3c(vol_coron))#
        vol_coron = self.pool3(vol_coron)
        vol_coron = vol_coron.view(vol_coron.size(0), -1)
        
        
        vol = torch.cat((vol_axial, vol_sagit, vol_coron), 1)
#         print("cat: ", vol.size()) 
        out = self.classifier(vol)
#         print("out: ", out.size()) 
        return out