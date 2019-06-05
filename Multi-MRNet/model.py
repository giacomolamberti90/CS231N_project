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
        x = torch.squeeze(self.classifier(x))
        
        return x    
    
class TripleMRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.batchnorm2d = nn.BatchNorm2d(256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256*3, 3)
        
    def forward(self, vol_axial, vol_sagit, vol_coron):
        vol_axial = torch.squeeze(vol_axial, dim=0)
        vol_sagit = torch.squeeze(vol_sagit, dim=0)
        vol_coron = torch.squeeze(vol_coron, dim=0)
       
        vol_axial = self.model.features(vol_axial)
        vol_sagit = self.model.features(vol_sagit)
        vol_coron = self.model.features(vol_coron)

        vol_axial = self.gap(vol_axial).view(vol_axial.size(0), -1)
        vol_axial = torch.max(vol_axial, 0, keepdim=True)[0]
        
        vol_sagit = self.gap(vol_sagit).view(vol_sagit.size(0), -1)
        vol_sagit = torch.max(vol_sagit, 0, keepdim=True)[0]
        
        vol_coron = self.gap(vol_coron).view(vol_coron.size(0), -1)
        vol_coron = torch.max(vol_coron, 0, keepdim=True)[0]

        vol = torch.cat((vol_axial, vol_sagit, vol_coron), 1)
        out = torch.squeeze(self.classifier(vol))
        
        return out
    
class TripleRnnMRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        #self.features = nn.Sequential(*list(self.model.classifier.children())[:0])
        self.lstm = nn.LSTM(256*7*7, 3, 1, batch_first=True)
        self.classifier = nn.Linear(9, 3)

    def forward(self, vol_axial, vol_sagit, vol_coron):

        vol_axial = torch.squeeze(vol_axial, dim=0)
        vol_sagit = torch.squeeze(vol_sagit, dim=0)
        vol_coron = torch.squeeze(vol_coron, dim=0)
          
        vol_axial = self.model(vol_axial).view(1,10,256*7*7)
        vol_sagit = self.model(vol_sagit).view(1,10,256*7*7)
        vol_coron = self.model(vol_coron).view(1,10,256*7*7)
                
        vol_axial, _ = self.lstm(vol_axial)
        vol_sagit, _ = self.lstm(vol_sagit)
        vol_coron, _ = self.lstm(vol_coron)
                
        vol_axial = torch.squeeze(vol_axial[:,-1,:])
        vol_sagit = torch.squeeze(vol_sagit[:,-1,:])
        vol_coron = torch.squeeze(vol_coron[:,-1,:])
                
        vol = torch.cat((vol_axial, vol_sagit, vol_coron), 0)        
        out = self.classifier(vol)
                
        return out
