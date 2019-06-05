from cams import save_cams
import os
import torch.nn as nn
from torchvision import models

from os import path
from evaluate import evaluate

import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib

from loader import load_data

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = models.alexnet(pretrained=True)

    def forward(self, x):
        x = self.conv.features(x)
        return x

preds, labels = evaluate('MRNet-v1.0/', "test", 'models/weights_baseline', use_gpu=True)
train_loader, valid_loader, test_loader = load_data('MRNet-v1.0/', use_gpu=True)

preds_npy = np.asarray(preds)
labels_npy = np.asarray(labels)

index = (labels_npy[:,2] == 1) * (preds_npy[:,2] < 0.5)
print(index)

index_max = np.argmin(preds_npy[:,2] * index)
print(index_max)

for index, batch in enumerate(test_loader):
    vol_axial_tensor, vol_coronal_tensor, vol_sagittal_tensor, label_tensor = batch
    
    cams = []
    cnn_model = CNN()
    for i in range(vol_sagittal_tensor.size(1)):
        cams.append(cnn_model(vol_sagittal_tensor[:,i,:,:,:]).detach().numpy().reshape(256, 7, 7))
        
    if index == index_max:
        # True Positive, cams is just list of AlexNet feature maps
        save_cams(cnn_model, cams)