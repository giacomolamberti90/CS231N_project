import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, Subset

from torch.autograd import Variable
from torchvision import transforms
import torchsample
import random 

INPUT_DIM = 256
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class Dataset(Dataset):
    def __init__(self, path, split, use_gpu, transform=None):
        super().__init__()
        self.use_gpu = use_gpu
        self.images = []
        self.labels = []
        self.path = path
        self.transform = transform
        
        if split == 'test':
            self.path += 'valid'
        else:
            self.path += 'train'
            
        directory = self.path + '/axial/'
                
        all_images = []
        num_images = 0
        for image in os.listdir(directory):
            if image != '.DS_Store':
                all_images.append(image)
                num_images += 1
                                        
        all_images = sorted(all_images)
        
        all_labels = []
        all_labels = np.zeros((num_images,3))
        for index, task in enumerate(['abnormal', 'acl', 'meniscus']):
            with open(self.path + '-' + task + '.csv') as f:
                all_labels[:,index] = pd.read_csv(f, header = None)[1]
                        
        if split != 'test':
            val_images = []
            val_labels = []
            val_indices = set()
            for i, label in enumerate(all_labels[:,0]):
                if (label == 1 and len(val_images) < 50):
                    val_images.append(all_images[i])
                    val_labels.append(all_labels[i])
                    val_indices.add(i)
            for i, label in reversed(list(enumerate(all_labels[:,0]))):
                if (len(val_images) < 120 and i not in val_indices):
                    val_images.append(all_images[i])
                    val_labels.append(all_labels[i])
                    val_indices.add(i)
            train_images = [x for x in all_images if x not in set(val_images)]
            train_labels = []
            for i in range(all_labels.shape[0]):
                if i not in val_indices:
                    train_labels.append(all_labels[i])         
                    
        if split == "train":
            self.images = train_images
            self.labels = np.asarray(train_labels)
        if split == "valid":
            self.images = val_images
            self.labels = np.asarray(val_labels)
        if split == 'test':
            self.images = all_images
            self.labels = all_labels            
                           
        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]
    
    def weighted_loss(self, prediction, target):
        weights_npy = np.zeros((1,3))
        for i in range(3):
            weights_npy[:,i] = np.array([self.weights[int(t.item())] for t in target.data.reshape(1,3)[:,i]])
            
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
            
        loss = F.binary_cross_entropy_with_logits(prediction.reshape(1,3), target.reshape(1,3), weight=Variable(weights_tensor))
        return loss
    
    def standard_loss(self, prediction, target):
        
        loss = F.binary_cross_entropy_with_logits(prediction, target.reshape(1,3))
            
        return loss
    
    def __getitem__(self, index):
        image = self.images[index]
        vol_axial = np.load(os.path.join(self.path, "axial", image))
        vol_coronal = np.load(os.path.join(self.path, "coronal", image))
        vol_sagittal = np.load(os.path.join(self.path, "sagittal", image))
        
        # standardize
        vol = vol_axial
        
        if self.transform:
            vol_transform = np.zeros_like(vol)
            for i in range(vol.shape[0]):
                angle = random.choice([-30, -15, 15, 30])
                vol_rotate = transforms.ToPILImage()(vol[i,:,:])
                vol_transform[i,:,:] = transforms.functional.rotate(vol_rotate, angle)
            vol = np.concatenate((vol, vol_transform), axis=0)        
        
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL               
        vol = (vol - MEAN) / STDDEV  
        vol = np.stack((vol,)*3, axis=1)
        
        vol_axial_tensor = torch.FloatTensor(vol)
        
        # standardize
        vol = vol_coronal
                
        if self.transform:
            vol_transform = np.zeros_like(vol)
            for i in range(vol.shape[0]):
                angle = random.choice([-30, -15, 15, 30])
                vol_rotate = transforms.ToPILImage()(vol[i,:,:])
                vol_transform[i,:,:] = transforms.functional.rotate(vol_rotate, angle)
            vol = np.concatenate((vol, vol_transform), axis=0)  
            
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL               
        vol = (vol - MEAN) / STDDEV  
        vol = np.stack((vol,)*3, axis=1)           
        
        vol_coronal_tensor = torch.FloatTensor(vol)
        
        # standardize
        vol = vol_sagittal
        
        if self.transform:
            vol_transform = np.zeros_like(vol)
            for i in range(vol.shape[0]):
                angle = random.choice([-30, -15, 15, 30])
                vol_rotate = transforms.ToPILImage()(vol[i,:,:])
                vol_transform[i,:,:] = transforms.functional.rotate(vol_rotate, angle)
            vol = np.concatenate((vol, vol_transform), axis=0)                         
        
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL               
        vol = (vol - MEAN) / STDDEV  
        vol = np.stack((vol,)*3, axis=1)
        
        vol_sagittal_tensor = torch.FloatTensor(vol)

        label_tensor = torch.FloatTensor([self.labels[index]])
                
        return vol_axial_tensor, vol_coronal_tensor, vol_sagittal_tensor, label_tensor

    def __len__(self):
        return len(self.images)
    
def load_data(path, use_gpu=True): 

    train_data = Dataset(path, 'train', use_gpu=True, transform=True)
    valid_data = Dataset(path, 'valid', use_gpu=True, transform=True)
    test_data = Dataset(path, 'test', use_gpu)
    
    train_loader = DataLoader(train_data, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=8, shuffle=False)
        
    return train_loader, valid_loader, test_loader

     