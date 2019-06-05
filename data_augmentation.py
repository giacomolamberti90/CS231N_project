import pickle
import pandas as pd
import argparse
import json
import numpy as np
import os
import torch

from datetime import datetime
from pathlib import Path
from sklearn import metrics

from torchvision import transforms

def flip_image(vol_tensor, flip_orientation, image_idx, directory):
    vol_transformed = np.zeros_like(vol_tensor)
    for i in range(vol_tensor.shape[0]):
        vol_transformed_ = transforms.ToPILImage()(vol_tensor[i,:,:])
        transformation = None
        if flip_orientation == 'horizontal':
            transformation = transforms.RandomHorizontalFlip(p=0.5)
        else: 
            transformation = transforms.RandomVerticalFlip(p=0.5)
        vol_transformed_ = transformation(vol_transformed_)   
        vol_transformed[i,:,:] = torch.tensor(np.array(vol_transformed_))
#     print(vol_tensor.size())
#     print(torch.tensor(vol_transformed).size())
    image_outfile = str(image_idx) + '.npy'
#     print(image_outfile)
#     np.save(directory + image_outfile, vol_tranformed) 

def rotate_image(vol_tensor, angle, image_idx, directory):
    vol_transformed = np.zeros_like(vol_tensor)
    for i in range(vol_tensor.shape[0]):
        vol_transformed_ = transforms.ToPILImage()(vol_tensor[i,:,:])
        vol_transformed_ = transforms.functional.rotate(vol_transformed_, angle)
        vol_transformed[i,:,:] = torch.tensor(np.array(vol_transformed_))
    
    image_outfile = str(image_idx) + '.npy'
#     print(image_outfile)
#     np.save(directory + image_outfile, vol_tranformed) 
    
def augment_plane(path, split, plane, abnormal_labels, acl_labels, meniscus_labels):
    images = []
    directory = path + split + '/' + plane + '/'
    for image in os.listdir(directory):
        if image != '.DS_Store':
            images.append(directory + image)
    images.sort()
    
    image_idx = 1130
    
    for index in range(len(images)):
        path = images[index]
        with open(path, 'rb') as file_handler: # Must use 'rb' as the data is binary
            vol = np.load(file_handler).astype(np.int32)
            if (abnormal_labels[index] == 0):
                vol_tensor = torch.FloatTensor(vol)
                             
                angles = [-15, 15]
                for angle in angles:
                    rotate_image(vol_tensor, angle, image_idx, directory)
                    image_idx += 1
                    
                    if plane == 'axial':
                        abnormal_labels.append(0)
                        acl_labels.append(acl_labels[index])
                        meniscus_labels.append(meniscus_labels[index])
                
                flip_image(vol_tensor, 'horizontal', image_idx, directory)
                image_idx += 1
            
                # only need to update abnormal_labels for one plane
                if plane == 'axial':
                    abnormal_labels.append(0)
                    acl_labels.append(acl_labels[index])
                    meniscus_labels.append(meniscus_labels[index])
                
#                 flip_image(vol_tensor, 'vertical', image_idx, directory)
#                 image_idx += 1
                
#                 if plane == 'axial':
#                     abnormal_labels.append(0)            
    

def augment(case):
    
    path = 'MRNet-v1.0/'
    split = 'train'

    abnormal_labels = []
    acl_labels = []
    meniscus_labels = []

    with open(path + split + '-' + case + '.csv') as f:
        abnormal_labels = pd.read_csv(f, header = None)[1].tolist()
    
    with open(path + split + '-' + 'acl' + '.csv') as f:
        acl_labels = pd.read_csv(f, header = None)[1].tolist()
        
    with open(path + split + '-' + 'meniscus' + '.csv') as f:
        meniscus_labels = pd.read_csv(f, header = None)[1].tolist()
    
    print(np.sum(np.mean(abnormal_labels)))
    print(np.sum(np.mean(acl_labels)))
    print(np.sum(np.mean(meniscus_labels)))
          
    augment_plane(path, split, 'sagittal', abnormal_labels, acl_labels, meniscus_labels)
    augment_plane(path, split, 'axial', abnormal_labels, acl_labels, meniscus_labels)
    augment_plane(path, split, 'coronal', abnormal_labels, acl_labels, meniscus_labels)
   
    print(len(abnormal_labels))
    print(len(acl_labels))
    print(len(meniscus_labels))
    
    print(np.sum(np.mean(abnormal_labels)))
    print(np.sum(np.mean(acl_labels)))
    print(np.sum(np.mean(meniscus_labels)))
    
    out_abnormal_labels = dict(enumerate(abnormal_labels, start=0))
    abnormal_labels_csv = CSV ="\n".join([str(k)+','+','.join(str(v)) for k,v in out_abnormal_labels.items()]) 
    
#     print(abnormal_labels_csv)
    
    # save csv

if __name__ == '__main__':
    augment('abnormal')