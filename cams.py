import os
# import evaluate.predict as predict
from evaluate import evaluate
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
# from testset import util
from loader import load_data

# names = {0: 'abnormal', 1: 'acl', 2: 'meniscus'}

# labels = #GROUND-TRUTH LABELS FOR IMAGES...util.get_labels(all_files)

# model_predictions = #MODEL PREDICTIONS FOR IMAGES (OR ELSE JUST MAKE SURE MODEL CLASSIFIED IMAGES CORRECTLY) #np.load('testset/predictions.npy')

def save_cams(model, cams, use_gpu = False):
    
    def get_cam(cam_samp, weighted=None):
        if weighted:
            dot_product = weight_softmax.dot(cam_samp.reshape(cam_samp.shape[0], -1))
            class_cam = dot_product.reshape((cam_samp.shape[1], cam_samp.shape[2]))
        else:
            class_cam = np.mean(cam_samp, axis=0)
        return class_cam

    #sagittal weights from fully-connected layer
    weight_softmax = np.squeeze(list(model.parameters())[-2][0, -256:].data.cpu().numpy()) #where is this used: in get_cam, if weighted
    #vol_sagittal (set of all slices from single exam) np.load(os.path.join(self.path, "axial", image))
    unnormalized_images = np.load(os.path.join('MRNet-v1.0/valid/', "sagittal", "1130.npy"))
    num_slices, _, _ = unnormalized_images.shape
    unnormalized_images = unnormalized_images
#     unnormalized_images = loader.dataset.unnormalize(batch[0][view][0].cpu().data.numpy()) #get original images to display from dataset

    cams_weighted = []
    #iterate over images in single exam
    for i in range(len(cams)):
        weighted = True
        cam_samp = cams[i]
        class_cam = get_cam(cam_samp, weighted=weighted)
        if class_cam is None: continue

        cams_weighted.append(class_cam)
    #(number of images, number of weights)
    cams_weighted = np.array(cams_weighted)
    print(cams_weighted.min(), cams_weighted.max())
    cams_weighted = cams_weighted - np.min(cams_weighted)
    cams_weighted = (cams_weighted / np.max(cams_weighted) * 255).astype('int')
    print(cams_weighted.min(), cams_weighted.max())
    cmap = plt.get_cmap('magma') #HELPED
    cams_weighted = cmap(cams_weighted)[:, :, :, :3] #:3

    #fig = plt.figure(figsize=(40, 40))

    #iterate over images
    for i in range(cams_weighted.shape[0]):
        class_cam = cams_weighted[i] #[i, ...]
        sample_image = unnormalized_images[i]
#         sample_image = np.swapaxes(sample_image, 0, 2)
#         sample_image = np.swapaxes(sample_image, 1, 0)
        sample_image = sample_image / sample_image.max()
        
        class_cam = cv2.resize(class_cam.astype('float32'), (256, 256)) #upsampling to 256x256 pixels so that CAM is same size as image
        sample_image = np.stack((sample_image,)*3, axis=2)
        merge = 0.3*sample_image + 0.7*class_cam 
    #     merge_normalized = merge / merge.max()
        
        filename = './CAMs_meniscus_wrong'
        os.makedirs(filename, exist_ok=True)
        filename += '/%02d.png' % i
        plt.imsave(filename,merge)
    print(filename)