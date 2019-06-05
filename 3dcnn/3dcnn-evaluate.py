import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from sklearn import metrics
from torch.autograd import Variable

from loader import load_data
from model import MRNet, TripleMRNet
from tqdm import tqdm

def get_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--gpu', action='store_true')
    
    return parser

def plot_roc(fpr_abnormal, tpr_abnormal, fpr_acl, tpr_acl, fpr_meniscus, tpr_meniscus, auc_abnormal, auc_acl, auc_meniscus):
    plt.figure()
    lw = 2
    plt.plot(fpr_abnormal, tpr_abnormal, color='darkorange',
         lw=lw, label='Abnormal Task ROC curve (area = %0.2f)' % auc_abnormal)
    plt.plot(fpr_acl, tpr_acl, color='green',
             lw=lw, label='ACL Task ROC curve (area = %0.2f)' % auc_acl)
    plt.plot(fpr_meniscus, tpr_meniscus, color='blue',
         lw=lw, label='Meniscus Task ROC curve (area = %0.2f)' % auc_meniscus)
        
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('3D CNN: Receiver operating characteristic', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig("roc_2d3d_cnn.png")

def run_model(model, loader, train=False, optimizer=None):
    
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0
    
    losses = {}
    best = None
    worst = None
    
    min_loss = 100
    max_loss = 0

    for batch in tqdm(loader):
        
        vol_axial, vol_coronal, vol_sagittal, label = batch
        
        if train:
            optimizer.zero_grad()

        if loader.dataset.use_gpu:
            vol_axial, vol_coronal, vol_sagittal = vol_axial.cuda(), vol_coronal.cuda(), vol_sagittal.cuda()
            label = label.cuda()
        vol_axial, vol_coronal, vol_sagittal = Variable(vol_axial), Variable(vol_coronal), Variable(vol_sagittal)
        label = Variable(label)

        logit = model.forward(vol_axial, vol_coronal, vol_sagittal)

#         loss = loader.dataset.standard_loss(logit, label)
        loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()
    
    
# #         losses[loss] = label.cpu()
# #         print(sorted(losses.items()))
# #         print(label[0])
        if loss < min_loss:
            min_loss = loss
            best = label[0]
#             print("best: ", best)
        elif loss > max_loss:
            max_loss = loss
            worst = label[0]
#             print("worst: ", worst)
        
        pred = torch.sigmoid(logit)

        pred_npy = pred.data.cpu().numpy()[0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches
    
    labels_npy = np.asarray(labels)
    preds_npy = np.asarray(preds)
        
    accuracy = [np.mean((preds_npy[:,0] > 0.5) == (labels_npy[:,0] == 1))]
    accuracy.append(np.mean((preds_npy[:,1] > 0.5) == (labels_npy[:,1] == 1)))
    accuracy.append(np.mean((preds_npy[:,2] > 0.5) == (labels_npy[:,2] == 1)))
    
    fpr, tpr, threshold = metrics.roc_curve(labels_npy[:,0], preds_npy[:,0])
    auc = [metrics.auc(fpr, tpr)]
    fpr_abnormal = fpr
    tpr_abnormal = tpr
    auc_abnormal = metrics.auc(fpr, tpr)
    print("recall abnormal: ", metrics.recall_score(labels_npy[:,0], np.where(preds_npy[:,0] > 0.5, 1, 0)))
    print("accuracy abnormal: ", metrics.accuracy_score(labels_npy[:,0], np.where(preds_npy[:,0] > 0.5, 1, 0)))
    print("precision abnormal: ", metrics.precision_score(labels_npy[:,0], np.where(preds_npy[:,0] > 0.5, 1, 0)))
    
#     print("sum preds: ", np.sum(np.where(preds_npy[:,0] > 0.5, 1, 0)))
#     print("sum actual: ", np.sum(labels_npy[:,0]))
    
    fpr, tpr, threshold = metrics.roc_curve(labels_npy[:,1], preds_npy[:,1])
    auc.append(metrics.auc(fpr, tpr))
    fpr_acl = fpr
    tpr_acl = tpr
    auc_acl = metrics.auc(fpr, tpr)
    print("recall acl: ", metrics.recall_score(labels_npy[:,1], np.where(preds_npy[:,1] > 0.5, 1, 0)))
    print("accuracy acl: ", metrics.accuracy_score(labels_npy[:,1], np.where(preds_npy[:,1] > 0.5, 1, 0)))
    print("precision acl: ", metrics.precision_score(labels_npy[:,1], np.where(preds_npy[:,1] > 0.5, 1, 0)))
#     print("sum preds: ", np.sum(np.where(preds_npy[:,1] > 0.5, 1, 0)))
#     print("sum actual: ", np.sum(labels_npy[:,1]))
  
    fpr, tpr, threshold = metrics.roc_curve(labels_npy[:,2], preds_npy[:,2])
    auc.append(metrics.auc(fpr, tpr))
    fpr_meniscus = fpr
    tpr_meniscus = tpr
    auc_meniscus = metrics.auc(fpr, tpr)
    print("recall meniscus: ", metrics.recall_score(labels_npy[:,2], np.where(preds_npy[:,2] > 0.5, 1, 0)))
    print("accuracy meniscus: ", metrics.accuracy_score(labels_npy[:,2], np.where(preds_npy[:,2] > 0.5, 1, 0)))
    print("precision meniscus: ", metrics.precision_score(labels_npy[:,2], np.where(preds_npy[:,2] > 0.5, 1, 0)))
    
#     print("sum preds: ", np.sum(np.where(preds_npy[:,1] > 0.5, 1, 0)))
#     print("sum actual: ", np.sum(labels_npy[:,1]))
    
    
#     print("best: ", best)
#     print("worst: ", worst)
    
#     plot_roc(fpr_abnormal, tpr_abnormal, fpr_acl, tpr_acl, fpr_meniscus, tpr_meniscus, auc_abnormal, auc_acl, auc_meniscus)

    return avg_loss, auc, accuracy, preds, labels

def evaluate(path, split, model_path, use_gpu):
    
    train_loader, valid_loader, test_loader = load_data(path, use_gpu)

    model = TripleMRNet()
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    loss, auc, accuracy, preds, labels = run_model(model, loader)
        
    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC_abnormal: {auc[0]:0.4f}')
    print(f'{split} AUC_acl: {auc[1]:0.4f}')
    print(f'{split} AUC_meniscus: {auc[2]:0.4f}')

    return preds, labels

if __name__ == '__main__':
    
    #args = get_parser().parse_args()
    #evaluate(args.split, args.model_path, args.diagnosis, args.gpu)
    evaluate(path='MRNet-v1.0/', split='test', model_path='models/val0.1988_train0.2127_epoch12', use_gpu=True)