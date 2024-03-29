import argparse
import os
import numpy as np
import torch
import pickle
import sys

from sklearn import metrics
from torch.autograd import Variable

from loader import load_data
from model import CNN, Combine
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--gpu', action='store_true')

    return parser


def run_model(model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0

    for batch in tqdm(loader):
        vol_axial, vol_coronal, vol_sagittal, label = batch

        if train:
            optimizer.zero_grad()

        if loader.dataset.use_gpu:
            vol_axial, vol_coronal, vol_sagittal = vol_axial.cuda(), vol_coronal.cuda(), vol_sagittal.cuda()
            label = label.cuda()
        vol_axial, vol_coronal, vol_sagittal = Variable(vol_axial), Variable(vol_coronal), Variable(vol_sagittal)
        label = Variable(label)

        logit = model.forward(batch)

        loss = loader.dataset.weighted_loss(logit, label)  # weighted
        total_loss += loss.data.cpu().numpy()

        pred = torch.sigmoid(logit)

        pred_npy = pred.data.cpu().numpy()[0]  # .item()
        label_npy = label.data.cpu().numpy()[0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    labels_npy = np.asarray(labels)
    preds_npy = np.asarray(preds)

    accuracy = None
    #     accuracy = [np.mean((preds_npy[:,0] > 0.5) == (labels_npy[:,0] == 1))]
    #     accuracy.append(np.mean((preds_npy[:,1] > 0.5) == (labels_npy[:,1] == 1)))

    fpr, tpr, threshold = metrics.roc_curve(labels_npy[:, 0], preds_npy[:, 0])
    auc = [metrics.auc(fpr, tpr)]

    fpr, tpr, threshold = metrics.roc_curve(labels_npy[:, 1], preds_npy[:, 1])
    auc.append(metrics.auc(fpr, tpr))

    #     fpr, tpr, threshold = metrics.roc_curve(labels_npy[:,2], preds_npy[:,2])
    #     auc.append(metrics.auc(fpr, tpr))

    #     fpr, tpr, threshold = metrics.roc_curve(labels_npy, preds_npy)
    #     auc = [metrics.auc(fpr, tpr)]

    return avg_loss, auc, accuracy, preds_npy, labels_npy


def evaluate(split, model_dir, use_gpu=True):
    model = Combine()
    if use_gpu:
        model = model.cuda()
    state_dict = torch.load('/home/Mara/run_baseline_acl_meniscus_gap/val0.3271_train0.2068_epoch22',
                            map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    train_loader, valid_loader, test_loader = load_data(model_dir, use_gpu)
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
    print(f'{split} AUC_acl: {auc[0]:0.4f}')
    print(f'{split} AUC_meniscus: {auc[1]:0.4f}')

    #     print(f'{split} AUC_abnormal: {auc[0]:0.4f}')

    return preds, labels, model, loader


# def evaluate(paths, task, use_gpu):

#     model = MRNet()

#     np.random.seed(42)

#     all_preds = []

#     for view in range(3):
#         view_list = ['sagittal', 'coronal', 'axial']
#         loader = load_data(paths, task, view_list[view], shuffle=False, use_gpu=use_gpu)

#         state_dict = torch.load('src/models/'+view_list[view]+'-'+task, map_location=(None if use_gpu else 'cpu'))
#         model.load_state_dict(state_dict)

#         if use_gpu:
#             model = model.cuda()

#         loss, auc, preds, labels = run_model(model, loader)

#         all_preds.append(preds)

#         # print(f'{split} loss: {loss:0.4f}')
#         # print(f'{split} AUC: {auc:0.4f}')

#     preds = np.stack(all_preds, axis=1)
#     return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu)