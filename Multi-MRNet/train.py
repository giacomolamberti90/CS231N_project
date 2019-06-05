import argparse
import json
import numpy as np
import os
import torch

from datetime import datetime
from pathlib import Path
from sklearn import metrics

from evaluate import run_model
from loader import load_data
from model import MRNet, TripleMRNet

import matplotlib.pyplot as plt

def train(rundir, path, epochs, learning_rate, use_gpu):
    
    rundir = rundir + '/'
    
    train_loader, valid_loader, test_loader = load_data(path, use_gpu)
    
    model = TripleMRNet() #1, 32, 64
    
    if use_gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-5)

    best_val_loss = float('inf')

    start_time = datetime.now()
    
    for epoch in range(epochs):
                
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}\n'.format(epoch+1, str(change)))
        
        train_loss, train_auc, train_accuracy, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC_abnormal: {train_auc[0]:0.4f}')
        print(f'train AUC_acl: {train_auc[1]:0.4f}')
        print(f'train AUC_meniscus: {train_auc[2]:0.4f}\n')
        #print(f'train accuracy_abnormal: {train_accuracy[0]:0.4f}')
        #print(f'train accuracy_acl: {train_accuracy[1]:0.4f}')
        #print(f'train accuracy_meniscus: {train_accuracy[2]:0.4f}\n')              
        
        val_loss, val_auc, val_accuracy, _, _ = run_model(model, valid_loader)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC_abnormal: {val_auc[0]:0.4f}')
        print(f'valid AUC_acl: {val_auc[1]:0.4f}')
        print(f'valid AUC_meniscus: {val_auc[2]:0.4f}\n')
        #print(f'valid accuracy_abnormal: {val_accuracy[0]:0.4f}')
        #print(f'valid accuracy_acl: {val_accuracy[1]:0.4f}')
        #print(f'valid accuracy_meniscus: {val_accuracy[2]:0.4f}\n')
              
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            file_name = f'val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name
            torch.save(model.state_dict(), save_path)                

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)

    return parser

if __name__ == '__main__':
        
    np.random.seed(42)
    torch.manual_seed(42)

    train('models', path='MRNet-v1.0/', epochs=20, learning_rate=1e-5, use_gpu=True)

    '''
    args = get_parser().parse_args()
   
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.rundir, exist_ok=True)
    
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train(args.rundir, args.diagnosis, args.epochs, args.learning_rate, args.gpu)
    '''
