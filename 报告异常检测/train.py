import os
import time
import yaml
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from bisect import bisect
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from nets.abnormal import AbnormalDetect
from utils.TextDataset import TextDataset


if __name__ == "__main__":
    config = yaml.load(open('config.yml'))
    model_save_dir = 'checkpoint/bs' +  str(config["solver"]["batch_size"])
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    datapath = '/home/van/文档/0-进行中/0-比赛/0-code/0-天池/全球人工智能技术创新大赛【赛道一】/data/track1_round1_train_20210222.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('batch size',config["solver"]["batch_size"],'\nnum_epochs',config["solver"]["num_epochs"])
    start_epoch = 0
    iterations = 10000 // config["solver"]["batch_size"] + 1
    def lr_lambda_fun(current_iteration: int) -> float:
        current_epoch = float(current_iteration) / iterations
        if current_epoch <= config["solver"]["warmup_epochs"]:
            alpha = current_epoch / float(config["solver"]["warmup_epochs"])
            return config["solver"]["warmup_factor"] * (1.0 - alpha) + alpha
        else:
            idx = bisect(config["solver"]["lr_milestones"], current_epoch)
            return pow(config["solver"]["lr_gamma"], idx)

    criterion = torch.nn.BCEWithLogitsLoss()    
    #------------------------------------------------------#11
    #   kfold 交叉验证    
    #------------------------------------------------------#
    a = list(range(10000))
    valindex = []
    trainindex = []
    for i in range(0, len(a), 2000):
        valindex.append(a[i:i + 2000])
        trainindex.append(a[:i]+a[i+2000:])
                         
    for kfold in range(5):
        model = AbnormalDetect(config['model'])
        model.to(device)

        optimizer = optim.Adamax(model.parameters(), lr=config["solver"]["initial_lr"])
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

        best_loss, best_epoch = 1e7, 0
        fold_iteration_step = start_epoch * iterations
        print('\ntrain fold {}'.format(kfold+1))
        traindataset = TextDataset(datapath,trainindex[kfold])
        trainloader = DataLoader(traindataset,
                                 batch_size=config["solver"]["batch_size"],
                                 shuffle=True,
                                 num_workers=0)
        valdataset = TextDataset(datapath,valindex[kfold])
        valloader = DataLoader(valdataset,
                               # batch_size=config["solver"]["batch_size"],
                               batch_size=2000,
                               num_workers=0)
    
        for epoch in range(config["solver"]['num_epochs']):
            # print(f"\nTraining for epoch {epoch+1}:")
            with tqdm(trainloader) as pbar:
                for iteration, batch in enumerate(trainloader):
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    target = batch['label']
                    output = model(batch['text'],batch['len'])
                    
                    losses, num_pos_all = [], 0
                    batch_loss = criterion(output, target)
                    
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    
                    # scheduler.step(fold_iteration_step)
                    fold_iteration_step += 1
                    torch.cuda.empty_cache()

                    pbar.set_postfix(**{'epoch':epoch, 'loss': batch_loss.item()})
                    pbar.update(1)
                
            model.eval()
            # print(f"Validation after epoch {epoch+1}:")
            for i, batch in enumerate(valloader):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    output = model(batch['text'],batch['len'])
                target = batch['label']
                output = model(batch['text'],batch['len'])
            val_loss = criterion(output, target)
            val_auc = metrics.roc_auc_score(target.detach().cpu().numpy(), output.detach().cpu().numpy(), multi_class='ovo')
            
            model_out_path = model_save_dir+"/"+'fold_'+str(kfold+1)+'_'+str(epoch) + '.pth'
            best_model_out_path = model_save_dir+"/"+'fold_'+str(kfold+1)+'_best'+'.pth'
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch=epoch
                torch.save(model.state_dict(), best_model_out_path)
                print("save best epoch: {} best auc: {:.4f} best logloss: {:.4f}".format(best_epoch + 1,val_auc,val_loss))
            model.train()
            torch.cuda.empty_cache()
