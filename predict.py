

from net import CNN_Text
from nets.abnormal import AbnormalDetect
from utils.TextDataset import TextDataset
from torch.utils.data import Dataset,DataLoader

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
def load_model(weight_path):
    print(weight_path)
    model=AbnormalDetect(config)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model

if __name__=="__main__":

    config = {}
    config["bs"] = 64
    config["vocabulary"] = 859
    config["word_embedding_size"] = 300
    config["lstm_hidden_size"] = 512
    config["lstm_num_layers"] = 2
    config["dropout"] = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_list=[]
    save_dir='submits/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    for i in range(5):
        model_list.append(load_model('checkpoint/bs'+str(config["bs"])+'/fold_'+str(i+1)+'_best.pth'))

    path = '/home/van/文档/0-进行中/0-比赛/0-code/0-天池/全球人工智能技术创新大赛【赛道一】/data/track1_round1_testA_20210222.csv'
    testdataset = TextDataset(path,list(range(3000)),'test')
    trainloader = DataLoader(testdataset,
                            batch_size=3000,
                            num_workers=0)

    for batch in trainloader:
        for key in batch:
            batch[key] = batch[key].to(device)
        for i in range(len(model_list)):
            model=model_list[i]
            outputs=model(batch['text'],batch['len'])
            outputs=outputs.sigmoid().detach().cpu().numpy()
            
            if i==0:
                fold_ave=outputs
            else:
                fold_ave+=outputs
                
    repo = []
    for f in fold_ave:
        rep = [str(r/len(model_list)) for r in f]
        rep = ' '.join(rep)
        repo.append(rep)

    save = ''
    for i in range(3000):
        save+=(str(i)+'|,|'+repo[i] + '\n')

    with open(save_dir+'submit'+str(config["bs"])+'.csv','w') as f:
        save=save.strip('\n')
        f.write(save)

