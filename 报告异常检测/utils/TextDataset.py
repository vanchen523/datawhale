from torch.utils.data import Dataset,DataLoader
import csv
import torch
import numpy as np
class TextDataset(Dataset):
    def __init__(self, path, indexlist, split='train'):
        super().__init__()
        self.split = split
        textlist = []
        lenlist = []
        labellist = []
        with open(path,'r') as f:
            for line in csv.reader(f, delimiter='\n'):
                line = line[0].split('|,|')
                if int(line[0]) in indexlist:
                    text = [int(i)+1 for i in line[1].split()]
                    textlist.append(text)
                    lenlist.append(len(text))
                    if self.split != 'test':labellist.append(line[2])
        self.textlist = textlist
        self.lenlist = lenlist
        
        self.maxlen = max(lenlist)
        wordtype = 0
        for word in textlist:
            wordtype = max(wordtype, max(word))
        self.wordtype = wordtype

        for i in range(len(labellist)):
            onehot =[0]*17
            if labellist[i] != '':
                for a in [int(j) for j in labellist[i].split()]:
                    onehot[a] = 1
            labellist[i] = onehot
        self.labellist = labellist
            
    def __len__(self):
        return len(self.textlist)
    
    def __getitem__(self, index):
        item = {}
        text = self.textlist[index]
        textlen = self.lenlist[index]
        if len(text) != self.maxlen:
            text = text + [0]*(self.maxlen-len(text))
        item['text'] = torch.tensor(text).long()
        item['len'] = torch.tensor(textlen).long()
        if self.split != 'test':   
            label = self.labellist[index]
            item['label'] = torch.tensor(label).float()
        return item
#%%
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

class DynamicRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.

        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            Single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True
        )

        if initial_state is not None:
            hx = initial_state
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None

        self.rnn_model.flatten_parameters()
        outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)

        # pick hidden and cell states of last layer
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        c_n = c_n[-1].index_select(dim=0, index=bwd_order)

        outputs = pad_packed_sequence(
            outputs, batch_first=True, total_length=max_sequence_length
        )
        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(
            lens.contiguous().view(-1), 0, descending=True
        )
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order
    
    
class AbnormalDetect(nn.Module):
    def __init__(self, config, static=False):
        super(AbnormalDetect, self).__init__()
        class_num = 17
        Ci = 1
        kernel_num = 100
        Ks = [3,4,5]

        self.embed = nn.Embedding(
            config["vocabulary"],
            config["word_embedding_size"],
            padding_idx=0,
        )
        self.rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.rnn = DynamicRNN(self.rnn)

        self.convs = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, config["word_embedding_size"])) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(config["lstm_hidden_size"], class_num)
        self.fc2 = nn.Linear(len(Ks) * kernel_num, class_num)
        self.fc2 = nn.Linear(2 * class_num, class_num)

        if static:
            self.embed.weight.requires_grad = False

    def forward(self, x, xlen):
        x = self.embed(x)  # (N, W, D)-batch,单词数量，维度
        _, (x1, _) = self.rnn(x, xlen)
        x1 = self.dropout(x1)  # (N, len(Ks)*Co)        

        x = x.unsqueeze(1)
        x2 = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x2]
        print(x1,'\n',x2)
        x2 = torch.cat(x2, 1)
        x2 = self.dropout(x2)     
        x1 = self.fc1(x1)  # (N, C)
        x2 = self.fc2(x2)  # (N, C)
        x = self.fc3(torch.cat(x1,x2))
        # print(torch.sigmoid(x))
        return x
    
if __name__=="__main__":
    config = {}
    config["vocabulary"] = 859
    config["word_embedding_size"] = 300
    config["lstm_hidden_size"] = 128
    config["lstm_num_layers"] = 2
    config["dropout"] = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net=AbnormalDetect(config).to(device)
    path = '/home/van/文档/0-进行中/0-比赛/0-code/0-天池/全球人工智能技术创新大赛【赛道一】/data/track1_round1_train_20210222.csv'
    path = '/home/van/文档/0-进行中/0-比赛/0-code/0-天池/全球人工智能技术创新大赛【赛道一】/data/track1_round1_testA_20210222.csv'
    traindataset = TextDataset(path,list(range(3000)),'test')
    trainloader = DataLoader(traindataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0)
    for iteration, batch in enumerate(trainloader):
        a = [iteration, batch]
        for key in batch:
            batch[key] = batch[key].to(device)
        # print(batch['text'])
        # text = batch['text'].to(device)
        # print(batch['label'])
        # label = batch['label'].to(device)
        pass
    output = net(batch['text'], batch['len'])
    # x = torch.LongTensor([[1,2,4,5,2,35,43,113,111,451,455,22,45,55],
    #                       [14,3,12,9,13,4,51,45,53,17,57,121,156,23]])
    # net(x)
