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
        self.fc = nn.Linear(config["lstm_hidden_size"] + len(Ks) * kernel_num, class_num)
        # self.fc1 = nn.Linear(config["lstm_hidden_size"], class_num)
        # self.fc2 = nn.Linear(len(Ks) * kernel_num, class_num)
        # self.fc3 = nn.Linear(class_num*2, class_num)

        if static:
            self.embed.weight.requires_grad = False

    def forward(self, x, xlen):
        x = self.embed(x)  # (N, W, D)-batch,单词数量，维度
        _, (x1, _) = self.rnn(x, xlen)
        # x1 = self.fc1(x1)
        x = x.unsqueeze(1)
        x2 = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x2]
        x2 = torch.cat(x2, 1)
        # x2 = self.fc2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)        
        x = self.fc(x)  # (N, C)
        # print(torch.sigmoid(x))
        return x
    
if __name__=="__main__":
    config = {}
    config["vocabulary"] = 858
    config["word_embedding_size"] = 300
    config["lstm_hidden_size"] = 512
    config["lstm_num_layers"] = 2
    config["dropout"] = 0.5

    net=AbnormalDetect(config)
    x=torch.LongTensor([[1,2,4,5,2,35,43,113,111,451,455,22,45,55],
                        [14,3,12,9,13,4,51,45,53,17,57,151,156,23]])
    logit=net(x,torch.LongTensor(104))
