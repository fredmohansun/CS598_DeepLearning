import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.backends.cudnn as cudnn

## Hyperparas
is_cuda = torch.cuda.is_available()

class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM,self).__init__()
        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size
        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        batch_size = x.size(0)
        if self.h is None:
            state_size = [batch_size, self.out_size]
            if is_cuda:
                self.c = Variable(torch.zeros(state_size)).cuda()
                self.h = Variable(torch.zeros(state_size)).cuda()
            else:
                self.c = Variable(torch.zeros(state_size)).cuda()
                self.h = Variable(torch.zeros(state_size)).cuda()
        self.h, self.c = self.lstm(x,(self.h,self.c))

        return self.h

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if not train:
            return x
        if self.m is None:
            self.m = x.data.new(x.size()).bernoilli_(1-dropout)
        if is_cuda:
            mask = Variable(self.m, requires_grad=False).cuda()/(1-dropout)
        else:
            mask = Variable(self.m, requires_grad=False)/(1-dropout)

        return mask * x

class RNN_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(RNN_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, no_of_hidden_units)#padding_idx=0)

        self.lstm1 = StatefulLSTM(no_of_hidden_units, no_of_hidden_units)
        self.bn_lstm1 = nn.BatchNorm1d(no_of_hidden_Units)
        self.dropout1 = LockedDropout(dropout)#nn.Dropout(p=dropout)

        self.lstm2 = StatefulLSTM(no_of_hidden_units, no_of_hidden_units)
        self.bn_lstm2 = nn.BatchNorm1d(no_of_hidden_Units)
        self.dropout2 = LockedDropout(dropout)#nn.Dropout(p=dropout)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self, twice=False):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        if twice:
            self.lstm2.reset_state()
            self.dropout2.reset_state()

    def forward(self, x, t, twice=False, train=True):
        embed = self.embedding(x) #[batch_size, time_steps, features]

        no_of_timesteps = embed.shape[1]

        self.reset_state(twice)
        
        outputs = []
        for i in range(no_of_timesteps):
            h = self.lstm1(embed[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout(h, 0.5, train)

            if twice:
                h = self.lstm2(h)
                h = self.bn_lstm2(h)
                h = self.dropout2(h, 0.3, train)

            outputs.append(h)

        outputs = torch.stack(outputs) #[time_steps, batch_size, features]
        outputs = outputs.permute(1,2,0) #[batch_size, features, time_steps]

        pool = nn.MaxPool1d(no_of_timesteps)
        h = pool(outputs)
        h = h.view(h.size(0),-1)
        #h = self.dropout(h)

        h = self.fc_output(h)

        return self.loss(h[:,0],t), h[:,0]