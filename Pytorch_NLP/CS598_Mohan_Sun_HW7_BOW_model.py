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

class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, no_of_hidden_units)
        self.fc_hidden = nn.Linear(no_of_hidden_units, no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t):
        bow_embedding = []
        for i in range(len(x)):
            if is_cuda:
                lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            else:
                lookup_tensor = Variable(torch.LongTensor(x[i]))
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)

        x = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        x = self.fc_output(x)

        return self.loss(x[:,0],t), x[:,0]

class BOW_model_GloVe(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model_GloVe, self).__init__()
        
        self.fc_hidden1 = nn.Linear(300, no_of_hidden_units)
        self.bn_hidden1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc_hidden2 = nn.Linear(no_of_hidden_units, no_of_hidden_units)
        self.bn_hidden2 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t, twice):
        
        x = self.dropout1(F.relu(self.bn_hidden1(self.fc_hidden1(x))))
        if twice:
            x = self.dropout2(F.relu(self.bn_hidden2(self.fc_hidden2(x))))
        x = self.fc_output(x)

        return self.loss(x[:,0],t), x[:,0]
