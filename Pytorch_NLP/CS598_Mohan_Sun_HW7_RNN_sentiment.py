import numpy as np
import time
import os
import sys
import io
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from CS598_Mohan_Sun_HW7_BOW_model import BOW_model 

## parser
parser = argparse.ArgumentParser()
parser.add_argument('--ADAM', action='store_true')
parser.add_argument('-n', '--batch_size', type=int, help='Batch size')
parser.add_argument('-t', '--max_epoch', type=int, help='Max epoch')
parser.add_argument('-H', '--no_of_hidden_units', type=int, help='No of hidden units')
parser.add_argument('-V', '--vocab_size', type=int, help='Vocab size')
parser.add_argument('--lr', type=float, help='Learning rate')
args = parser.parse_args()

## Hyperparas
is_cuda = torch.cuda.is_available()
vocab_size = 8000 if args.vocab_size is None else args.vocab_size
no_of_hidden_units = 500 if args.no_of_hidden_units is None else args.no_of_hidden_units
batch_size = 200 if args.batch_size is None else args.batch_size
max_epoch = 6 if args.max_epoch is None else args.max_epoc
lr = (0.001 if args.ADAM else 0.01) if args.lr is None else args.lr
savefile = '_'.join(['1a', 'ADAM' if args.ADAM else 'SGD', str(lr),str(max_epoch), str(batch_size), str(no_of_hidden_units), str(vocab_size)]) + '.out'
print(savefile)

## Data
train_inputs = []
with io.open('preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)
    line[line > vocab_size] = 0
    train_inputs.append(line)

train_inputs = train_inputs[0:25000]
train_label = np.zeros((25000,))
train_label[0:12500] = 1

test_inputs = []
with io.open('preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)
    line[line > vocab_size] = 0
    test_inputs.append(line)

test_label = np.zeros((25000,))
test_label[0:12500] = 1

vocab_size += 1

## Model setup
model = BOW_model(vocab_size, no_of_hidden_units)
if is_cuda:
    model.cuda()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.ADAM:
    optimizer = optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

## Epochs
L_train_label = len(train_label)
L_test_label = len(test_label)

train_loss = []
train_acc = []
test_acc = []
with io.open('result/' + savefile,'w',encoding = 'utf-8') as f:
    for epoch in range(max_epoch):
        model.train()
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0

        timer1 = time.time()

        I_permutation = np.random.permutation(L_train_label)

        for i in range(0, L_train_label, batch_size):
            
            inputs = [train_inputs[j] for j in I_permutation[i:i+batch_size]]
            label = np.asarray([train_label[j] for j in I_permutation[i:i+batch_size]], dtype=np.int)
            if is_cuda:
                target = Variable(torch.FloatTensor(label)).cuda()
            else:
                target = Variable(torch.FloatTensor(label))

            optimizer.zero_grad()
            loss, prediction = model(inputs, target)
            loss.backward()

            optimizer.step()

            prediction = prediction >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data.item()
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        timer2 = time.time()
        f.write('Epoch {0}: training accuracy: {1:.4f}%, loss: {2:.6f}, time: {3:.2f} seconds'.format(epoch, epoch_acc*100.0, epoch_loss, float(timer2-timer1)))

        model.eval()

        epoch_acc = 0.0
        epoch_counter = 0

        I_permutation = np.random.permutation(L_test_label)

        for i in range(0, L_test_label, batch_size):
            
            inputs = [test_inputs[j] for j in I_permutation[i:i+batch_size]]
            label = np.asarray([test_label[j] for j in I_permutation[i:i+batch_size]], dtype=np.int)
            if is_cuda:
                target = Variable(torch.FloatTensor(label)).cuda()
            else:
                target = Variable(torch.FloatTensor(label))

            with torch.no_grad():
                loss, prediction = model(inputs, target)

            prediction = prediction >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_counter += batch_size

        epoch_acc /= epoch_counter

        test_acc.append(epoch_acc)

        f.write(', test accuracy {1:.4f}%, time: {2:.2f} seconds\n'.format(epoch, epoch_acc*100.0, float(time.time()-timer2)))
