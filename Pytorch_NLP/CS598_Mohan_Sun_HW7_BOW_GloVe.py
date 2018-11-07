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

from CS598_Mohan_Sun_HW7_BOW_model import BOW_model_GloVe

## parser
parser = argparse.ArgumentParser()
parser.add_argument('--ADAM', action='store_true')
parser.add_argument('-n', '--batch_size', type=int, help='Batch size')
parser.add_argument('-t', '--max_epoch', type=int, help='Max epoch')
parser.add_argument('-H', '--no_of_hidden_units', type=int, help='No of hidden units')
parser.add_argument('--twice', action='store_true')
parser.add_argument('--lr', type=float, help='Learning rate')
args = parser.parse_args()

## Hyperparas
is_cuda = torch.cuda.is_available()
no_of_hidden_units = 500 if args.no_of_hidden_units is None else args.no_of_hidden_units
batch_size = 200 if args.batch_size is None else args.batch_size
max_epoch = 6 if args.max_epoch is None else args.max_epoch
lr = (0.001 if args.ADAM else 0.01) if args.lr is None else args.lr
glove_embeddings = np.load('preprocessed_data/glove_embeddings.npy')
vocab_size = 100000
savefile = '_'.join(['1b', 'ADAM' if args.ADAM else 'SGD', str(lr),str(max_epoch), str(batch_size), str(no_of_hidden_units), str(vocab_size), 'twice' if args.twice else 'once']) + '.out'
print(savefile)

## Hyperparas
train_inputs = []
with io.open('preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)
    line[line > vocab_size] = 0
    line = line[line!=0]
    line = np.mean(glove_embeddings[line], axis=0)
    train_inputs.append(line)

train_inputs = np.asarray(train_inputs)
train_inputs = train_inputs[0:25000]
train_label = np.zeros((25000,))
train_label[0:12500] = 1

test_inputs = []
with io.open('preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)
    line[line > vocab_size] = 0
    line = line[line!=0]
    line = np.mean(glove_embeddings[line], axis=0)
    test_inputs.append(line)

test_inputs = np.asarray(test_inputs)
test_label = np.zeros((25000,))
test_label[0:12500] = 1

vocab_size += 1

model = BOW_model_GloVe(vocab_size, no_of_hidden_units)
if is_cuda:
    model.cuda()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.ADAM:
    optimizer = optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

L_train_label = len(train_label)
L_test_label = len(test_label)

train_loss = []
train_acc = []
test_acc = []

with io.open('result/' + savefile,'w', encoding='utf-8') as f:
    for epoch in range(max_epoch):
        model.train()
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0

        timer1 = time.time()

        I_permutation = np.random.permutation(L_train_label)

        for i in range(0, L_train_label, batch_size):
            
            inputs = train_inputs[I_permutation[i:i+batch_size]]
            label = train_label[I_permutation[i:i+batch_size]]
            if is_cuda:
                inputs = Variable(torch.FloatTensor(inputs)).cuda()
                target = Variable(torch.FloatTensor(label)).cuda()
            else:
                inputs = Variable(torch.FloatTensor(inputs))
                target = Variable(torch.FloatTensor(label))

            optimizer.zero_grad()
            loss, prediction = model(inputs, target, args.twice)
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
            
            inputs = test_inputs[I_permutation[i:i+batch_size]]
            label = test_label[I_permutation[i:i+batch_size]]
            if is_cuda:
                inputs = Variable(torch.FloatTensor(inputs)).cuda()
                target = Variable(torch.FloatTensor(label)).cuda()
            else:
                inputs = Variable(torch.FloatTensor(inputs))
                target = Variable(torch.FloatTensor(label))

            with torch.no_grad():
                loss, prediction = model(inputs, target, args.twice)

            prediction = prediction >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_counter += batch_size

        epoch_acc /= epoch_counter

        test_acc.append(epoch_acc)

        f.write(', Epoch {0}: test accuracy {1:.4f}%, time: {2:.2f} seconds\n'.format(epoch, epoch_acc*100.0, float(time.time()-timer2)))
