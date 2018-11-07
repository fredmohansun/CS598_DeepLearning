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

from CS598_Mohan_Sun_HW7_RNN_model import RNN_model

## parser
parser = argparse.ArgumentParser()
parser.add_argument('--ADAM', action='store_true')
parser.add_argument('--twice', action='store_true')
parser.add_argument('--nnDropout', action='store_true')
parser.add_argument('-n', '--batch_size', type=int, help='Batch size')
parser.add_argument('-H', '--no_of_hidden_units', type=int, help='No of hidden units')
parser.add_argument('-t', '--max_epoch', type=int, help='Max epoch')
parser.add_argument('-V', '--vocab_size', type=int, help='Vocab size')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--TestSL', type=int, help='Test Sequence Length')
parser.add_argument('--TrainSL', type=int, help='Train Sequence Length')
args = parser.parse_args()

## Hyperparas
is_cuda = torch.cuda.is_available()
vocab_size = 8000 if args.vocab_size is None else args.vocab_size
no_of_hidden_units = 500 if args.no_of_hidden_units is None else args.no_of_hidden_units
batch_size = 200 if args.batch_size is None else args.batch_size
max_epoch = 21 if args.max_epoch is None else args.max_epoch 
lr = (0.001 if args.ADAM else 0.01) if args.lr is None else args.lr
train_sl = 100 if args.TrainSL is None else args.TrainSL
test_sl = 200 if args.TestSL is None else args.TestSL
switches = [args.nnDropout,args.twice]
savefile = '_'.join(['2a', 'ADAM' if args.ADAM else 'SGD', str(lr), str(max_epoch), str(batch_size), str(no_of_hidden_units), str(vocab_size), ''.join(list(map(str,map(int,switches)))), str(train_sl),str(test_sl)])

## Data
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
model = torch.load('model/'+savefile + '.model')
if is_cuda:
    model.cuda()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.ADAM:
    optimizer = optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

## Epochs
L_test_label = len(test_label)

with io.open('result/' + savefile +'_test.out','w',encoding = 'utf-8') as f:
    for epoch in range(10):
        model.eval()
        timer1 = time.time()
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0
        I_permutation = np.random.permutation(L_test_label)
        for i in range(0, L_test_label, batch_size):
            inputs2 = [test_inputs[j] for j in I_permutation[i:i+batch_size]]
            sequence_length = (epoch+1)*50
            inputs = np.zeros((batch_size, sequence_length), dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(inputs2[j])
                sl = x.shape[0]
                if(sl < sequence_length):
                    inputs[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    inputs[j,:] = x[start_index:(start_index+sequence_length)]
            label = test_label[I_permutation[i:i+batch_size]]
            if is_cuda:
                inputs = Variable(torch.LongTensor(inputs)).cuda()
                target = Variable(torch.FloatTensor(label)).cuda()
            else:
                inputs = Variable(torch.LongTensor(inputs))
                target = Variable(torch.FloatTensor(label))

            with torch.no_grad():
                loss, prediction = model(inputs, target, train=False)
            prediction = prediction >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data.item()
            epoch_counter += batch_size
            
        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)
        f.write('SL: {0}, test accuracy: {1:.4f}%, test loss {2:.6f},time: {3:.2f} seconds\n'.format(sequence_length, epoch_acc*100.0, epoch_loss,float(time.time()-timer1)))
