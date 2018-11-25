import argparse
import numpy as np
import os
import sys
import time
import h5py
import cv2
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torchvision
import torch.backends.cudnn as cudnn

from helper import getUCF101
from helper import loadFrame

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--no_of_layer',type=int,choices = range(1,8))
args = parser.parse_args()

# Hyperparas
is_cuda = torch.cuda.is_available()
IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
max_epoch = 10
no_of_layer = 2 if args.no_of_layer is None else args.no_of_layer
data_dir = '/projects/training/bauh/AR/'

# Data
class_list, train, test = getUCF101()

# Model set up
model =  torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,NUM_CLASSES)

for param in model.parameters():
    param.requires_grad_(False)

params = []
if no_of_layer >= 7:
    for param in model.conv1.parameters():
        param.requires_grad_(True)
        params.append(param)
elif no_of_layer >= 6:
    for param in model.bn1.parameters():
        param.requires_grad_(True)
        params.append(param)
elif no_of_layer >= 5:
    for param in model.layer1.parameters():
        param.requires_grad_(True)
        params.append(param)
elif no_of_layer >= 4:
    for param in model.layer2.parameters():
        param.requires_grad_(True)
        params.append(param)
elif no_of_layer >= 3:
    for param in model.layer3.parameters():
        param.requires_grad_(True)
        params.append(param)
elif no_of_layer >= 2:
    for param in model.layer4[2].parameters():
        param.requires_grad_(True)
        params.append(param)
else:
    for param in model.fc.parameters():
        param.requires_grad_(True)
        params.append(param)

if is_cuda:
    model.cuda()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

optimizer = optim.Adam(params, lr=lr)
criterion = nn.CrossEntropyLoss()

pool_threads = Pool(8, maxtasksperchild=200)

for epoch in range(max_epoch):
    model.train()
    running_acc = 0.0
    running_loss = 0.0
    counter = 0

    timer1 = time.time()
    I_permutation = np.random.permutation(len(train[0]))
    augment = True
    for i in range(0, len(train[0])-batch_size,batch_size):
        video_list = [(train[0][k],augment) for k in I_permutation[i:(batch_size+i)]]
        data = pool_threads.map(loadFrame, video_list)

        next_batch = 0
        for video in data:
            if video.size==0:
                next_batch = 1
                break
        if next_batch:
            continue

        inputs = np.asarray(data, dtype=np.float32)
        label = train[1][I_permutation[i:(batch_size+i)]]
        if is_cuda:
            inputs = Variable(torch.FloatTensor(inputs)).cuda().contiguous()
            label = torch.from_numpy(label).cuda()
        else:
            inputs = Variable(torch.FloatTensor(inputs)).contiguous()
            label = torch.from_numpy(label)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        prediction = output.data.max(1)[1]
        running_acc += prediction.eq(label.data).sum().cpu().data.numpy()
        running_loss += loss.data.item()
        counter += batch_size
    timer2 = time.time()
    print('Epoch {0}: train accuracy: {1:.4f}%, loss: {2:.6f}, time: {3:.2f} seconds'.format(epoch,running_acc/counter*100.0, running_loss/(counter/batch_size), timer2-timer1))
    pool_threads.close()
    pool_threads.terminate()

    model.eval()
    running_acc = 0.0
    counter = 0

    I_permutation = np.random.permutation(len(test[0]))
    augment = False
    for i in range(0, len(test[0])-batch_size,batch_size):
        video_list = [(test[0][k],augment) for k in I_permutation[i:(batch_size+i)]]
        data = pool_threads.map(loadFrame, video_list)

        next_batch = 0
        for video in data:
            if video.size==0:
                next_batch = 1
                break
        if next_batch:
            continue

        inputs = np.asarray(data, dtype=np.float32)
        label = test[1][I_permutation[i:(batch_size+i)]]
        if is_cuda:
            inputs = Variable(torch.FloatTensor(inputs)).cuda().contiguous()
            label = torch.from_numpy(label).cuda()
        else:
            inputs = Variable(torch.FloatTensor(inputs)).contiguous()
            label = torch.from_numpy(label)

        with torch.no_grad():
            output = model(inputs)

        prediction = output.data.max(1)[1]
        running_acc += prediction.eq(label.data).sum().cpu().data.numpy()
        counter += batch_size
    print('Epoch {0}: test accuracy: {1:.4f}%, time: {2:.2f} seconds'.format(epoch,running_acc/counter*100.0, time.time()-timer2))
    pool_threads.close()
    pool_threads.terminate()

torch.save(model.state_dict(),'model/single_frame.model')
