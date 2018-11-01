import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

## Hyperparas
batch_size = 128
max_epoch = 100
learning_rate = 0.0001

## Data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

## Model
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 196, 3, 1, 1),
            nn.LayerNorm([196,32,32]),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm([196,16,16]),
            nn.LeakyReLU(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm([196,16,16]),
            nn.LeakyReLU(0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm([196,8,8]),
            nn.LeakyReLU(0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm([196,8,8]),
            nn.LeakyReLU(0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm([196,8,8]),
            nn.LeakyReLU(0.1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm([196,8,8]),
            nn.LeakyReLU(0.1)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm([196,4,4]),
            nn.LeakyReLU(0.1)
        )

        self.pool = nn.MaxPool2d(4,4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        return (self.fc1(x), self.fc10(x))

## Optimizer and Cuda
aD = discriminator()

is_cuda = torch.cuda.is_available()
if is_cuda:
    aD.cuda()
    aD = nn.DataParallel(aD, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(aD.parameters(), lr = learning_rate)

for epoch in range(max_epoch):
    aD.train()
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    running_loss = []
    acc1 = []
    for i, data in enumerate(trainloader):
        inputs, label = data
        if inputs.shape[0] < batch_size:
            continue

        if is_cuda:
            inputs, label = Variable(inputs).cuda(), Variable(label).cuda()
        else:
            inputs, label = Variable(inputs), Variable(label)

        _, output = aD(inputs)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        prediction = output.data.max(1)[1]
        train_acc = (float(prediction.eq(label.data).sum())/float(batch_size))*100.0
        running_loss.append(loss.item())
        acc1.append(train_acc)
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
    print('Epoch {0}: running_loss: {1:.6f}, train accuracy: {2:.4f}%'.format(epoch, np.mean(running_loss),np.mean(acc1)))

    aD.eval()
    acc2 = []
    for i, data in enumerate(testloader):
        inputs, label = data
        if is_cuda:
            inputs, label = Variable(inputs).cuda(), Variable(label).cuda()
        else:
            inputs, label = Variable(inputs), Variable(label)

        with torch.no_grad():
            _, output = aD(inputs)
        prediction = output.data.max(1)[1]
        test_acc = (float(prediction.eq(label.data).sum())/float(batch_size))*100.0
        acc2.append(test_acc)
    print('Epoch {0}: test accuracy: {1:.4f}%'.format(epoch, np.mean(acc2)))
torch.save(aD, 'model/D.model')
