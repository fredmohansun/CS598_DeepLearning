import numpy as np
#import h5py
import math
#from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

## HyperParameters
batch_size = 100
max_epoch = 30

## Datasets
train_set = torchvision.datasets.CIFAR100(root='../data',train=True,download=True,
            transform=transforms.Compose([transforms.Resize(size=(224,224)),
					  transforms.RandomHorizontalFlip(),
					  transforms.RandomCrop(224,padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.507,0.487,0.441),std=(0.267,0.256,0.276))
                                         ]))

test_set = torchvision.datasets.CIFAR100(root='../data',train=False,download=True,
            transform=transforms.Compose([transforms.Resize(size=(224,224)),
					  transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.507,0.487,0.552),std=(0.267,0.256,0.276))
                                         ]))

#DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
trainloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=8)

testloader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=8)

print('Data is ready...')
## Network Structure

net = torchvision.models.resnet18(pretrained=True)
net = torch.nn.Sequential(net, torch.nn.Linear(net.fc.out_features,100))

is_cuda = torch.cuda.is_available()

if is_cuda:
    net.cuda()
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    print('Cuda is set up...')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr = 0.001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

print('Start Training...')
## Training
for epoch in range(max_epoch):
    net.train()
    #progress = tqdm(trainloader)
    #progress.mininterval = 1
    running_loss, correct, total = 0,0,0
    for i, data in enumerate(trainloader, 0):
        inputs, label = data
        if is_cuda:
            inputs, label = Variable(inputs.cuda()), Variable(label.cuda())
        else:
            inputs, label = Variable(inputs), Variable(label)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        if(epoch > 5):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step'] >= 1024):
                        state['step'] = 1000
        optimizer.step()
        running_loss = running_loss * (i/(i+1)) + loss.item()/(i+1)
        _, prediction = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (prediction==label.data).sum().item()
    print('Epoch:{0}, Running Loss: {1:.6f}, Accuracy: {2:.6f}%'.format(epoch,running_loss,correct/total*100))

    #Glimpse test acc
    net.eval()
    correct, total = 0,0
    for data in testloader:
        inputs, label = data
        if is_cuda:
            inputs, label = Variable(inputs.cuda()), Variable(label.cuda())
        else:
            inputs, label = Variable(inputs), Variable(label)
        outputs = net(inputs)
        _, prediction = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (prediction==label.data).sum().item()
    print('Test Accuracy: {0:.6f}%'.format(correct/total*100))

## Testing
print('Training done... Now testing...')
net.eval()
correct, total = 0,0
for data in testloader:
    inputs, label = data
    if is_cuda:
        inputs, label = Variable(inputs.cuda()), Variable(label.cuda())
    else:
        inputs, label = Variable(inputs), Variable(label)
    outputs = net(inputs)
    _, prediction = torch.max(outputs.data, 1)
    total += label.size(0)
    correct += (prediction==label.data).sum().item()
print('Test Accuracy: {0:.6f}%'.format(correct/total*100))
