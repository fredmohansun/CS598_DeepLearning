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
batch_size = 128
max_epoch = 100

## Datasets
train_set = torchvision.datasets.CIFAR100(root='../data',train=True,download=True,
            transform=transforms.Compose([transforms.RandomHorizontalFlip(),
					  transforms.RandomCrop(32,padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.507,0.487,0.441),std=(0.267,0.256,0.276))
                                         ]))

test_set = torchvision.datasets.CIFAR100(root='../data',train=False,download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.507,0.487,0.552),std=(0.267,0.256,0.276))
                                         ]))

#DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
trainloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=8)

testloader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=8)

print('Data is ready...')
## Network Structure
class ResidualBlock(nn.Module):
    def __init__(self, in_size, out, first_stride=1, kernal=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out, kernal, first_stride, padding),
            nn.BatchNorm2d(out),
            nn.ReLU(),
            nn.Conv2d(out, out, kernal, 1, padding),
            nn.BatchNorm2d(out)
        )
        if first_stride != 1 or in_size != out:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_size,out, 1, first_stride),
                nn.BatchNorm2d(out)
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        out += residual
        return out


class network(nn.Module):
    def __init__(self, block):
        super(network, self).__init__()
        self.init_layers = nn.Sequential(
            #Conv2d(in, out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(3,32,3,1,1),
            #ReLU(inplace=False)
            #BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            #Dropout2d(p=0.5, inplace=False)
            nn.Dropout2d(p=0.2)
        )
        n_conv_layers = {'conv_block_256':2, 'conv_block_128':4,
                             'conv_block_64':4, 'conv_block_32':2}
        n_conv_layers_init_stride = {'conv_block_256':2, 'conv_block_128':2,
                                    'conv_block_64':2, 'conv_block_32':1}
        conv_layers = []
        
        for i in range(n_conv_layers['conv_block_32']):
            if i == 0:
                conv_layers.append(block(32, 32, first_stride = n_conv_layers_init_stride['conv_block_32']))    
            conv_layers.append(block(32, 32))
            
        for i in range(n_conv_layers['conv_block_64']):
            if i == 0:
                conv_layers.append(block(32, 64, first_stride = n_conv_layers_init_stride['conv_block_64']))    
            conv_layers.append(block(64, 64))

        for i in range(n_conv_layers['conv_block_128']):
            if i == 0:
                conv_layers.append(block(64, 128, first_stride = n_conv_layers_init_stride['conv_block_128']))    
            conv_layers.append(block(128, 128))
        
        for i in range(n_conv_layers['conv_block_256']):
            if i == 0:
                conv_layers.append(block(128, 256, first_stride = n_conv_layers_init_stride['conv_block_256']))    
            conv_layers.append(block(256, 256))
        
        self.conv_layers = nn.Sequential(*conv_layers)

        self.max_pool =	nn.MaxPool2d(3, 2, 1)
        self.fc = nn.Linear(256*2*2,100)

    def forward(self, x):
        out = self.init_layers(x)
        out = self.conv_layers(out)
        out = self.max_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

## Initialization
def initialization(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0,math.sqrt(1/(m.kernel_size[0]*m.kernel_size[1]*m.out_channels)))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0,1)
        m.bias.data.zero_()

is_cuda = torch.cuda.is_available()

net = network(ResidualBlock)
net.apply(initialization)
print('Initialization is done...')
if is_cuda:
    net.cuda()
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    print('Cuda is set up...')

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr = 0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

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
        #if(epoch > 5):
        #    for group in optimizer.param_groups:
        #        for p in group['params']:
        #            state = optimizer.state[p]
        #            if(state['step'] >= 1024):
        #                state['step'] = 1000
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
