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
max_epoch = 50

## Datasets
train_set = torchvision.datasets.CIFAR10(root='../data',train=True,download=True,
            transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                         ]))

test_set = torchvision.datasets.CIFAR10(root='../data',train=False,download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                         ]))

#DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
trainloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=8)

testloader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=8)

print('Data is ready...')
## Network Structure
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv_layers = nn.Sequential(
            #Conv2d(in, out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(3,64,4,1,2),
            #ReLU(inplace=False)
            nn.ReLU(),
            #BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
            nn.BatchNorm2d(64),

            nn.Conv2d(64,64,4,1,2),
            nn.ReLU(),
            #MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d(2,2),
            #Dropout2d(p=0.5, inplace=False)
            #nn.Dropout2d(p=0.2),

            nn.Conv2d(64,64,4,1,2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,4,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64,64,4,1,2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,0),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(64,64,3,1,0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.4)
        )
        self.fc_layers = nn.Sequential(
            #Linear(in_features, out_features, bias=True)
            nn.Linear(64*4*4,500),
            nn.ReLU(),
            nn.Linear(500,500)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)

## Initialization
def initialization(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0,math.sqrt(1/(m.kernel_size[0]*m.kernel_size[1]*m.out_channels)))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0,1)
        m.bias.data.zero_()

is_cuda = torch.cuda.is_available()

net = network()
net.apply(initialization)
print('Initialization is done...')
if is_cuda:
    net.cuda()
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    print('Cuda is set up...')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

print('Start Training...')
## Training
net.train()
for epoch in range(max_epoch):
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
    torch.save(net.state_dict(),'./save/model_{0}.pkl'.format(epoch))
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
