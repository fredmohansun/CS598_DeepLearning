import os
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## parser
parser = argparse.ArgumentParser()
parser.add_argument('--extract_features', type=int, default=8, help='extract features')
parser.add_argument('--Generator', type=int, default=0, help='trained with generator')
opt = parser.parse_args()

## Hyperpara
batch_size = 128
lr = 0.1
weight_decay = 0.001

## Data
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
        transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

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

    def forward(self, x, extract_features=0):
        x = self.conv1(x)
        if extract_features==1:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
            return x
        x = self.conv2(x)
        if extract_features==2:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
            return x
        x = self.conv3(x)
        if extract_features==3:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
            return x
        x = self.conv4(x)
        if extract_features==4:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
            return x
        x = self.conv5(x)
        if extract_features==5:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
            return x
        x = self.conv6(x)
        if extract_features==6:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
            return x
        x = self.conv7(x)
        if extract_features==7:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
            return x
        x = self.conv8(x)
        if extract_features==8:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
            return x
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        return (self.fc1(x), self.fc10(x))

## Plotting
def myplot(samples):
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(10,10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

if bool(opt.Generator):
    model = torch.load('download/D.model')
else:
    model = torch.load('download/D1.model')
model.cuda()
model.eval()

i, data = testloader.__next__()
inputs, label = data
inputs = Variable(inputs, requires_grad=True).cuda()

X = inputs.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

for i in range(200):
    output = model(X,opt.extract_features)
    print(output.size())

    loss = -output[torch.arange(batch_size).type(torch.int64), torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1]
    accuracy = (float(prediction.eq(Y.data).sum())/float(batch_size))*100.0
    print(i, accuracy, -loss)

    X = X - lr * gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0]=1.0
    X[X<-1.0]=-1.0

samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = myplot(samples[0:100])
plt.savefig('visualization/max_features_'+('w_' if bool(opt.Generator) else 'wo_')+str(opt.extract_features)+'.png', bbox_inches='tight')
plt.close(fig)
