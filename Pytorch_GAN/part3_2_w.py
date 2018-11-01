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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 196, 3, 1, 1),
            nn.LayerNorm([196,32,32]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm([196,16,16]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm([196,16,16]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm([196,8,8]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm([196,8,8]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm([196,8,8]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(196, 196, 3, 1, 1),
            nn.LayerNorm([196,8,8]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(196, 196, 3, 2, 1),
            nn.LayerNorm([196,4,4]),
            nn.LeakyReLU(0.1)
        )

        self.pool = nn.MaxPool2d(4,4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x):
        x = self.conv_layer(x)
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

model = torch.load('download/Epoch_200_D.model')
model.cuda()
model.eval()

i, data = testloader.__next__()
inputs, label = data
inputs = Variable(inputs, requires_grad=True).cuda()

X = inputs.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

for i in range(200):
    _, output = model(X)

    loss = -output[torch.arange(10).type(torch.int64), torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1]
    accuracy = (float(prediction.eq(Y.data).sum())/float(10.0))*100.0
    print(i, accuracy, -loss)

    X = X - lr * gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0]=1.0
    X[X<-1.0]=-1.0

samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = myplot(samples)
plt.savefig('visualization/max_class_w.png', bbox_inches='tight')
plt.close(fig)
