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

model = torch.load('download/D1.model')
model.cuda()
model.eval()

i, data = testloader.__next__()
inputs, label = data
inputs = Variable(inputs, requires_grad=True).cuda()
label_alternate = (label+1)%10
label_alternate = Variable(label_alternate).cuda()
label = Variable(label).cuda()

samples = inputs.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = myplot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

_, output = model(inputs)
prediction = output.data.max(1)[1]
accuracy = (float(prediction.eq(label.data).sum())/float(batch_size))*100.0
print(accuracy)

criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, label_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=inputs,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)

fig = myplot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)

gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
inputs_modified = inputs - gain*0.007843137*gradients
inputs_modified[inputs_modified>1.0] = 1.0
inputs_modified[inputs_modified<-1.0] = -1.0

## evaluate new fake images
_, output = model(inputs_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(label.data).sum() ) /float(batch_size))*100.0
print(accuracy)

## save fake images
samples = inputs_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)
print(samples.shape)

fig = myplot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)
