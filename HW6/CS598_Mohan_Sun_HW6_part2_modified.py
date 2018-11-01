import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision
import torchvision.transforms as transforms

## Hyperparas
batch_size = 128
max_epoch = 200
learning_rate = 0.0001
gen_train = 1
n_z = 100
n_classes = 10

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

## Test Cuda
is_cuda = torch.cuda.is_available()

## Model & Model Specific F
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

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Sequential(
             nn.Linear(100, 196*4*4),
             nn.BatchNorm1d(196*4*4)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(196,196,4,2,1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(196,196,3,1,1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(196,196,3,1,1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(196,196,3,1,1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(196,196,4,2,1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(196,196,3,1,1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(196,196,4,2,1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )
        self.conv8 = nn.Conv2d(196,3,3,1,1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 196, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return self.conv8(x)

def calc_gradient_penalty(aD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    if is_cuda:
        alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    if is_cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = aD(interpolates)

    if is_cuda:
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def noise_generator():
    label = np.random.randint(0, n_classes, batch_size)
    noise = np.random.normal(0, 1, (batch_size, n_z))
    label_onehot = np.zeros((batch_size, n_classes))
    label_onehot[np.arange(batch_size), label] = 1
    noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
    noise = noise.astype(np.float32)
    noise = torch.from_numpy(noise)
    if is_cuda:
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
    else:
        noise = Variable(noise)
        fake_label = Variable(torch.from_numpy(label))
    return noise, fake_label

## Optimizer and Cuda
aD = discriminator()
aG = generator()

if is_cuda:
    aD.cuda()
    aG.cuda()
    aD = nn.DataParallel(aD, device_ids=range(torch.cuda.device_count()))
    aG = nn.DataParallel(aG, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer_D = optim.Adam(aD.parameters(), lr = learning_rate, betas=(0,0.9))
optimizer_G = optim.Adam(aG.parameters(), lr = learning_rate, betas=(0,0.9))

for epoch in range(max_epoch):
    timer1 = time.time() 
    # Training
    aG.train()
    aD.train()
    for i, data in enumerate(trainloader):
        inputs, target = data
        if inputs.shape[0] < batch_size:
            continue

        if is_cuda:
            real_data, real_label = Variable(inputs).cuda(), Variable(target).cuda()
        else:
            real_data, real_label = Variable(inputs), Variable(target)

        # Train G
        if (i%gen_train)==0:
            for p in aD.parameters():
                p.requires_grad_(False)

        aG.zero_grad()

        noise, fake_label = noise_generator()

        fake_data = aG(noise)
        gen_source, gen_class = aD(fake_data)

        gen_source = gen_source.mean()
        gen_class = criterion(gen_class, fake_label)

        gen_cost = -gen_source + gen_class
        gen_cost.backward()

        for group in optimizer_G.param_groups:
            for p in group['params']:
                state = optimizer_G.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer_G.step()

        #Train D
        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        noise, fake_label = noise_generator()

        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)
        disc_fake_source = disc_fake_source.mean()
        disc_real_source, disc_real_class = aD(real_data)
        disc_real_source = disc_real_source.mean()

        prediction = disc_real_class.data.max(1)[1]
        train_acc = (float(prediction.eq(real_label.data).sum())/float(batch_size))*100.0

        disc_real_class = criterion(disc_real_class, real_label)
        disc_fake_class = criterion(disc_fake_class, fake_label)
        gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty

        disc_cost.backward()
        for group in optimizer_D.param_groups:
            for p in group['params']:
                state = optimizer_D.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer_D.step()

    # Testing
    aD.eval()
    with torch.no_grad():
        acc2 = []
        for i, data in enumerate(testloader):
            inputs, target = data

            if is_cuda:
                inputs, target = Variable(inputs).cuda(), Variable(target).cuda()
            else:
                inputs, target = Variable(inputs), Variable(target)
            with torch.no_grad():
                _, output = aD(inputs)

            prediction = output.data.max(1)[1]
            test_acc = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0
            acc2.append(test_acc)
        timer2 = time.time()
        print('Epoch {0}: test_acc: {1:.4f}%, epoch_time: {2:.2f}'.format(
            epoch, np.mean(acc2), timer2-timer1
        ))

    torch.save(aG, 'model/Epoch_{0}_G.model'.format(epoch))
    torch.save(aD, 'model/Epoch_{0}_D.model'.format(epoch))

torch.save(aG, 'model/G.model')
torch.save(aD, 'model/D.model')
