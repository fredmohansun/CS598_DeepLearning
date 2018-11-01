import numpy as np
import h5py
import time
import copy
from random import randint
import itertools
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#number of inputs
D = 28
#number of outputs
K = 10
#number of Channels per Layer
C = 3
#size of filter
kxy = 5
model = {}
model['W'] = np.random.randn(K,D-kxy+1,D-kxy+1,C) / D
model['b'] = (np.random.randn(K) / np.sqrt(D))[:,np.newaxis]
model['K'] = np.random.randn(kxy,kxy,C) / np.sqrt(D)
model_grads = copy.deepcopy(model)
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def sigmoidp(Z):
    temp = sigmoid(Z)
    return temp*(1-temp)
def ReLU(Z):
    ans = np.copy(Z)
    ans[ans < 0] = 0
    return ans
def ReLUp(Z):
    ans = np.copy(Z)
    ans[ans >=0] = 1
    ans[ans<0]=0
    return ans
def convolution(X, K):
    dx, dy = X.shape
    kx, ky, p = K.shape
    ans = np.zeros((dx-kx+1,dy-ky+1,p))
    for k in range(ans.shape[2]):
        for j in range(ans.shape[1]):
            for i in range(ans.shape[0]):
                ans[i,j,k] = np.einsum('ij,ij',X[i:i+kx,j:j+ky], K[:,:,k])
    return ans
def conv(x, K):
    return np.sum(x*K)
def convolution2(X, K):
    dx, dy = X.shape
    kx, ky, p = K.shape
    ans = np.zeros((dx-kx+1,dy-ky+1,p))
    for k in range(ans.shape[2]):
        temp = [X[i:i+kx,j:j+ky] for i,j in itertools.product(range(ans.shape[0]),range(ans.shape[1]))]
        ans[:,:,k] = np.array(list(map(conv,temp,[K[:,:,k] for i in range(len(temp))]))).reshape(ans.shape[0],ans.shape[1])
    return ans
def forward(x, y, model):
    Z = convolution(x, model['K'])
    H = sigmoid(Z)
    U = np.einsum('hijk,ijk',model['W'],H).reshape(model['W'].shape[0],1) + model['b']
    p = softmax_function(U)
    return Z, H, p
def backward(x, y, Z, H, p, model, model_grads):
    dU = np.copy(p)
    dU[y] = dU[y] - 1
    delta = np.einsum('hijk,h',model['W'],dU.reshape(dU.shape[0]))
    dK = convolution(x, delta * sigmoidp(Z))
    dW = np.zeros_like(model['W'])
    for i in range(dW.shape[0]):
        dW[i,:,:,:] = dU[i] * H
    model_grads['W']=dW
    model_grads['b']=dU
    model_grads['K']=dK
    return model_grads
time1 = time.time()
LR = .1
num_epochs = 20
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.01
    if (epochs > 10):
        LR = 0.001
    if (epochs > 15):
        LR = 0.0001
    total_correct = 0
    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1)
        y = y_train[n_random]
        x = x_train[n_random,:].reshape(D,D)
        Z, H, p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, Z, H, p, model, model_grads)
        model['W'] = model['W'] - LR*model_grads['W']
        model['b'] = model['b'] - LR*model_grads['b']
        model['K'] = model['K'] - LR*model_grads['K']
    print('No. {0} epoch accuracy: {1:.10f}'.format(epochs,total_correct/np.float(len(x_train))))
time2 = time.time()
print('Train took {0:.10f} seconds'.format(time2-time1))
######################################################
#test data
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:].reshape(D,D)
    _ , _, p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print('Test accuracy: {0:.10f}'.format(total_correct/np.float(len(x_test))))
