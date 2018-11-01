import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#number of inputs
D = 28*28
#number of outputs
K = 10
#number of Hidden units per Layer
Dh = 100
model = {}
model['W'] = np.random.randn(Dh,D) / np.sqrt(D)
model['b1'] = (np.random.randn(Dh) / np.sqrt(D))[:,np.newaxis]
model['C'] = np.random.randn(K, Dh) / np.sqrt(D)
model['b2'] = (np.random.randn(K) / np.sqrt(D))[:,np.newaxis]
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
def forward(x,y, model):
    Z = model['W'] @ x + model['b1']
    H = sigmoid(Z)
    U = model['C'] @ H + model['b2']
    p = softmax_function(U)
    return Z, H, p
def backward(x, y, Z, H, p, model, model_grads):
    dU = np.copy(p)
    dU[y] = dU[y] - 1
    dC = dU @ H.T
    delta = model['C'].T @ dU
    db1 = delta * sigmoidp(Z)
    dW = db1 @ x.T
    model_grads['W']=dW
    model_grads['b1']=db1
    model_grads['b2']=dU
    model_grads['C']=dC
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
        temp = x_train[n_random,:]
        x = temp[:,np.newaxis]
        Z, H, p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, Z, H, p, model, model_grads)
        model['W'] = model['W'] - LR*model_grads['W']
        model['b1'] = model['b1'] - LR*model_grads['b1']
        model['C'] = model['C'] - LR*model_grads['C']
        model['b2'] = model['b2'] - LR*model_grads['b2']
        #print(model_grads['b2'])
    print('No. {0} epoch accuracy: {1:.10f}'.format(epochs,total_correct/np.float(len(x_train))))
time2 = time.time()
print('Train took {0:.10f} seconds'.format(time2-time1))
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    temp = x_test[n][:]
    x = temp[:,np.newaxis]
    _ , _, p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print('Test accuracy: {0:.10f}'.format(total_correct/np.float(len(x_test))))
