import numpy as np
import data_processer
import random
import deep_learning_plain

random.seed(98121)
arrays = [np.array(map(str, line.split('\t'))) for line in open('./housedata-regular.csv')]
X_train = []
Y_train = []
X_dev = []
Y_dev = []

indexRow = []
for index, element in enumerate(arrays):
    if(index == 0):
        indexRow = element
        print(element)
        continue
    x, y = data_processer.processRow(indexRow, element)
    if len(x) != 0:
        r = random.random()
        if r < 0.95:
            X_train.append(x)
            Y_train.append(y)
        else:
            X_dev.append(x)
            Y_dev.append(y)

X_train = np.array(X_train).T
X_train_mean = np.mean(X_train, axis=1).reshape((1, -1)).T
X_train_std = np.std(X_train, axis=1).reshape((1, -1)).T

X_train[0:8,:] = (X_train[0:8,:] - X_train_mean[0:8])/X_train_std[0:8]
X_train[-1,:] = (X_train[-1,:] - X_train_mean[-1])/X_train_std[-1]
print(X_train[:,0])

Y_train = np.array(Y_train).reshape((1, -1))
X_dev = np.array(X_dev).T

X_dev[0:8,:] = (X_dev[0:8,:] - X_train_mean[0:8])/X_train_std[0:8]
X_dev[-1,:] = (X_dev[-1,:] - X_train_mean[-1])/X_train_std[-1]

Y_dev = np.array(Y_dev).reshape((1, -1))
print(X_train.shape)
print(Y_train.shape)
print(X_dev.shape)
print(Y_dev.shape)

parameters = deep_learning_plain.model(X_train, Y_train, X_dev, Y_dev)
