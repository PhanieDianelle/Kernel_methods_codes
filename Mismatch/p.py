import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse 

from models.kernel_SVM import KSVM
from kernels.spectrum_kernel import spectrum_kernel
from mismatch_kernel.mismatch_kernel import MismatchKernel


ALPHABET=['A','G','T','C','a','g','t','c']
K=5
m=1
mk = MismatchKernel(ALPHABET, K, m)



test_predictions = np.array([], dtype=int)

# hyperparameters


DATA_PATH="./data"


#print(f'processing dataset {i+1}')

TRAIN_DATA_PATH = os.path.join(DATA_PATH, f'Xtr2.csv')
TRAIN_LABEL_PATH = os.path.join(DATA_PATH, f'Ytr2.csv')
TEST_DATA_PATH = os.path.join(DATA_PATH, f'Xte2.csv')

data = np.loadtxt(TRAIN_DATA_PATH , dtype=str,
                delimiter=',', skiprows=1, usecols=1)

test_data = np.loadtxt(TEST_DATA_PATH, dtype=str,
                    delimiter=',', skiprows=1, usecols=1)

labels = pd.read_csv(TRAIN_LABEL_PATH, index_col=0).to_numpy().squeeze()


n=len(data)
training_data, test_data = data[:int(0.8*n)] , data[int(0.8*n):]
n_training, n_test = training_data.shape[0], test_data.shape[0]
training_labels = labels[:int(0.8*n)]
test_labels=labels[int(0.8*n):]

k =11


lambd = 1.0

# training kernel matrix
K_train = np.zeros((n_training, n_training))
for i in range(n_training):
    for j in range(i+1):
        source, target = training_data[i], training_data[j]
        K_train[i, j] = spectrum_kernel(source, target, k)+mk.get_kernel(source,target)
        K_train[j, i] = K_train[i, j]

print('computing the kernel matrix')
classifier = KSVM(training_data, training_labels, lambd, kernel=K_train)

print('fiting')
classifier.fit()

# # test kernel matrix
K_test = np.zeros((n_test, n_training))
for i in range(n_test):
    for j in range(n_training):
        source, target = test_data[i], training_data[j]
        K_test[i, j] = spectrum_kernel(source, target, k)+mk.get_kernel(source,target)

print('predicting')
predictions = classifier.predict(K_test)
test_predictions = np.append(test_predictions, predictions, 0)

def error(ypred, ytrue):
    e = (ypred != ytrue).mean()
    return e

def accuracy(ypred, ytrue):
    acc=np.mean(ypred==ytrue)
    return acc
print(accuracy(predictions,test_labels))
print(error(predictions,test_labels))
