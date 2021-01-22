import torch
from torchvision import datasets, transforms
import torchvision
import os
import random as r
import numpy as np
import torch

def seeder(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    r.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seeder(1)


trainset = datasets.MNIST('data/MNIST/', download=True, train=True, transform=None)
testset = datasets.MNIST('data/MNIST/', download=True, train=False, transform=None)


def normalize(a):
   n = 2. * (a - np.min(a)) / np.ptp(a) - 1
   return n

def TRAIN_SIZE():

    x_train = torch.reshape(trainset.data, (trainset.data.shape[0], 784))
    y_train = trainset.targets

    return x_train, y_train


def TEST_SIZE():
    x_test = torch.reshape(testset.data, (testset.data.shape[0], 784))
    y_test = testset.targets

    return x_test, y_test


def prepare_MNIST(normal, anomalous):


    All_training_digits, All_training_digits_labels = TRAIN_SIZE()
    All_testing_digits, All_testing_digits_labels = TEST_SIZE()

    All_training_digits = All_training_digits.data.numpy()
    All_training_digits_labels = All_training_digits_labels.data.numpy()

    All_testing_digits = All_testing_digits.data.numpy()
    All_testing_digits_labels = All_testing_digits_labels.data.numpy()

    #concatenate all training and test data and their corresponding labels
    x = np.concatenate((All_training_digits, All_testing_digits), axis=0)
    y = np.concatenate((All_training_digits_labels, All_testing_digits_labels), axis=0)

    x = np.array(x, dtype=np.float32)/255.0

    # separate normal data and anomalous data
    normal_idx = np.where(y == normal)
    if anomalous != -1:
        anomalous_idx = np.where(y == anomalous)
    else:
        anomalous_idx = np.where(y != normal)

    # Grab the relevant data and labels
    normal_data = x[normal_idx]
    anomalous_data = x[anomalous_idx]

    normal_data_label = np.zeros(normal_data.shape[0])
    anomalous_data_label = np.ones(anomalous_data.shape[0])


    return normal_data, normal_data_label, anomalous_data, anomalous_data_label


