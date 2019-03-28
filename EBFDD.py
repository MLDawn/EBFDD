'''
This piece of code executes experiements between the EBFDD, RBFDD, Isolation Forest, GMM, classic Auto-Encoder, and One-class SVM
On a variety of benchmark datasets. By setting the name of the dataset, the desired algorithms, the scenario in which we choose the normal and anomalous labels
, and finally by choosing the desired hyper-parameters, this code will train the selected algorithms on the desired datasets.
It will save the results separately for each algorithm. These files can be used by the EBFDD Rank Visualiser.py code so we can compare
the BEST performance of each algorithm on the selected dataset, accompanied by the corresponding winner hyper-parameters.
'''
import random as r
import numpy as np
import time as time
from pylab import *
import matplotlib.pyplot as plt
import math
from sklearn.metrics import precision_recall_fscore_support
# Import the datasetscripts from the folder DataPreparationScripts
# These scripts will normalize the data between 0 and 1 and separate
# the data between Normal and Anomalous categories
from DataPreparationScripts import letter_Script, wave_Script, Fashion_MNIST_Script, \
gamma_Script, Page_Script, Fault_Script, PARTICLE_Script, ImageSegmentation_Script,\
Spambase_Script, landsat_Script, Skin_Script

# Used to shuffle the data at the begining of every epoch for the EBFDD, and RBFDD algorithms
from sklearn.utils import shuffle
# Used to pickle the output results in a file so we could use it by the EBFDD Rank Visualiser.py
# To see the performance of each algorithm
import pickle
# Import the Support Vector Machine for the One-Class SVM algorithm
from sklearn import svm
# Import the classes required to build the Auto Encoder
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
# Import the Isolation Forest
from sklearn.ensemble import IsolationForest
# The kmeans is used for the pre-training phase of the EBFDD/RBFDD algorithms
from sklearn.cluster import KMeans
# This is for the Gaussian Mixture Model
from sklearn import mixture
# This is used to apply the PCA algorithm on the data should we want compression
from sklearn import decomposition
from itertools import permutations, combinations
from sklearn.metrics import confusion_matrix
# Used for converting covariance matrices for both the EBFDD and RBFDD algorithms
from numpy.linalg import inv
# This is used to visualise the gaussians for the EBFDD, RBFDD, and GMM (is used if the dimensionality is reduced to 2)
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import sys
# Ensure reproducibility
np.random.seed(1)
r.seed(1)

def normalize(X):
    ''' 
      If PCA reduced the dimensionality, this will normalize the compressed data between 0 and 1
      Before training.Normalises the data between 0 and 1:
        - Input: Data to be normalised numpy (N,D)
        - Output: Normalised data numpy (N,D)
        - Called by PCA_compress function
    '''
    maximum = np.max(X, axis=0)
    minimum = np.min(X, axis=0)
    denum = maximum - minimum
    num = np.subtract(X, minimum)
    X = np.divide(num, denum)
    return X

def PCA_compress(dimensionality, normal_data, anomalous_data):
    ''' 
      If PCA reduced the dimensionality, this will normalize the compressed data between 0 and 1
      Before training.Normalises the data between 0 and 1:
        - Input: 
            - Normal and Anomalous Data to be compressed numpy (N,D)
            - Desired dimensionality for compression
        - Output: 
            - Compressed data numpy (N,D), which is normalised between 0 and 1
        - Called by PCA_compress function
    '''
    pca_train = decomposition.PCA(n_components=dimensionality, whiten=True)
    pca_train.fit(normal_data)
    normal_data = pca_train.transform(normal_data)
    anomalous_data = pca_train.transform(anomalous_data)
    # Normalize the normal and anomalous data by calling the Normalize() function
    normal_data = normalize(normal_data)
    anomalous_data = normalize(anomalous_data)
    return normal_data, anomalous_data


def prepare_training_data(normal_data, sample_size=0.80):
    ''' 
      Randomly samples the normal data by the sample_size
        - Input: 
            - Normal and Anomalous Data to be sampled numpy (N,D)
            - the sample_size which dictates the fraction for sampling
        - Output: 
            - the training data: Sampled portion of the entire normal data numpy (N,D), with the indices. These indices will be used by 
              prepare_testing_data function, so it could make sure thet it will not use these for preparing the test data
        - Called by the most inner loop of each algorithm, everytime we need to sample.
    '''
    # bootstrap sample from the train_data
    boot_strap_size = int(normal_data.shape[0] * sample_size)
    # Generate a uniform random sample from np.arange(boot_strap_size) of size boot_strap_size:
    # generate the index of the sampled values NO REPLACEMENT
    boot_strap_train_index = np.random.choice(normal_data.shape[0], boot_strap_size, replace=False)
    boot_strap_train_data = normal_data[boot_strap_train_index]
    return boot_strap_train_data, boot_strap_train_index

def prepare_testing_data(normal_data, normal_data_label, boot_strap_train_index):
    ''' 
      Uses the entire anomalous data and concatenates it with the portion of the normal data NOT used for training by prepare_training_data
        - Input: 
            - Normal data numpy (N,D)
            - the boot_strap_train_indexused to find the portion of the normal data that is not used for the training
        - Output: 
            - the test data and the test labels
        - Called by the most inner loop of each algorithm, everytime we need to sample.
    '''
    test_data_normal_portion = np.delete(normal_data, boot_strap_train_index, axis=0)
    test_label_normal_portion = np.delete(normal_data_label, boot_strap_train_index, axis=0)
    # Now concatenate these newly extracted normal samples to the existing
    # anomalous data to create the the final test set for the prediction
    concatenated_test_data = np.concatenate((test_data_normal_portion, anomalous_data), axis=0)
    concatenated_test_label = np.concatenate((test_label_normal_portion, anomalous_data_label), axis=0)
    # Before passing the test data, let's shuffle it to avoid ANY STREAMIMG assumption
    concatenated_test_data, concatenated_test_label = shuffle(concatenated_test_data, concatenated_test_label, random_state=0)
    return concatenated_test_data, concatenated_test_label

# The class for the Elliptical Basis Function Data Descriptor network
class EBFDD:
    def __init__(self, dataset_name, mini_batch_size,  H, beta, theta, bp_eta, bp_epoch, normal, anomalous, statistics=False, boundary=False):
        '''
        This constructor sets the following class attributes:
            - The name of the dataset correntlu under experiment
            - The size of the mini-batch during training
            - Number of Elliptical gaussian units in the hidden layer (H)
            - beta: That is the coefficients of the l-2 norm of the diagonal elements of the covariance matrices
            - theta: That is the coefficients of the l-2 norm of the weights connecting the hidden layer to the output neuron
            - bp_eta: The learning rate during back-propagation
            - bp_epoch: The number of epochs for training
            - normal: the labels of the normal class
            - anomalous: the labales of the anomalous class
            - statistics: If true, then certain statistics of the training process will be visualised: Like the trend of the
                          of the output, the change in the actual cost function, in the size of the Gaussians and the magnitude of the weights
            - boundary: If true, we will see the actual decision boundary of the trained EBFDD (only happens for the dimensionality of 2)
        '''
        self.mini_batch = mini_batch_size
        self.H = H
        self.beta = beta
        self.theta = theta
        self.BP_eta = bp_eta
        self.BP_epoch = bp_epoch
        self.normal = normal
        self.anomalous = anomalous
        self.dataset_name = dataset_name
        self.low = float(-0.01) / (math.sqrt(self.H))
        self.high = float(0.01) / (math.sqrt(self.H))
        self.statistics = statistics
        self.boundary = boundary
    # It happens sometimes during updating the covariance matrices, that the matrix is no longer invertible.
    # The nearPSD function computes the nearest positive semi-definite matrix of the uninvertible covariance matrices
    def nearPSD(self, A, epsilon=1e-23):
        '''
            Input: The uninvertible covariance matrix (d, d) numpy array
            Output: The nearest positive semi-definite covariance matrix, which is going to be invertible
        '''
        n = A.shape[0]
        eigval, eigvec = np.linalg.eig(A)
        val = np.matrix(np.maximum(eigval, epsilon))
        vec = np.matrix(eigvec)
        T = 1 / (np.multiply(vec, vec) * val.T)
        T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
        B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
        out = B * B.T
        return (out)
    def clip_power(self, power):
        power[np.where(power < -1e+1000)] = -1000
        # power[np.where(np.abs(power) < 1e-10)] = 0
        power[np.where(power > 709)] = 709
        return power
    def ebfdd_forward(self, X, m, cov, W):
        '''
            This function performs the forward pass.
            Inputs: the input mini-batch of data(X), the means of the Gaussians(m), the covariance metrices(cov), and the weight vector(W)
            Outputs:
                - The output vector (y) of the network, that is a vector of the size of our minibatch, X
                - The likelihoods of all the data points in X from each Gaussian (P), that is (H,mini-batchsize)
                - Preactivation vector (Z) at the output neuron, that is a vector of the size of our minibatch, X
                - The matrix (a) that holds the difference between X and our means m as this is important for the backprop
        '''
        # It is important to store the INVERTED covariance matrices in ONE GO, as they are needed during
        # the Back-Propagation phase
        invcov = inv(cov)
        a = (X - m[:, np.newaxis])
        d = np.matmul(np.matmul(a, invcov), np.transpose(a, [0, 2, 1]))
        power = -0.50 * np.diagonal(d, axis1=1, axis2=2)
        # This power should be exponentiated. But first we make sure to clip it to avoid overflow!
        power = self.clip_power(power)
        # Now the array P has ALL the likelihoods of ALL the training data in X, for ALL the Gaussian Kernels
        P = np.exp(power)
        # Z is what the output neuron has received
        Z = np.sum((P.T * W).T, axis=0)
        # Here we compute the output vector, y, using Yann Lecun's recommended tanh() function in Efficient Backpropagation.
        y = 1.7159 * np.tanh(float(2 / 3) * Z)
        return y, P, Z, a
    
    def ebfdd_backward(self, y, P, Z, a, cov, W, NUM):
        '''
        This function performs the backprop after the current mini-batch has gone through the forwardpass and produced the output vector, y
        Inputs:
            - The output vector of the network, y
            - The likelihood matrix P, that holdes the likelihood of every data vector in the mini-batch, across all the hidden nodes
            - The preactivation vector of the output neuron, Z
            - The matrix of the differences between the input vectors and the means of the centroids (computed by the forward pass)
            - The covariance matrices, cov, of all the hidden nodes
            - The weight vector W, connecting the hidden layer to the output neuron
            - The number of data vectors in the mini-batch, NUM
        Outputs:
            - The gradients of the error (our proposed error function found in the paper) wrt. the weights, 
            the covariance matrices and the Gaussian means denotes as: dEdW, dEdCov, and dEdM
        '''
        invcov = inv(cov)
        dEdZ = (y - 1) * (1.1439 * (1 - np.square(np.tanh(float(2 / 3) * Z))))
        dEdW = np.sum(P * dEdZ, axis=1) + NUM * self.theta * W
        dZdP = np.repeat([W], NUM, axis=0).T
        
        dEdP = np.multiply(dEdZ, dZdP) * P
        
        dPdM = np.matmul(invcov, np.transpose(a, [0, 2, 1]))

        dPdM = np.transpose(dPdM, [0, 2, 1])
        
        dEdM = np.reshape(np.matmul(dEdP[:, None], dPdM), (self.H, input_dim))
        
        A = a.reshape(a.shape[0], a.shape[1], 1, a.shape[2])
        A = A * np.transpose(A, [0, 1, 3, 2])
        dEdP = np.reshape(dEdP, (dEdP.shape[0], NUM, 1))
        A = np.sum(A * dEdP[:, :, np.newaxis], axis=1)
        dEdCov = 0.50 * np.matmul(np.matmul(invcov, A), invcov)
        dEdCov = dEdCov + self.beta * NUM * cov * np.identity(cov.shape[-2])
        return dEdW, dEdCov, dEdM
    def kmeans(self, train_data):
        '''
        The kmeans is used as the pre-training phase before the EBFDD's training gets started, to initialize the
        Gaussians with a good set of parameters.
        Inputs:
            - The whole training data
        Outputs:    
            - The initial centers and covariance matrices of the Gaussians
        '''
        kmeans = KMeans(n_clusters=self.H, random_state=0).fit(train_data)
        print("K-means is DONE!!!")
        # Here we grab the means, computed by the k-means
        m = kmeans.cluster_centers_
        labels = np.array(kmeans.labels_)
        # For every Gaussian center, we compute its distance fro its farthest member, divide it by 3, and that becomes
        # The initial variance for that Gaussian
        sd = []
        for h in range(self.H):
            index = np.argwhere(labels == h)
            data = train_data[np.reshape(index, (index.shape[0],))]
            distance = (np.sum((data - m[h]) ** 2, axis=1)) ** (0.50)
            maximum_distance = np.max(distance)
            if maximum_distance > 0:
                sd.append(maximum_distance / float(3))
            else:
                sd.append(sys.float_info.epsilon)
        sd = np.array(sd)
        # In order to build the covariance matrices, this value will be repeated 
        # across the diagonal of the covariance matrix of the Gaussian, while zeroing out all the other values --> A radial kernel
        cov = np.random.rand(self.H, input_dim, input_dim)
        for h in range(self.H):
            cov[h] = sd[h] * np.identity(cov[h].shape[-2])
        return cov, m
    def plot_gaussians(self, train_data, anomalous_data, m, cov):
        # A simple plotting function of the Gaussians. Gets activated if the dimensionality of 
        # the data has been reduced to 2 using our defined PCA_compress function.
        xx, yy = np.mgrid[-5:5:.1, -5:5:.1]
        pos = np.dstack((xx, yy))
        figure()
        plt.grid()
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.axes().set_aspect('equal', 'datalim')
        plt.scatter(anomalous_data[:, 0], anomalous_data[:, 1], c="b", alpha=0.07)
        plt.scatter(train_data[:, 0], train_data[:, 1], c="r", alpha=0.07)
        plt.xlabel("First PC", fontsize=15)
        plt.ylabel("Second PC", fontsize=15)

        for h in range(self.H):
            try:
                rv = multivariate_normal(m[h], cov[h])
                plt.xlim(-3, 3)
                plt.ylim(-3, 3)
                plt.axes().set_aspect('equal', 'datalim')
                plt.contour(xx, yy, rv.pdf(pos))
                plt.xlabel("First PC", fontsize=15)
                plt.ylabel("Second PC", fontsize=15)

            except ValueError:
                cov[h] = self.nearPSD(cov[h])
                rv = multivariate_normal(m[h], cov[h])
                plt.xlim(-3, 3)
                plt.ylim(-3, 3)
                plt.axes().set_aspect('equal', 'datalim')
                plt.contour(xx, yy, rv.pdf(pos))
                plt.xlabel("First PC", fontsize=15)
                plt.ylabel("Second PC", fontsize=15)
    def plot_statistics(self, Avg_Error, Avg_output, Avg_l2_weights, Avg_sum_variances):
        # This plots certain statistics regarding the training experience, if statistics = True in the constructor
        # Statistics such as: The trend of the error, the l2 norm of the weights, the l2 norm of the variances of the 
        # Covariance matrices, and the trend of the output of the network during training
        figure()
        plt.subplot(2, 2, 1)
        plt.plot(Avg_Error, 'r')
        plt.grid()
        plt.ylabel("Avg. Cost per Epoch")
        plt.xlabel("Epochs")
        plt.subplot(2, 2, 2)
        plt.plot(Avg_output)
        plt.grid()
        plt.ylabel("Avg. Output per Epoch")
        plt.xlabel("Epochs")
        plt.subplot(2, 2, 3)
        plt.plot(Avg_l2_weights)
        plt.grid()
        plt.ylabel("Avg. L-2 of Weights per Epoch")
        plt.xlabel("Epochs")
        plt.subplot(2, 2, 4)
        plt.grid()
        plt.plot(Avg_sum_variances)
        plt.ylabel("Avg. Sum of Variances per Epoch")
        plt.xlabel("Epochs")
        plt.show()
    def plot_decision_boundary(self, cov, m, W):
        # Plot the decision boundary if boundary == True, in the constructor
        plt.figure()
        x = np.arange(-5, 5, 0.1)
        y = np.arange(-5, 5, 0.1)
        xx, yy = np.meshgrid(x, y, sparse=True)
        output = []
        for i in range(xx.shape[1]):
            for j in range(yy.shape[0]):
                a = xx[0][i]
                b = yy[j][0]
                X = np.array([[a, b]])
                invcov = inv(cov)
                a = (X - m[:, np.newaxis])
                # # Optimised
                d = np.matmul(np.matmul(a, invcov), np.transpose(a, [0, 2, 1]))
                P = np.exp(-0.50 * np.diagonal(d, axis1=1, axis2=2))
                Z = np.sum((P.T * W).T, axis=0)
                output.append(1.7159 * np.tanh(float(2 / 3) * Z))
        output = np.reshape(np.array(output), (xx.shape[1], yy.shape[0]))
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.axes().set_aspect('equal', 'datalim')
        plt.contourf(x, y, output.T)
        plt.colorbar()
        plt.xlabel('First PC', fontsize=15)
        plt.ylabel('Second PC', fontsize=15)
        plt.show()
    def train(self, train_data, anomalous_data):
        input_num = train_data.shape[0]
        input_dim = train_data.shape[1]
        if (input_num % self.mini_batch) == 0:
            end = 0
        else:
            end = 1
        input_dim = train_data.shape[1]
        # Pre-training Stage
        [cov, m] = self.kmeans(train_data)
        # Initiating the Weights
        W = np.random.uniform(low=self.low, high=self.high, size=(self.H,))
        Avg_Error = []
        Avg_output = []
        Avg_sum_variances = []
        Avg_l2_weights = []

        for epoch in range(self.BP_epoch):

            if input_dim == 2 and epoch % 10 == 0:
                self.plot_gaussians(train_data, anomalous_data, m, cov)
            t0 = time.time()
            #Shuffle the received train_data
            random_index = np.arange(train_data.shape[0])
            np.random.shuffle(random_index)
            train_data = train_data[random_index]
            batch_counter = 0
            for i in range(int(input_num / self.mini_batch) + end):
                X = train_data[batch_counter: batch_counter + self.mini_batch, :]
                NUM = X.shape[0]
                [y, P, Z, a] = self.ebfdd_forward(X, m, cov, W)
                [dEdW, dEdCov, dEdM] = self.ebfdd_backward(y, P, Z, a, cov, W, NUM)
                # Update Rules
                W = W - self.BP_eta*dEdW/NUM
                m = m - self.BP_eta*dEdM/NUM
                cov = cov - self.BP_eta*dEdCov/NUM

                cost = np.mean(0.50*((1 - y) + self.beta*np.sum(np.diagonal(cov, axis1=1, axis2=2)**2) + self.theta*np.sum(W**2)))
                Avg_Error.append(cost)
                Avg_l2_weights.append(np.sum(W**2))
                Avg_output.append(np.mean(y))
                Avg_sum_variances.append(np.sum(np.diagonal(cov, axis1=1, axis2=2)))
                batch_counter = batch_counter + NUM
            t1 = time.time()
            print('Current Epoch: %d out of %d: %.2f Seconds' % (epoch+1, self.BP_epoch, (t1-t0)))

        if self.statistics:
            self.plot_statistics(Avg_Error, Avg_output, Avg_l2_weights, Avg_sum_variances)
        if self.boundary:
            self.plot_decision_boundary(cov, m, W)
        # ------------------------------------ Backward pass Ends------------------------------------
        # Learn the mean of a window size of choice o the output of the trained network only on the normal data
        batch_counter = 0
        trained_y = []
        for i in range(int(input_num / self.mini_batch) + end):
            X = train_data[batch_counter: batch_counter + self.mini_batch, :]
            NUM = X.shape[0]
            [y, _, _, _] = self.ebfdd_forward(X, m, cov, W)
            trained_y.append(y)
            batch_counter = batch_counter + NUM
        trained_y = np.concatenate(trained_y).ravel()
        return m, cov, W, trained_y
    def test(self, test_data, test_labels, m, cov, W):
        print("Test starts")
        input_num = test_data.shape[0]
        if (input_num % self.mini_batch) == 0:
            end = 0
        else:
            end = 1
        batch_counter = 0
        raw_output = []
        for i in range(int(input_num / self.mini_batch) + end):
            X = test_data[batch_counter: batch_counter + self.mini_batch, :]
            NUM = X.shape[0]
            [y, _, _, _] = self.ebfdd_forward(X, m, cov, W)
            raw_output.append(y)
            batch_counter = batch_counter + NUM
        raw_output = np.concatenate(raw_output).ravel()
        ground_truth = []
        for item in test_labels:
            # All vs 1
            if len(normal) == 1 and normal[0] == -1:
                if item in self.anomalous:
                    ground_truth.append(0)
                else:
                    ground_truth.append(1)
            # 1 vs All
            elif len(anomalous) == 1 and anomalous[0] == -1:
                if item in self.normal:
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)
            # some vs some
            else:
                if item in self.normal:
                    ground_truth.append(1)
                elif item in self.anomalous:
                    ground_truth.append(0)
        ground_truth = np.array(ground_truth)
        return raw_output, ground_truth
class RBFDD:
    def __init__(self, dataset_name, mini_batch_size,  H, beta, theta, bp_eta, bp_epoch, normal, anomalous):
        self.H = H
        self.beta = beta
        self.theta = theta
        self.BP_eta = bp_eta
        self.BP_epoch = bp_epoch
        self.normal = normal
        self.anomalous = anomalous
        self.dataset_name = dataset_name
        self.mini_batch = mini_batch_size
        self.low = float(-0.001) / (math.sqrt(self.H))
        self.high = float(0.001) / (math.sqrt(self.H))
    def nearPSD(self, A, epsilon=1e-23):
        n = A.shape[0]
        eigval, eigvec = np.linalg.eig(A)
        val = np.matrix(np.maximum(eigval, epsilon))
        vec = np.matrix(eigvec)
        T = 1 / (np.multiply(vec, vec) * val.T)
        T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
        B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
        out = B * B.T
        return (out)
    def clip_power(self, power):
        power[np.where(power < -1e+1000)] = -1000
        # power[np.where(np.abs(power) < 1e-10)] = 0
        power[np.where(power > 709)] = 709
        return power
    def train(self, train_data):
        #-----------------------------------------Scikit Kmeans starts-------------------------------------
        input_num = train_data.shape[0]
        if (input_num % self.mini_batch) == 0:
            end = 0
        else:
            end = 1
        input_dim = train_data.shape[1]
        try:
            kmeans = KMeans(n_clusters=self.H, random_state=0).fit(train_data)
            print("K-means is DONE!!!")
            m = kmeans.cluster_centers_
            labels = np.array(kmeans.labels_)
        except Exception as exp:
            print(exp)
        sd = []
        for h in range(self.H):
            index = np.argwhere(labels == h)
            data = train_data[np.reshape(index, (index.shape[0],))]
            distance = (np.sum((data - m[h])**2, axis=1))**(0.50)
            maximum_distance = np.max(distance)
            if maximum_distance > 0:
                sd.append(maximum_distance / float(3))
            else:
                sd.append(sys.float_info.epsilon)
        sd = np.array(sd).reshape(self.H,1)
        #-------------------------------------------------- Scikit Kmeans Ends---------------------------------
        # Initialize the weight matrix before RBF layer and the output layer
        W = np.random.uniform(low=self.low, high=self.high, size=(self.H,))
        # -------------------------------------------------Training Error Progress----------------------------------
        Avg_Error = []
        Avg_output = []
        Avg_sum_variances = []
        Avg_l2_weights = []
        xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
        pos = np.dstack((xx, yy))
        counter = 1
        # Let the BP on RBFDD begin
        for epoch in range(self.BP_epoch):

            if input_dim == 2:
                # convert sd values into equi-variance covariance matrices so we can visualize stuff
                cov = []
                for h in range(sd.shape[0]):
                    cov.append(sd[h]*np.identity(input_dim))
                cov = np.array(cov)
                if epoch % 1 == 0:
                    # plt.subplot(2, int(self.BP_epoch / 5) + 1, counter)
                    figure()
                    plt.grid()
                    plt.scatter(train_data[:, 0], train_data[:, 1], c="r",
                                alpha=0.1)
                    plt.xlabel("First PC", fontsize=15)
                    plt.ylabel("Second PC", fontsize=15)
                    for h in range(self.H):
                        try:
                            rv = multivariate_normal(m[h], cov[h])
                            plt.contour(xx, yy, rv.pdf(pos))
                            plt.xlabel("First PC", fontsize=15)
                            plt.ylabel("Second PC", fontsize=15)
                        except ValueError:

                            cov[h] = self.nearPSD(cov[h])
                            rv = multivariate_normal(m[h], cov[h])
                            plt.contour(xx, yy, rv.pdf(pos))
                            plt.xlabel("First PC", fontsize=15)
                            plt.ylabel("Second PC", fontsize=15)
                    counter = counter + 1
                    # plt.show()
            t0 = time.time()
            # Shuffle the received train_data
            random_index = np.arange(train_data.shape[0])
            np.random.shuffle(random_index)
            train_data = train_data[random_index]
            batch_counter = 0

            for i in range(int(input_num / self.mini_batch) + end):
                X = train_data[batch_counter: batch_counter + self.mini_batch, :]
                NUM = X.shape[0]
                #------------------------------------ Forward pass Starts-------------------------------------
                var = sd**2
                a = (X - m[:, np.newaxis])
                # b is Hxn and for every Gaussian we have n likelihoods
                Dist = np.sum(a**2, axis=2)
                power = np.divide(-0.50*Dist, var)
                power = self.clip_power(power)
                P = np.exp(power)
                # Z(n) is what is what the output neuron has received
                Z = np.sum((P.T * W).T, axis=0)
                # Here we apply the Lecun's recommended tanh() function to get the outputs
                # The output vector has n number of elements
                y = 1.7159 * np.tanh(float(2 / 3) * Z)
                if np.isnan(y).any() == True:
                    print(y)

                # ------------------------------------ Backward pass Starts------------------------------------
                dEdZ = (y - 1) * (1.1439 * (1 - np.square(np.tanh(float(2 / 3) * Z))))
                # We will have to have a ROW-WIZE multiplication of dEdZ upon P(Hxn) (i.e., P*dEdZ), so that every element of dEdZ
                # is multiplied by its relevant likelihood (# of elements in each row of P), across ALL kernels (# of Rows in P)
                # Then a sum across all the rows will result in a vector of size (H,), which is exactly the dEdW!!!
                dEdW = np.sum(P * dEdZ, axis=1) + NUM * self.theta * W
                # We do NOT update just now! This is Batch-Learning!
                # --------------------------Let's compute the Centroid derivatives-------------------------
                # First we will take the derivatives right uo to the Kernels' boubdaries (i.e., dEdP)
                # We already have dEdZ, which is the common term in all our update rules. So, let's now
                # Compute dZdP, which is a matrix the same size as P (H,n), which is nothing but a matrix
                # of size (H,n), whose columns are exact replications of the weight vector W(H,).
                dZdP = np.repeat([W], NUM, axis=0).T
                # Now, having both dEdZ(n,) and dZdP(H,n) we can compute dEdP. We need every element of dEdZ
                # To be multiplied by its corresponding column in the dZdP(H,n) matrix. np.multiply() does that!
                dEdP = np.multiply(dEdZ, dZdP) * P

                dPdM = np.divide(a,var[:, np.newaxis])
                dEdM = np.matmul(dEdP[:,None],dPdM).reshape(self.H, input_dim)

                dPdsd = np.divide(Dist, sd**3)
                dEdsd = np.reshape(np.sum(dEdP*dPdsd, axis=1), (self.H, 1)) + self.beta*NUM*sd

                # Update the parameters
                # Update Rules
                W = W - self.BP_eta * dEdW/NUM
                m = m - self.BP_eta * dEdM/NUM
                sd = sd - self.BP_eta * dEdsd/NUM

                cost = 0.50*(np.mean(np.square(1-y)) + self.beta*(np.sum(sd**2)) + self.theta*(np.sum(W**2)))
                Avg_Error.append(cost)
                Avg_l2_weights.append(np.sum(W ** 2))
                Avg_output.append(np.mean(y))
                Avg_sum_variances.append(np.sum(sd**2))
                batch_counter = batch_counter + NUM
            t1 = time.time()
            print('Current Epoch: %d out of %d: %.5f Seconds' % (epoch + 1, self.BP_epoch, (t1 - t0)))
            # ------------------------------------ Backward pass Ends------------------------------------
        ###################################################Visualisations##########################################
        if input_dim == 2:
            figure()
            plt.subplot(2, 2, 1)
            plt.plot(Avg_Error, 'r')
            plt.grid()
            plt.ylabel("Avg. Cost per Epoch")
            plt.xlabel("Epochs")
            plt.subplot(2, 2, 2)
            plt.plot(Avg_output)
            plt.grid()
            plt.ylabel("Avg. Output per Epoch")
            plt.xlabel("Epochs")
            plt.subplot(2, 2, 3)
            plt.plot(Avg_l2_weights)
            plt.grid()
            plt.ylabel("Avg. L-2 of Weights per Epoch")
            plt.xlabel("Epochs")
            plt.subplot(2, 2, 4)
            plt.grid()
            plt.plot(Avg_sum_variances)
            plt.ylabel("Avg. Sum of Variances per Epoch")
            plt.xlabel("Epochs")
            # Plot the decision boundary
            plt.figure()
            x = np.arange(-5, 5, 0.1)
            y = np.arange(-5, 5, 0.1)
            xx, yy = np.meshgrid(x, y, sparse=True)
            output = []
            for i in range(xx.shape[1]):
                for j in range(yy.shape[0]):
                    a = xx[0][i]
                    b = yy[j][0]
                    X = np.array([[a, b]])
                    invcov = inv(cov)
                    a = (X - m[:, np.newaxis])

                    # # Optimised
                    d = np.matmul(np.matmul(a, invcov), np.transpose(a, [0, 2, 1]))
                    #################
                    # Now the array P(Hxn) has ALL the likelihoods of ALL the training data of ALL the Kernels
                    P = np.exp(-0.50 * np.diagonal(d, axis1=1, axis2=2))
                    # Here we have a column-wise multiplication between P(Hxn) and Weights (CAN be BETTER CODED)
                    # Z(n) is what is what the output neuron has received
                    Z = np.sum((P.T * W).T, axis=0)
                    # Here we apply the Lecun's recommended tanh() function to get the outputs
                    # The output vector has n number of elements
                    output.append(1.7159 * np.tanh(float(2 / 3) * Z))
            output = np.reshape(np.array(output), (xx.shape[1], yy.shape[0]))
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.axes().set_aspect('equal', 'datalim')
            plt.contourf(x, y, output.T)
            plt.colorbar()
            plt.xlabel('First PC', fontsize=15)
            plt.ylabel('Second PC', fontsize=15)
            plt.show()
        # Learn the mean of a window size of choice o the output of the trained network only on the normal data
        var = sd ** 2
        a = (train_data - m[:, np.newaxis])
        # b is Hxn and for every Gaussian we have n likelihoods
        Dist = np.sum(a ** 2, axis=2)
        power = np.divide(-0.50 * Dist, var)
        power = self.clip_power(power)
        P = np.exp(power)
        # Z(n) is what is what the output neuron has received
        Z = np.sum((P.T * W).T, axis=0)
        # Here we apply the Lecun's recommended tanh() function to get the outputs
        # The output vector has n number of elements
        trained_y = 1.7159 * np.tanh(float(2 / 3) * Z)
        return m, sd, W, trained_y
    def test(self,test_data, test_labels, m, sd, W):
        input_num = test_data.shape[0]
        if (input_num % self.mini_batch) == 0:
            end = 0
        else:
            end = 1
        input_dim = test_data.shape[1]
        batch_counter = 0
        raw_output = []
        for i in range(int(input_num / self.mini_batch) + end):
            X = test_data[batch_counter: batch_counter + self.mini_batch, :]
            NUM = X.shape[0]
            var = sd ** 2
            a = (X - m[:, np.newaxis])
            # b is Hxn and for every Gaussian we have n likelihoods
            Dist = np.sum(a ** 2, axis=2)
            power = np.divide(-0.50 * Dist, var)
            power = self.clip_power(power)
            P = np.exp(power)
            # Z(n) is what is what the output neuron has received
            Z = np.sum((P.T * W).T, axis=0)
            # Here we apply the Lecun's recommended tanh() function to get the outputs
            # The output vector has n number of elements
            raw_output.append(1.7159 * np.tanh(float(2 / 3) * Z))
            batch_counter = batch_counter + NUM
        raw_output = np.concatenate(raw_output).ravel()

        ground_truth = []
        for item in test_labels:
            # All vs 1
            if len(normal) == 1 and normal[0] == -1:
                if item in self.anomalous:
                    ground_truth.append(0)
                else:
                    ground_truth.append(1)
            # 1 vs All
            elif len(anomalous) == 1 and anomalous[0] == -1:
                if item in self.normal:
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)
            # some vs some
            else:
                if item in self.normal:
                    ground_truth.append(1)
                elif item in self.anomalous:
                    ground_truth.append(0)

        ground_truth = np.array(ground_truth)
        return raw_output, ground_truth
class OCSVM:
    def __init__(self, dataset_name, nu, gamma, normal, anomalous):
        self.normal = normal
        self.anomalous = anomalous
        self.dataset_name = dataset_name
        self.nu = nu
        self.gamma = gamma
    def train(self, training_data):
        clf = svm.OneClassSVM(nu=self.nu, kernel="rbf", gamma=self.gamma)
        # train OCSVM
        clf.fit(training_data)
        # Measure the distance of the training_data from the learned hyper-plain
        train_dist = clf.decision_function(training_data)
        # Plot the decision boundary
        fig = plt.figure()
        x = np.arange(-5, 5, 0.1)
        y = np.arange(-5, 5, 0.1)

        xx, yy = np.meshgrid(x, y, sparse=True)
        output = []
        for i in range(xx.shape[1]):
            for j in range(yy.shape[0]):
                a = xx[0][i]
                b = yy[j][0]
                c = np.array([[a, b]])
                output.append(clf.decision_function(c))
        output = np.reshape(np.array(output), (xx.shape[1], yy.shape[0]))
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.axes().set_aspect('equal', 'datalim')
        plt.contourf(x, y, output.T)
        plt.colorbar()
        plt.xlabel('First PC', fontsize=15)
        plt.ylabel('Second PC', fontsize=15)
        plt.show()
        return clf, train_dist
    def test(self, test_data, test_labels, model):
        # In order to compute the accuracy and macro-f-measure
        raw_output = model.decision_function(test_data)
        raw_output = np.reshape(raw_output, (raw_output.shape[0],))
        # Compute Predictions and Ground-truth
        prediction = model.predict(test_data)
        ground_truth = []
        for item in test_labels:
            # All vs 1
            if len(normal) == 1 and normal[0] == -1:
                if item in self.anomalous:
                    ground_truth.append(0)
                else:
                    ground_truth.append(1)
            # 1 vs All
            elif len(anomalous) == 1 and anomalous[0] == -1:
                if item in self.normal:
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)
            # some vs some
            else:
                if item in self.normal:
                    ground_truth.append(1)
                elif item in self.anomalous:
                    ground_truth.append(0)

        ground_truth = np.array(ground_truth)
        return raw_output, ground_truth, prediction
class AEN:
    def __init__(self, dataset_name, mini_batchsize, n_components, bp_eta,  encoding_dim, encoding_activation, final_layer_activation, max_epoch, normal, anomalous, error_function):
        self.dataset_name = dataset_name
        self.n_components = n_components
        self.encoding_dim = encoding_dim
        self.encoding_activation = encoding_activation
        self.final_layer_activation = final_layer_activation
        self.max_epoch = max_epoch
        self.normal = normal
        self.anomalous = anomalous
        self.loss = error_function
        self.mini_batch = mini_batchsize
        self.bp_eta = bp_eta
    def train(self, training_data):
        # Build the model
        # this is our input placeholder
        input_data = Input(shape=(self.n_components,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(self.encoding_dim, activation=self.encoding_activation)(input_data)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(self.n_components, activation=self.final_layer_activation)(encoded)
        # this model maps an input to its reconstruction
        autoencoder = Model(input_data, decoded)
        # this model maps an input to its encoded representation
        encoder = Model(input_data, encoded)
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        adam = optimizers.adam(lr=self.bp_eta)
        autoencoder.compile(optimizer=adam, loss=self.loss)

        # Let the actual training start
        autoencoder.fit(training_data, training_data, epochs=self.max_epoch,
                              batch_size=self.mini_batch, shuffle=True, verbose=False)

        # Before going through the actual testing, gather the errors for the trained model on the training set
        # Learn the mean of a window size of choice o the output of the trained network only on the normal data
        encoded_data = encoder.predict(training_data)
        reconstructed = decoder.predict(encoded_data)
        trained_Normal_E = np.subtract(training_data, reconstructed) ** 2
        trained_Normal_E = np.sum(trained_Normal_E, axis=1)
        # Plot the decision boundary
        plt.figure()
        x = np.arange(-100, 100, 1)
        y = np.arange(-100, 100, 1)
        xx, yy = np.meshgrid(x, y, sparse=True)
        output = []
        for i in range(xx.shape[1]):
            for j in range(yy.shape[0]):
                a = xx[0][i]
                b = yy[j][0]
                c = np.array([[a, b]])
                encoded_data = encoder.predict(c)
                predict = decoder.predict(encoded_data)
                trained_Normal_E = np.square(np.subtract(c, predict))
                trained_Normal_E = np.sum(trained_Normal_E, axis=1)
                output.append(-trained_Normal_E)
        output = np.reshape(np.array(output), (xx.shape[1], yy.shape[0]))
        # plt.xlim(-160, 160)
        # plt.ylim(-160, 160)
        # plt.axes().set_aspect('equal', 'datalim')
        plt.contourf(x, y, output.T)
        plt.colorbar()
        plt.xlabel('First PC', fontsize=15)
        plt.ylabel('Second PC', fontsize=15)
        plt.show()

        return encoder, decoder, training_data, reconstructed, trained_Normal_E
    def test(self, test_data, test_labels, encoder, decoder):

        input_num = test_data.shape[0]
        batch_counter = 0
        raw_error = []
        for i in range(int(input_num / self.mini_batch) + 1):
            X = test_data[batch_counter: batch_counter + self.mini_batch, :]
            NUM = X.shape[0]
            encoded = encoder.predict(X)
            decoded = decoder.predict(encoded)
            raw_error.append(np.sum(np.subtract(X, decoded) ** 2, axis=1))
            batch_counter = batch_counter + NUM
        # we will save raw_error, but we will also
        raw_error = np.concatenate(raw_error).ravel()

        # encoded = encoder.predict(test_data)
        # decoded = decoder.predict(encoded)

        # Compute the error
        # raw_error = np.sum(np.subtract(test_data, decoded) ** 2, axis=1)
        ## Compute Predictions and Ground-truth
        ground_truth = []
        for item in test_labels:
            # All vs 1
            if len(normal) == 1 and normal[0] == -1:
                if item in self.anomalous:
                    ground_truth.append(0)
                else:
                    ground_truth.append(1)
            # 1 vs All
            elif len(anomalous) == 1 and anomalous[0] == -1:
                if item in self.normal:
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)
            # some vs some
            else:
                if item in self.normal:
                    ground_truth.append(1)
                elif item in self.anomalous:
                    ground_truth.append(0)

        return raw_error, ground_truth
class GMM:
    def __init__(self, dataset_name, n_components, normal, anomalous):
        self.dataset_name = dataset_name
        self.H = n_components
        self.normal = normal
        self.anomalous = anomalous
    def plot_GMMs(self, GMM, train_data):
        # Just for the sake of Visualisation let's run a few lines here
        splot = matplotlib.pyplot.subplot(1, 1, 1)
        Y_ = GMM.predict(train_data)
        for i, (mean, cov) in enumerate(zip(GMM.means_, GMM.covariances_)):
            v, w = linalg.eigh(cov)
            if not np.any(Y_ == i):
                continue
            plt.scatter(train_data[Y_ == i, 0], train_data[Y_ == i, 1], .8)
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180. * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(.5)
            splot.add_artist(ell)
        plt.xticks(())
        plt.yticks(())
        plt.title('Selected GMM: Full Covariance, 2 components')
        plt.subplots_adjust(hspace=.35, bottom=.02)
        plt.show()
    def train(self, train_data):
        # Fit the GMM to the training data
        gmm = mixture.GaussianMixture(n_components=self.H, covariance_type='full')
        # train OCSVM
        gmm.fit(train_data)
        # One last run through the training data to get the average likelihoods
        trained_prob = gmm.score_samples(train_data)
        # self.plot_GMMs(gmm, train_data)
        if train_data.shape[1] == 2:
            xx, yy = np.mgrid[-5:5:.1, -5:5:.1]
            pos = np.dstack((xx, yy))
            m = gmm.means_
            cov = gmm.covariances_
            figure()
            plt.grid()
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.scatter(train_data[:, 0], train_data[:, 1], c="r", alpha=0.1)
            for h in range(self.H):
                rv = multivariate_normal(m[h], cov[h])
                plt.xlim(-3, 3)
                plt.ylim(-3, 3)
                plt.axes().set_aspect('equal', 'datalim')
                plt.contour(xx, yy, rv.pdf(pos))
                plt.xlabel("First PC", fontsize=15)
                plt.ylabel("Second PC", fontsize=15)
            plt.show()
            # Plot the decision boundary
            plt.figure()
            x = np.arange(-5, 5, 0.1)
            y = np.arange(-5, 5, 0.1)
            xx, yy = np.meshgrid(x, y, sparse=True)
            output = []
            for i in range(xx.shape[1]):
                for j in range(yy.shape[0]):
                    a = xx[0][i]
                    b = yy[j][0]
                    X = np.array([[a, b]])
                    output.append(np.exp(gmm.score_samples(X)))
            output = np.reshape(np.array(output), (xx.shape[1], yy.shape[0]))
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.axes().set_aspect('equal', 'datalim')
            plt.contourf(x, y, output.T)
            plt.colorbar()
            plt.xlabel('First PC', fontsize=15)
            plt.ylabel('Second PC', fontsize=15)
            plt.show()

        return gmm, trained_prob
    def test(self, test_data, test_labels, gmm):
        raw_output = gmm.score_samples(test_data)
        prediction = gmm.predict(test_data)
        ground_truth = []
        for item in test_labels:
            # All vs 1
            if len(normal) == 1 and normal[0] == -1:
                if item in self.anomalous:
                    ground_truth.append(0)
                else:
                    ground_truth.append(1)
            # 1 vs All
            elif len(anomalous) == 1 and anomalous[0] == -1:
                if item in self.normal:
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)
            # some vs some
            else:
                if item in self.normal:
                    ground_truth.append(1)
                elif item in self.anomalous:
                    ground_truth.append(0)

        ground_truth = np.array(ground_truth)
        return raw_output, ground_truth, prediction
class iForest:
    def __init__(self, dataset_name, n_estimators, normal, anomalous):
        self.dataset_name = dataset_name
        self.n_estimators = n_estimators
        self.normal = normal
        self.anomalous = anomalous
    def train(self, train_data):
        # Create the iForest object once and train it through mini-batches
        clf = IsolationForest(n_estimators=self.n_estimators, contamination=0.)
        clf.fit(train_data)
        # Plot the scores
        trained_scores = clf.decision_function(train_data)
        if train_data.shape[1] == 2:
            # Plot the decision boundary
            plt.figure()
            x = np.arange(-5, 5, 0.1)
            y = np.arange(-5, 5, 0.1)
            xx, yy = np.meshgrid(x, y, sparse=True)
            output = []
            for i in range(xx.shape[1]):
                for j in range(yy.shape[0]):
                    a = xx[0][i]
                    b = yy[j][0]
                    X = np.array([[a, b]])
                    output.append(clf.decision_function(X))
            output = np.reshape(np.array(output), (xx.shape[1], yy.shape[0]))
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.axes().set_aspect('equal', 'datalim')
            plt.contourf(x, y, output.T)
            plt.colorbar()
            plt.xlabel('First PC', fontsize=15)
            plt.ylabel('Second PC', fontsize=15)
            plt.show()

        return clf, trained_scores
    def test(self, test_data, test_labels, clf):
        # Raw anomaly scores
        raw_output = clf.decision_function(test_data)
        # Predictions
        prediction = clf.predict(test_data)
        # plt.hist(train_scores, fc=(0, 1, 0, 0.5))
        # plt.hist(y, fc=(1, 0, 0, 0.5))
        # plt.axvline(threshold, c='k')
        # plt.show()

        # Compute Predictions and Ground-truth
        ground_truth = []
        for item in test_labels:
            # All vs 1
            if len(normal) == 1 and normal[0] == -1:
                if item in self.anomalous:
                    ground_truth.append(0)
                else:
                    ground_truth.append(1)
            # 1 vs All
            elif len(anomalous) == 1 and anomalous[0] == -1:
                if item in self.normal:
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)
            # some vs some
            else:
                if item in self.normal:
                    ground_truth.append(1)
                elif item in self.anomalous:
                    ground_truth.append(0)

        ground_truth = np.array(ground_truth)

        return raw_output, ground_truth, prediction

# we choose a dataset on which the following will be applied to prepare the data for our algorithm
# options are: MNIST, FASHION-MNIST, WISCONSIN, IONOSPHERE, KDD

# The structure goes: "DatasetName": [[[Normal], [Anomalous]], ... Until we finish all the scenarios]
#{LANDSAT:[[[1], [-1]],[[2], [-1]],[[3], [-1]],[[4], [-1]],[[5], [-1]],[[7], [-1]]]}
#{"PAGE": [[[1], [-1]],[[2], [-1]],[[3], [-1]],[[4], [-1]],[[5], [-1]]]}
#{'WAVE':[[[0],[-1]],[[1],[-1]],[[2],[-1]]]}
# #{"FAULT": [[[7], [-1]]],[[2], [-1]],[[3], [-1]],[[4], [-1]],[[5], [-1]],[[6], [-1]],[[7], [-1]]]}#"GAMMA":[[['g'], ['h']]]"SPAM-BASE":[[[0], [1]]] "PARTICLE":[[[1], [0]]]"SKIN":[[[2], [1]]],#"GAMMA":[[['g'], ['h']]]}, "MNIST":[[[1], [0]]],"WISCONSIN": [[[2], [4]]]}#"KDD": [[['normal.'], ['satan.']],[['normal.'], ['smurf.']]]}
dataset_collaction = {"LANDSAT":[[[1], [-1]]]}#, [[1],[-1]], [[2],[-1]]]}#,[['B'], [-1]],[['C'], [-1]],[['D'], [-1]],[['E'], [-1]],[['F'], [-1]],[['G'], [-1]],[['H'], [-1]],[['I'], [-1]],[['J'], [-1]],[['K'], [-1]],[['L'], [-1]],[['M'], [-1]],[['N'], [-1]],[['O'], [-1]],[['P'], [-1]],[['Q'], [-1]],[['R'], [-1]],[['S'], [-1]],[['T'], [-1]],[['U'], [-1]],[['V'], [-1]],[['W'], [-1]],[['X'], [-1]],[['Y'], [-1]],[['Z'], [-1]]]}#,[[-1], ["GRASS"]],[[-1],["SKY"]],[[-1],["FOLIAGE"]],[[-1],["CEMENT"]],[[-1],["WINDOW"]],[[-1],["PATH"]],[[-1],["GRASS"]]]}#{"FAULT": [[[-1], [1]],[[-1], [2]],[[-1], [3]],[[-1], [4]],[[-1], [5]],[[-1], [6]],[[-1], [7]]]}#"SKIN":[[[2], [1]]] "MNIST":[[[1], [0]]]}#"SPAM-BASE":[[[0], [1]]]}#"GAMMA":[[['g'], ['h']]]}#,[["SKY"], [-1]],[["FOLIAGE"], [-1]],[["CEMENT"], [-1]],[["WINDOW"], [-1]],[["PATH"], [-1]],[["GRASS"], [-1]]]}#,[[-1], [2]],[[-1], [3]],[[-1], [4]],[[-1], [5]],[[-1], [6]],[[-1], [7]]]}#,[[2], [-1]],[[3], [-1]],[[4], [-1]],[[5], [-1]],[[6], [-1]],[[7], [-1]]]}#{"IMAGESEGMENTATION": [[["BRICKFACE"], [-1]],[["SKY"], [-1]],[["FOLIAGE"], [-1]],[["CEMENT"], [-1]],[["WINDOW"], [-1]],[["PATH"], [-1]],[["GRASS"], [-1]]]}
algorithm_collection = ['EBFDD']#['EBFDD','RBFDD','AEN', 'OCSVM','GMM','iForest']
# Determine the number of hidden nodes for both the RBFDD and AEN
# for MNIST
# EBFDD: 10,20,0.001, 0.9, 0.001

H = [5]
MAX_EPOCH =[20]

BP_ETA = [0.01]#, 0.001, 0.0001]
BETA = [0.1]#, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]
THETA = [0.01]#, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]
Nu = [0.001]#[0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
Gamma = [0.9]#[0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
N_ESTIMATORS =[1000]# [100, 200, 500, 800, 1000]
mini_batch = 32
SAMPLING_TIMES = 10

# Do we want  to reduce the dimensionality of the data
compress = False
compress_dim = 2
# Training sample size
sample_size = 0.80

for algorithm in algorithm_collection:
    # Go through all datasets, and all their scenarios
    for dataset in dataset_collaction.keys():
        for scenario in dataset_collaction[dataset]:
            # Deternine current scenario by assigning current normal, and anomalous so the datasets can be built
            normal = scenario[0]
            anomalous = scenario[1]
            # Now prepare the corresponding dataset, accordingly
            if dataset == 'GAMMA':
                [normal_data, normal_data_label, anomalous_data,anomalous_data_label] = gamma_Script.prepare_gamma(normal, anomalous)
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print('Gamma data is loaded...' + '\n' + "(N, A)=%s" % str(scenario))
            elif dataset == 'SKIN':
                # get all the data
                [normal_data, normal_data_label, anomalous_data, anomalous_data_label] = Skin_Script.prepare_skin(normal, anomalous)
                # for the sake of AEN, record the input dimension, which helps with choosing the encode_dim hyper parameter for the AEN
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print("SKIN loaded..."+'\n'+"(N, A)=%s" % str(scenario))
            elif dataset == 'SPAM-BASE':
                # get all the data
                [normal_data, normal_data_label, anomalous_data, anomalous_data_label] = Spambase_Script.prepare_Spambase(normal, anomalous)
                # for the sake of AEN, record the input dimension, which helps with choosing the encode_dim hyper parameter for the AEN
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print("SKIN loaded..."+'\n'+"(N, A)=%s" % str(scenario))
            elif dataset == 'FAULT':
                [normal_data, normal_data_label, anomalous_data, anomalous_data_label] = Fault_Script.prepare_fault(
                    normal, anomalous)
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print('Fault data is loaded...' + '\n' + "(N, A)=%s" % str(scenario))
            elif dataset == 'IMAGESEGMENTATION':
                [normal_data, normal_data_label, anomalous_data, anomalous_data_label] = ImageSegmentation_Script.prepare_imagesegmentation(
                    normal, anomalous)
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print('Fault data is loaded...' + '\n' + "(N, A)=%s" % str(scenario))
            elif dataset == 'LANDSAT':
                [normal_data, normal_data_label, anomalous_data, anomalous_data_label] = landsat_Script.prepare_landsat(
                    normal, anomalous)
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print('LANDSAT data is loaded...' + '\n' + "(N, A)=%s" % str(scenario))
            elif dataset == 'LETTER':
                [normal_data, normal_data_label, anomalous_data, anomalous_data_label] = letter_Script.prepare_letter(
                    normal, anomalous)
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print('Letter data is loaded...' + '\n' + "(N, A)=%s" % str(scenario))
            elif dataset == 'PAGE':
                [normal_data, normal_data_label, anomalous_data, anomalous_data_label] = Page_Script.prepare_page(
                    normal, anomalous)
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print('PAGE data is loaded...' + '\n' + "(N, A)=%s" % str(scenario))
            elif dataset == 'WAVE':
                [normal_data, normal_data_label, anomalous_data, anomalous_data_label] = wave_Script.prepare_wave(
                    normal, anomalous)
                input_dim = normal_data.shape[1]
                print("===============================================================")
                print('PAGE data is loaded...' + '\n' + "(N, A)=%s" % str(scenario))

            # If you need to compress your data, this bit will be executed
            if compress == True:
                normal_data, anomalous_data = PCA_compress(compress_dim, normal_data, anomalous_data)
                input_dim = compress_dim

            # define a dictionary for saving everything
            final_result = dict()
            # Set the desired set of hyper-parameters for the given algorithms
            if algorithm == "EBFDD":
                # Deviation used for detecting the threshold. Deviation from the average_y
                # define a dictionary for saving everything
                final_result = dict()
                # in the following section we will go through different combinations(i.e., n) of hyper-parameters
                # in order to know where we are in the middle of the execution
                round_counter = 1
                total_number_rounds = len(H)*len(BP_ETA)*len(MAX_EPOCH)*len(BETA)*len(THETA)
                for h in H:
                    for bp_epoch in MAX_EPOCH:
                        for bp_eta in BP_ETA:
                            for beta in BETA:
                                for theta in THETA:
                                    current_hyper_parameters = [h, bp_epoch, bp_eta, beta, theta]
                                    # this list wholse the information regarding each sampling round for a particular hyper-parameter combination
                                    step_result = []
                                    for sampleing_times in range(SAMPLING_TIMES):
                                        print("Current Algorithm: " + algorithm +'\n'+
                                              'Dataset being sampled: ' + dataset + ' Scenario: ' + str(scenario)+'\n'+
                                              'Current Round: %d ' % round_counter + 'Out of: %d' % total_number_rounds + ' Rounds'+'\n'+
                                              'Current Hyper-Parameters: (H, BP_epoch, BP_eta, beta(spreads), theta(weights)): (%d, %d, %.5f, %.5f, %.5f)' %
                                              (h, bp_epoch, bp_eta, beta, theta)+'\n'+ 'Sample Number: %d ' % (sampleing_times + 1) + 'Out of: %d' % SAMPLING_TIMES + ' Sampling Rounds'+'\n')
                                        # Prepare Training Data
                                        boot_strap_train_data, boot_strap_train_index = prepare_training_data(normal_data, sample_size)
                                        # Prepare the test data
                                        concatenated_test_data, concatenated_test_label = prepare_testing_data(normal_data, normal_data_label, boot_strap_train_index)
                                        # Learner Creation
                                        ebfdd = EBFDD(dataset, mini_batch,h, beta, theta, bp_eta, bp_epoch,  normal, anomalous)
                                        # Train the Learner
                                        [m, cov, W, trained_y] = ebfdd.train(boot_strap_train_data, anomalous_data)
                                        # Test the learner
                                        [raw_output, ground_truth] = ebfdd.test(concatenated_test_data,concatenated_test_label, m, cov, W)
                                        # Save results of this round
                                        step_result.append([raw_output, ground_truth, m, cov, W, trained_y])
                                    # Increment the round counter
                                    round_counter = round_counter + 1
                                    # Before changing the hyper-parameters, store step_result, into the final_result dictionary
                                    final_result[str(current_hyper_parameters)] = step_result
            elif algorithm == "RBFDD":
                # define a dictionary for saving everything
                final_result = dict()
                # in the following section we will go through different combinations(i.e., n) of hyper-parameters
                # in order to know where we are in the middle of the execution
                round_counter = 1
                total_number_rounds = len(H)*len(BP_ETA)*len(MAX_EPOCH)*len(BETA)*len(THETA)
                for h in H:
                    for bp_epoch in MAX_EPOCH:
                        for bp_eta in BP_ETA:
                            for beta in BETA:
                                for theta in THETA:
                                    current_hyper_parameters = [h, bp_epoch, bp_eta, beta, theta]
                                    # this list wholse the information regarding each sampling round for a particular hyper-parameter combination
                                    step_result = []
                                    for sampleing_times in range(SAMPLING_TIMES):
                                        print("Current Algorithm: " + algorithm)
                                        print('Dataset being sampled: ' + dataset + ' Scenario: ' + str(scenario))
                                        print(
                                            'Current Round: %d ' % round_counter + 'Out of: %d' % total_number_rounds + ' Rounds')
                                        print(
                                            'Current Hyper-Parameters: (H, BP_epoch, BP_eta, beta(spreads), theta(weights)): (%d, %d, %.5f, %.5f, %.5f)' %
                                            (h, bp_epoch, bp_eta, beta, theta))
                                        print('Sample Number: %d ' % (
                                        sampleing_times + 1) + 'Out of: %d' % SAMPLING_TIMES + ' Sampling Rounds')
                                        print("\n")
                                        # create an RBFDD object right here
                                        rbfdd = RBFDD(dataset, mini_batch,h, beta, theta, bp_eta, bp_epoch, normal, anomalous)
                                        # bootstrap sample from the train_data
                                        boot_strap_size = int(normal_data.shape[0] * 0.80)
                                        # Generate a uniform random sample from np.arange(boot_strap_size) of size boot_strap_size:
                                        # generate the index of the sampled values NO REPLACEMENT
                                        boot_strap_train_index = np.random.choice(normal_data.shape[0], boot_strap_size, replace=False)
                                        # extract the associated boot_strap_train data for the RBFDD training (No need for labels)
                                        boot_strap_train_data = normal_data[boot_strap_train_index]
                                        # Call the train function of the rbfdd using the boot_strap_train_data
                                        # Get the learned parameters
                                        [m, sd, W, trained_y] = rbfdd.train(boot_strap_train_data)
                                        # Find the remaining indices in normal_data and add them to the anomalous data
                                        # test_data_normal_portion = []
                                        # test_label_normal_portion = []
                                        # for index in range(normal_data.shape[0]):
                                        #     if index not in boot_strap_train_index:
                                        #         test_data_normal_portion.append(normal_data[index])
                                        #         test_label_normal_portion.append(normal_data_label[index])
                                        # test_data_normal_portion = np.array(test_data_normal_portion)
                                        test_data_normal_portion = np.delete(normal_data, boot_strap_train_index,
                                                                             axis=0)
                                        test_label_normal_portion = np.delete(normal_data_label, boot_strap_train_index,
                                                                              axis=0)
                                        test_label_normal_portion = np.array(test_label_normal_portion)
                                        # Now concatenate these newly extracted normal samples to the existing
                                        # anomalous data to create the the final test set for the prediction
                                        concatenated_test_data = np.concatenate((test_data_normal_portion, anomalous_data), axis=0)
                                        concatenated_test_label = np.concatenate((test_label_normal_portion, anomalous_data_label), axis=0)
                                        # Before passing the test data, let's shuffle it to avoid ANY STREAMIMG assumption
                                        concatenated_test_data, concatenated_test_label = shuffle(concatenated_test_data, concatenated_test_label, random_state=0)
                                        # test the rbfdd object on the stratified_test_data
                                        [raw_output, ground_truth] = \
                                            rbfdd.test(concatenated_test_data, concatenated_test_label, m, sd, W)
                                        # save the outputs of the train and test
                                        step_result.append([raw_output, ground_truth, m, sd, W, trained_y])
                                    # Increment the round counter
                                    round_counter = round_counter + 1
                                    # Before changing the hyper-parameters, store step_result, into the final_result dictionary
                                    final_result[str(current_hyper_parameters)] = step_result

        # Now we have all we need inside result. We now find the best-hyperparameters by finding the one combination
        # that has the highest average macro-fmeasure
            elif algorithm == "OCSVM":
                # define a dictionary for saving everything
                final_result = dict()
                # in the following section we will go through different combinations(i.e., n) of hyper-parameters
                # in order to know where we are in the middle of the execution
                round_counter = 1
                total_number_rounds = len(Nu) * len(Gamma)
                for nu in Nu:
                    for gamma in Gamma:
                        # this list wholse the information regarding each sampling round for a particular hyper-parameter combination
                        step_result = []
                        current_hyper_parameters = [nu, gamma]
                        for sampleing_times in range(SAMPLING_TIMES):
                            print('Dataset being sampled: ' + dataset + ' Scenario: ' + str(scenario))
                            print("Current Algorithm: " + algorithm)
                            print('Current Hyper-Parameters: (Nu, Gamma): (%.5f, %.5f)' % (nu, gamma))
                            print('Current Round: %d ' % round_counter + 'Out of: %d' % total_number_rounds + ' Rounds')
                            print('Sample Number: %d ' % (
                            sampleing_times + 1) + 'Out of: %d' % SAMPLING_TIMES + ' Sampling Rounds')
                            print("\n")
                            # create an RBFDD object right here
                            ocsvm = OCSVM(dataset, nu, gamma, normal, anomalous)
                            # bootstrap sample from the train_data
                            boot_strap_size = int(normal_data.shape[0] * 0.80)
                            # Generate a uniform random sample from np.arange(boot_strap_size) of size boot_strap_size:
                            # generate the index of the sampled values NO REPLACEMENT
                            boot_strap_train_index = np.random.choice(normal_data.shape[0], boot_strap_size, replace=False)
                            # extract the associated boot_strap_train data for the RBFDD training (No need for labels)
                            boot_strap_train_data = normal_data[boot_strap_train_index]
                            # Call the train function of the rbfdd using the boot_strap_train_data
                            # Get the learned parameters
                            # [clf, average_train_dist, std_train_dist, train_dist] = ocsvm.train(boot_strap_train_data)
                            [clf, train_dist] = ocsvm.train(boot_strap_train_data)
                            # Find the remaining indices in normal_data and add them to the anomalous data
                            # test_data_normal_portion = []
                            # test_label_normal_portion = []
                            # for index in range(normal_data.shape[0]):
                            #     if index not in boot_strap_train_index:
                            #         test_data_normal_portion.append(normal_data[index])
                            #         test_label_normal_portion.append(normal_data_label[index])
                            # test_data_normal_portion = np.array(test_data_normal_portion)
                            # test_label_normal_portion = np.array(test_label_normal_portion)
                            test_data_normal_portion = np.delete(normal_data, boot_strap_train_index, axis=0)
                            test_label_normal_portion = np.delete(normal_data_label, boot_strap_train_index, axis=0)
                            # Now concatenate these newly extracted normal samples to the existing
                            # anomalous data to create the the final test set for the prediction
                            concatenated_test_data = np.concatenate((test_data_normal_portion, anomalous_data), axis=0)
                            concatenated_test_label = np.concatenate((test_label_normal_portion, anomalous_data_label),
                                                                     axis=0)
                            # Before passing the test data, let's shuffle it to avoid ANY STREAMIMG assumption
                            concatenated_test_data, concatenated_test_label = shuffle(concatenated_test_data,
                                                                                      concatenated_test_label,
                                                                                      random_state=0)
                            # test the rbfdd object on the stratified_test_data
                            [raw_output, ground_truth, prediction] = ocsvm.test(concatenated_test_data, concatenated_test_label,clf)
                            # save the outputs of the train and test
                            step_result.append([raw_output, ground_truth, prediction, clf, train_dist])
                        # Increment the round counter
                        round_counter = round_counter + 1
                        # Before changing the hyper-parameters, store step_result, into the final_result dictionary
                        final_result[str(current_hyper_parameters)] = step_result
            elif algorithm == "AEN":
                ENCODING_DIM = H
                ENCODING_ACTIVATION =['relu']# ['sigmoid', 'relu']
                FINAL_LAYER_ACTIVATION = ['linear']#, 'sigmoid']
                ERROR_FUNCTION = ['MSE']#, 'binary_crossentropy']
                # define a dictionary for saving everything
                final_result = dict()
                # in the following section we will go through different combinations(i.e., n) of hyper-parameters
                # in order to know where we are in the middle of the execution
                round_counter = 1
                total_number_rounds = len(ENCODING_DIM)*len(MAX_EPOCH)*len(ENCODING_ACTIVATION)*len(FINAL_LAYER_ACTIVATION)*len(ERROR_FUNCTION)*len(BP_ETA)
                for encoding_dim in ENCODING_DIM:
                    for max_epoch in MAX_EPOCH:
                        for bp_eta in BP_ETA:
                            for encoding_activation in ENCODING_ACTIVATION:
                                for final_layer_activation in FINAL_LAYER_ACTIVATION:
                                    for error_function in ERROR_FUNCTION:
                                        current_hyper_parameters = [encoding_dim, max_epoch, bp_eta, encoding_activation, final_layer_activation, error_function]
                                        # this list whose  information regarding each sampling round for a particular hyper-parameter combination
                                        step_result = []
                                        for sampleing_times in range(SAMPLING_TIMES):
                                            print('Dataset being sampled: ' + dataset + ' Scenario: ' + str(scenario))
                                            print("Current Algorithm: " + algorithm)
                                            print('Current Hyper-Parameters: (encoding_dim, eta, max_epoch, encoding_activation, final_layer_activation, error function): ''(%d, %.2f, %d, %s, %s, %s)' % (encoding_dim, bp_eta, max_epoch, encoding_activation, final_layer_activation, error_function))
                                            print('Current Round: %d ' % round_counter + 'Out of: %d' % total_number_rounds + ' Rounds')
                                            print('Sample Number: %d ' % (sampleing_times + 1) + 'Out of: %d' % SAMPLING_TIMES + ' Sampling Rounds')
                                            print("\n")
                                            # create an RBFDD object right here
                                            aen = AEN(dataset, mini_batch, input_dim, bp_eta, encoding_dim, encoding_activation,final_layer_activation,max_epoch, normal, anomalous, error_function)
                                            # bootstrap sample from the train_data
                                            boot_strap_size = int(normal_data.shape[0] * 0.80)
                                            # Generate a uniform random sample from np.arange(boot_strap_size) of size boot_strap_size:
                                            # generate the index of the sampled values NO REPLACEMENT
                                            boot_strap_train_index = np.random.choice(normal_data.shape[0], boot_strap_size,
                                                                                      replace=False)
                                            # extract the associated boot_strap_train data for the RBFDD training (No need for labels)
                                            boot_strap_train_data = normal_data[boot_strap_train_index]
                                            # Call the train function of the rbfdd using the boot_strap_train_data
                                            # Get the learned parameters
                                            # [clf, encoder, decoder, average_E, sd_E, trained_Normal_E] = aen.train(boot_strap_train_data)
                                            [encoder, decoder, train_original, train_reconstructed, trained_Normal_E] = aen.train(boot_strap_train_data)
                                            # Find the remaining indices in normal_data and add them to the anomalous data
                                            # test_data_normal_portion = []
                                            # test_label_normal_portion = []
                                            # for index in range(normal_data.shape[0]):
                                            #     if index not in boot_strap_train_index:
                                            #         test_data_normal_portion.append(normal_data[index])
                                            #         test_label_normal_portion.append(normal_data_label[index])
                                            # test_data_normal_portion = np.array(test_data_normal_portion)
                                            # test_label_normal_portion = np.array(test_label_normal_portion)
                                            test_data_normal_portion = np.delete(normal_data, boot_strap_train_index,
                                                                                 axis=0)
                                            test_label_normal_portion = np.delete(normal_data_label,
                                                                                  boot_strap_train_index, axis=0)
                                            # Now concatenate these newly extracted normal samples to the existing
                                            # anomalous data to create the the final test set for the prediction
                                            concatenated_test_data = np.concatenate((test_data_normal_portion, anomalous_data),
                                                                                    axis=0)
                                            concatenated_test_label = np.concatenate(
                                                (test_label_normal_portion, anomalous_data_label), axis=0)
                                            # Before passing the test data, let's shuffle it to avoid ANY STREAMIMG assumption
                                            concatenated_test_data, concatenated_test_label = shuffle(concatenated_test_data,
                                                                                                      concatenated_test_label,
                                                                                                      random_state=0)
                                            # test the rbfdd object on the stratified_test_data
                                            [raw_error, ground_truth] = aen.test(concatenated_test_data,
                                                                                                   concatenated_test_label,
                                                                                                   encoder,
                                                                                                   decoder)
                                            # save the outputs of the train and test
                                            step_result.append([raw_error, ground_truth, train_original, train_reconstructed, trained_Normal_E])
                                            # Increment the round counter
                                        round_counter = round_counter + 1
                                        # Before changing the hyper-parameters, store step_result, into the final_result dictionary
                                        final_result[str(current_hyper_parameters)] = step_result
            elif algorithm == "GMM":
                # define a dictionary for saving everything
                final_result = dict()
                # in the following section we will go through different combinations(i.e., n) of hyper-parameters
                # in order to know where we are in the middle of the execution
                round_counter = 1
                total_number_rounds = len(H)
                for h in H:
                    current_hyper_parameters = [h]
                    # this list wholse the information regarding each sampling round for a particular hyper-parameter combination
                    step_result = []
                    for sampleing_times in range(SAMPLING_TIMES):
                        print("Current Algorithm: " + algorithm)
                        print('Dataset being sampled: ' + dataset + ' Scenario: ' + str(scenario))
                        print(
                            'Current Round: %d ' % round_counter + 'Out of: %d' % total_number_rounds + ' Rounds')
                        print(
                            'Current Hyper-Parameters: (H): (%d)' % h)
                        print('Sample Number: %d ' % (
                            sampleing_times + 1) + 'Out of: %d' % SAMPLING_TIMES + ' Sampling Rounds')
                        print("\n")
                        # create an RBFDD object right here
                        gmm = GMM(dataset, h, normal, anomalous)
                        # bootstrap sample from the train_data
                        boot_strap_size = int(normal_data.shape[0] * 0.80)
                        # Generate a uniform random sample from np.arange(boot_strap_size) of size boot_strap_size:
                        # generate the index of the sampled values NO REPLACEMENT
                        boot_strap_train_index = np.random.choice(normal_data.shape[0], boot_strap_size, replace=False)
                        # extract the associated boot_strap_train data for the RBFDD training (No need for labels)
                        boot_strap_train_data = normal_data[boot_strap_train_index]
                        # Call the train function of the rbfdd using the boot_strap_train_data
                        # Get the learned parameters
                        [Model, trained_prob] = gmm.train(boot_strap_train_data)
                        # Find the remaining indices in normal_data and add them to the anomalous data
                        # test_data_normal_portion = []
                        # test_label_normal_portion = []
                        # for index in range(normal_data.shape[0]):
                        #     if index not in boot_strap_train_index:
                        #         test_data_normal_portion.append(normal_data[index])
                        #         test_label_normal_portion.append(normal_data_label[index])
                        # test_data_normal_portion = np.array(test_data_normal_portion)
                        # test_label_normal_portion = np.array(test_label_normal_portion)
                        test_data_normal_portion = np.delete(normal_data, boot_strap_train_index, axis=0)
                        test_label_normal_portion = np.delete(normal_data_label, boot_strap_train_index, axis=0)
                        # Now concatenate these newly extracted normal samples to the existing
                        # anomalous data to create the the final test set for the prediction
                        concatenated_test_data = np.concatenate(
                            (test_data_normal_portion, anomalous_data), axis=0)
                        concatenated_test_label = np.concatenate(
                            (test_label_normal_portion, anomalous_data_label), axis=0)
                        # Before passing the test data, let's shuffle it to avoid ANY STREAMIMG assumption
                        concatenated_test_data, concatenated_test_label = shuffle(
                            concatenated_test_data, concatenated_test_label, random_state=0)
                        # test the rbfdd object on the stratified_test_data

                        [raw_output, ground_truth, prediction] = \
                            gmm.test(concatenated_test_data,
                                       concatenated_test_label, Model)
                        # save the outputs of the train and test
                        step_result.append(
                            [raw_output, ground_truth, prediction, Model, trained_prob])
                    # Increment the round counter
                    round_counter = round_counter + 1
                    # Before changing the hyper-parameters, store step_result, into the final_result dictionary
                    final_result[str(current_hyper_parameters)] = step_result
            elif algorithm == "iForest":
                # define a dictionary for saving everything
                final_result = dict()
                # in the following section we will go through different combinations(i.e., n) of hyper-parameters
                # in order to know where we are in the middle of the execution
                round_counter = 1
                total_number_rounds = len(N_ESTIMATORS)
                for n_estimators in N_ESTIMATORS:
                    current_hyper_parameters = [n_estimators]
                    # this list wholse the information regarding each sampling round for a particular hyper-parameter combination
                    step_result = []
                    for sampleing_times in range(SAMPLING_TIMES):
                        print("Current Algorithm: " + algorithm)
                        print('Dataset being sampled: ' + dataset + ' Scenario: ' + str(scenario))
                        print(
                            'Current Round: %d ' % round_counter + 'Out of: %d' % total_number_rounds + ' Rounds')
                        print(
                            'Current Hyper-Parameters: (# Base Estimators): (%d)' %
                            (n_estimators))
                        print('Sample Number: %d ' % (
                            sampleing_times + 1) + 'Out of: %d' % SAMPLING_TIMES + ' Sampling Rounds')
                        print("\n")
                        # create an RBFDD object right here
                        iforest = iForest(dataset, n_estimators, normal, anomalous)
                        # bootstrap sample from the train_data
                        boot_strap_size = int(normal_data.shape[0] * 0.80)
                        # Generate a uniform random sample from np.arange(boot_strap_size) of size boot_strap_size:
                        # generate the index of the sampled values NO REPLACEMENT
                        boot_strap_train_index = np.random.choice(normal_data.shape[0], boot_strap_size, replace=False)
                        # extract the associated boot_strap_train data for the RBFDD training (No need for labels)
                        boot_strap_train_data = normal_data[boot_strap_train_index]
                        # Call the train function of the rbfdd using the boot_strap_train_data
                        # Get the learned parameters
                        [Model, train_scores] = iforest.train(boot_strap_train_data)
                        # Find the remaining indices in normal_data and add them to the anomalous data
                        # test_data_normal_portion = []
                        # test_label_normal_portion = []
                        # for index in range(normal_data.shape[0]):
                        #     if index not in boot_strap_train_index:
                        #         test_data_normal_portion.append(normal_data[index])
                        #         test_label_normal_portion.append(normal_data_label[index])
                        # test_data_normal_portion = np.array(test_data_normal_portion)
                        # test_label_normal_portion = np.array(test_label_normal_portion)
                        test_data_normal_portion = np.delete(normal_data, boot_strap_train_index, axis=0)
                        test_label_normal_portion = np.delete(normal_data_label, boot_strap_train_index, axis=0)
                        # Now concatenate these newly extracted normal samples to the existing
                        # anomalous data to create the the final test set for the prediction
                        concatenated_test_data = np.concatenate((test_data_normal_portion, anomalous_data), axis=0)
                        concatenated_test_label = np.concatenate((test_label_normal_portion, anomalous_data_label),
                                                                 axis=0)
                        # Before passing the test data, let's shuffle it to avoid ANY STREAMIMG assumption
                        concatenated_test_data, concatenated_test_label = shuffle(concatenated_test_data,
                                                                                  concatenated_test_label,
                                                                                  random_state=0)
                        # test the rbfdd object on the stratified_test_data
                        [raw_output, ground_truth, prediction] = iforest.test\
                            (concatenated_test_data, concatenated_test_label, Model)
                        # save the outputs of the train and test
                        step_result.append([raw_output, ground_truth, prediction, Model, train_scores])
                    # Increment the round counter
                    round_counter = round_counter + 1
                    # Before changing the hyper-parameters, store step_result, into the final_result dictionary
                    final_result[str(current_hyper_parameters)] = step_result

            # Saving the objects before changing either the scenario or the dataset:
            if algorithm == 'EBFDD' or algorithm == 'RBFDD' or algorithm == 'AEN':
                name = str(algorithm)+'-'+str(dataset)+'-'+'(N, A)'+str(scenario)+str(H)+str(MAX_EPOCH)
            else:
                name = str(algorithm)+"-"+str(dataset)+'-'+'(N, A)'+str(scenario)
            with open(name+'.pkl', 'wb') as f:
                pickle.dump(final_result, f)
print("===============================================================")
print("===============================================================")
print("===============================================================")
print("All Data has been SUCCESSFULLY Saved :-)")
print("===============================================================")
print("===============================================================")
print("===============================================================")
