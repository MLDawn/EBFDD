# This script takes the desired train_size, and test_size and ultimately creates the training, and testing sets and returns them
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# def TRAIN_SIZE(num):
#     (x_train, y_train), (_, _) = mnist.load_data()
#     # x_train = mnist.train.images[:num, :]
#     # y_train = mnist.train.labels[:num,:]
#
#     return x_train, y_train
# def TEST_SIZE(num):
#
#     x_test = mnist.test.images[:num, :]
#     y_test = mnist.test.labels[:num,:]

    # return x_test, y_test
def prepare_MNIST(train_size, test_size, normal_digits, anomalous_digits):
    (All_training_digits, All_training_digits_labels), (All_testing_digits, All_testing_digits_labels) = mnist.load_data()
    All_training_digits = All_training_digits.astype('float32') / 255.
    All_testing_digits = All_testing_digits.astype('float32') / 255.
    All_training_digits = All_training_digits.reshape((len(All_training_digits), np.prod(All_training_digits.shape[1:])))
    All_testing_digits = All_testing_digits.reshape((len(All_testing_digits), np.prod(All_testing_digits.shape[1:])))

    # onehot_encoder = OneHotEncoder(sparse=False)
    # # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # All_training_digits_labels = onehot_encoder.fit_transform(All_training_digits_labels.reshape(All_training_digits_labels.shape[0], 1))
    # All_testing_digits_labels = onehot_encoder.fit_transform(All_testing_digits_labels.reshape(All_testing_digits_labels.shape[0], 1))


    # All_training_digits, All_training_digits_labels = TRAIN_SIZE(train_size)
    # All_testing_digits, All_testing_digits_labels = TEST_SIZE(test_size)
    # prepare them for the algorithms to work with, based on the normal/anomalous digits
    # Now let's put them in the desired shape
    normal_data = []
    normal_data_label = []
    anomalous_data = []
    anomalous_data_label = []
    # We would like all that is normal, inside All_training_digits to be extracted for train_data no label needed(RBFDD: Unsupervised)
    # We would like all that is anomalous, inside All_training_digits to be extracted for test_data and test_label
    for i in range(All_training_digits.shape[0]):
        # label = np.argmax(All_training_digits_labels[i])
        label = All_training_digits_labels[i]
        if label in normal_digits:
            normal_data.append(All_training_digits[i])
            normal_data_label.append(label)
        elif label in anomalous_digits:
            anomalous_data.append(All_training_digits[i])
            anomalous_data_label.append(label)
    for i in range(All_testing_digits.shape[0]):
        # label = np.argmax(All_testing_digits_labels[i])
        label = All_testing_digits_labels[i]
        if label in normal_digits:
            normal_data.append(All_testing_digits[i])
            normal_data_label.append(label)
        elif label in anomalous_digits:
            anomalous_data.append(All_testing_digits[i])
            anomalous_data_label.append(label)
    # take the whole All_testing_images as the test_data that is used for stradified sampling and prediction by RBFDD.test()
    normal_data = np.array(normal_data)
    normal_data_label = np.array(normal_data_label)
    anomalous_data = np.array(anomalous_data)
    anomalous_data_label = np.array(anomalous_data_label)
    return normal_data, normal_data_label, anomalous_data, anomalous_data_label


