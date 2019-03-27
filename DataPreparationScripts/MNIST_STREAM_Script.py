# This script takes the desired train_size, and test_size and ultimately creates the training, and testing sets and returns them
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
from matplotlib import pyplot
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def TRAIN_SIZE(num):

    x_train = mnist.train.images[:num, :]
    y_train = mnist.train.labels[:num,:]

    return x_train, y_train
def TEST_SIZE(num):

    x_test = mnist.test.images[:num, :]
    y_test = mnist.test.labels[:num,:]

    return x_test, y_test
def prepare_Streamn_MNIST(train_size, test_size, normal_digits, anomalous_digits, num_normal_patterns, num_anomalous_patterns):
    # {label:[actual digit]}
    separated_normal_digits = {}
    separated_anomalous_digits = {}

    All_training_digits, All_training_digits_labels = TRAIN_SIZE(train_size)
    All_testing_digits, All_testing_digits_labels = TEST_SIZE(test_size)
    # prepare them for the algorithms to work with, based on the normal/anomalous digits
    # Now let's put them in the desired shape
    normal_stream = []
    normal_stream_label = []
    anomalous_stream = []
    anomalous_stream_label = []
    # We would like all that is normal, inside All_training_digits to be extracted for train_data no label needed(RBFDD: Unsupervised)
    # We would like all that is anomalous, inside All_training_digits to be extracted for test_data and test_label
    for i in range(All_training_digits.shape[0]):
        label = np.argmax(All_training_digits_labels[i])
        if label in normal_digits:
            if label not in separated_normal_digits:
                separated_normal_digits[label] = [All_training_digits[i]]
            else:
                # two_d = All_training_digits[i].reshape(28, 28)
                # pyplot.imshow(two_d, cmap='gray')
                # pyplot.show()
                separated_normal_digits[label].append(All_training_digits[i])

    for i in range(All_testing_digits.shape[0]):
        label = np.argmax(All_testing_digits_labels[i])
        if label in anomalous_digits:
            if label not in separated_anomalous_digits:
                separated_anomalous_digits[label] = [All_testing_digits[i]]
            else:
                # two_d = All_testing_digits[i].reshape(28, 28)
                # pyplot.imshow(two_d, cmap='gray')
                # pyplot.show()
                separated_anomalous_digits[label].append(All_testing_digits[i])
    # Now normal and anomalous digits are nicely separated into 2 dictionaries {label:[image]}
    # Now depending on the order of normal _digits and anomalous_digits, we can construct STREAMS
    # Of Normal and Anomalous patterns
    for counter in range(num_normal_patterns):
        for n in normal_digits:
            for label in separated_normal_digits:
                if label == n:
                    List = separated_normal_digits[label]
                    Rand_Image = List[random.randrange(len(List))]
                    # two_d = Rand_Image.reshape(28, 28)
                    # pyplot.imshow(two_d, cmap='gray')
                    # pyplot.show()
                    normal_stream.append(Rand_Image)
        # add label 1 for the normal stream
        normal_stream_label.append(1)
    # normal_stream(num_normal_patterns, stream_length, dimensionality of every data at each time stamp)
    normal_stream = np.array(normal_stream)
    normal_stream_label = np.array(normal_stream_label)

    for counter in range(num_anomalous_patterns):
        for a in anomalous_digits:
            for label in separated_anomalous_digits:
                if label == a:
                    List = separated_anomalous_digits[label]
                    Rand_Image = List[random.randrange(len(List))]
                    # two_d = Rand_Image.reshape(28, 28)
                    # pyplot.imshow(two_d, cmap='gray')
                    # pyplot.show()
                    anomalous_stream.append(Rand_Image)
        # add label 0 for the normal stream
        anomalous_stream_label.append(0)
    anomalous_stream = np.array(anomalous_stream)
    anomalous_stream_label = np.array(anomalous_stream_label)


    return normal_stream, normal_stream_label, anomalous_stream, anomalous_stream_label


