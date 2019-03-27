import numpy as np
import pandas as pd

def prepare_Ionosphere(normal, anomalous):
    dataset = pd.read_csv('D:\PhD\Benchmark Datasets and Papers\Adapting Radial Basis Function Neural Networks for One-Class Classification\Ionosphere\ionosphere.data')
    X = np.array(dataset[dataset.columns[1:34]])
    Y = np.array(dataset[dataset.columns[34]])

    # Now let's put them in the desired shape
    normal_data = []
    normal_data_label = []
    anomalous_data = []
    anomalous_data_label = []
    # Labels are 'g' for good, and 'b' for bad (replace with integers for future simplification)
    for i in range(X.shape[0]):
        if Y[i] in normal:
            normal_data.append(X[i])
            normal_data_label.append(Y[i])
        elif Y[i] in anomalous:
            anomalous_data.append(X[i])
            anomalous_data_label.append(Y[i])

    # Before mixing up adding anything normal to the All_testing_images, and removing them from the All_training_images
    # We need to do the Normalization
    normal_data = np.array(normal_data)
    normal_data_label = np.array(normal_data_label)
    anomalous_data = np.array(anomalous_data)
    anomalous_data_label = np.array(anomalous_data_label)
    return normal_data, normal_data_label, anomalous_data, anomalous_data_label
