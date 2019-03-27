import numpy as np
import pandas as pd

def prepare_Fashion_MNIST(normal, anomalous):
    dataset = pd.read_csv('D:\PhD\Benchmark Datasets and Papers\Fashionmnist\Fashion-mnist_train.csv')
    X = dataset[dataset.columns[1:]]
    Y = np.array(dataset["label"])
    # Now let's put them in the desired shape
    normal_data = []
    normal_data_label = []
    anomalous_data = []
    anomalous_data_label = []
    for i in range(len(X)):
        if Y[i] in normal:
            normal_data.append(X.iloc[i].values)
            normal_data_label.append(Y[i])
        elif Y[i] in anomalous:
            anomalous_data.append(X.iloc[i].values)
            anomalous_data_label.append(Y[i])
    # Now about the test data
    dataset = pd.read_csv('D:\PhD\Benchmark Datasets and Papers\Fashionmnist\Fashion-mnist_test.csv')
    X = dataset[dataset.columns[1:]]
    Y = np.array(dataset["label"])

    for i in range(len(X)):
        if Y[i] in normal:
            normal_data.append(X.iloc[i].values)
            normal_data_label.append(Y[i])
        elif Y[i] in anomalous:
            anomalous_data.append(X.iloc[i].values)
            anomalous_data_label.append(Y[i])

    normal_data = np.array(normal_data, dtype=np.float64)
    normal_data_label = np.array(normal_data_label, dtype=np.float64)
    anomalous_data = np.array(anomalous_data, dtype=np.float64)
    anomalous_data_label = np.array(anomalous_data_label, dtype=np.float64)

    # Normalize
    normal_data = normal_data/255
    anomalous_data = anomalous_data/255

    return normal_data, normal_data_label, anomalous_data, anomalous_data_label