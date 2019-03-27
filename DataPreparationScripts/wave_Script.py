import numpy as np
import pandas as pd

def prepare_wave(normal, anomalous):
    # Depending on what normal is, build the anomalous label's list
    dataset = pd.read_csv('D:\PhD\AD-Benchmark Datasets\Multiclass Classification\\wave.original.csv')
    X = dataset[dataset.columns[1:]]
    X = X.values
    Y = np.array(dataset["Y"])
    # Normalize the data before segmentation
    maximum = np.max(X, axis=0)
    minimum = np.min(X, axis=0)
    denum = maximum - minimum
    num = np.subtract(X, minimum)
    X = np.divide(num, denum)

    # Now let's put them in the desired shape
    normal_data = []
    normal_data_label = []
    anomalous_data = []
    anomalous_data_label = []
    for i in range(X.shape[0]):
        # All vs 1
        if len(normal) == 1 and normal[0] == -1:
            if Y[i] in anomalous:
                anomalous_data.append(X[i])
                anomalous_data_label.append(Y[i])
            else:
                normal_data.append(X[i])
                normal_data_label.append(Y[i])
        # 1 vs All
        elif len(anomalous) == 1 and anomalous[0] == -1:
            if Y[i] in normal:
                normal_data.append(X[i])
                normal_data_label.append(Y[i])
            else:
                anomalous_data.append(X[i])
                anomalous_data_label.append(Y[i])
        # some vs some
        else:
            if Y[i] in normal:
                normal_data.append(X[i])
                normal_data_label.append(Y[i])
            elif Y[i] in anomalous:
                anomalous_data.append(X[i])
                anomalous_data_label.append(Y[i])
    # Now get rid of the transferred units, in the original training data structures
    normal_data = np.array(normal_data, dtype=np.float64)
    normal_data_label = np.array(normal_data_label)
    anomalous_data = np.array(anomalous_data, dtype=np.float64)
    anomalous_data_label = np.array(anomalous_data_label)
    return normal_data, normal_data_label, anomalous_data, anomalous_data_label