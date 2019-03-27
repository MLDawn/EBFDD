import numpy as np
import pandas as pd

def prepare_Wisconsin(normal, anomalous):
    dataset = pd.read_csv(

        'D:\PhD\AD-Benchmark Datasets\Binary Classification\Breast-cancer-wisconsin.data')
    X = np.array(dataset[dataset.columns[1:10]])
    Y = np.array(dataset[dataset.columns[10]])
    missing_counter = 0
    missing_index = []
    for i in range(X.shape[0]):
        if '?' not in X[i]:
            X[i] = X[i].astype(np.float)
        else:
            missing_counter = missing_counter + 1
            missing_index.append(i)

    # Get rid of the missing rows from both X and Y
    temp = []
    temp_index = []
    for i in range(X.shape[0]):
        if i not in missing_index:
            temp.append(X[i])
            temp_index.append(Y[i])
    X = np.array(temp)
    Y = np.array(temp_index)

    maximum = np.max(X, axis=0)
    minimum = np.min(X, axis=0)
    denum = maximum - minimum
    num = np.subtract(X, minimum)
    X = np.divide(num, denum)

    # Count the umber of rows with '?' missing values
    # (2 for benign, 4 for malignant)
    # Now let's put them in the desired shape
    normal_data = []
    normal_data_label = []
    anomalous_data = []
    anomalous_data_label = []
    for i in range(X.shape[0]):
        if Y[i] in normal:
            normal_data.append(X[i])
            normal_data_label.append(Y[i])
        elif Y[i] in anomalous:
            anomalous_data.append(X[i])
            anomalous_data_label.append(Y[i])

    # Before mixing up adding anything normal to the All_testing_images, and removing them from the All_training_images
    # We need to do the Normalization
    normal_data = np.array(normal_data, dtype=np.float64)
    normal_data_label = np.array(normal_data_label,dtype=np.float64)
    anomalous_data = np.array(anomalous_data,dtype=np.float64)
    anomalous_data_label = np.array(anomalous_data_label,dtype=np.float64)

    return normal_data, normal_data_label, anomalous_data, anomalous_data_label