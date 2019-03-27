import numpy as np
import pandas as pd

def prepare_KDD_CUP_1999(complete_dataset, normal, anomalous):
    if complete_dataset:
        dataset = pd.read_csv("D:\PhD\AD-Benchmark Datasets\Binary Classification\kddcup.data.corrected")
    else:
        dataset = pd.read_csv('D:\PhD\AD-Benchmark Datasets\Binary Classification\kddcup.data_10_percent.corrected')

    X = np.array(dataset[dataset.columns[0:41]])
    Y = np.array(dataset[dataset.columns[41]])


    # Convert categorical features
    # Categorical features indices in X: 1,2,3,6,11,20,21
    categorical_features = ['service', 'protocol_type', 'flag']
    service = ['icmp', 'tcp', 'udp']
    protocol_type = ['IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns',
                     'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i',
                     'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher',
                     'harvest', 'hostnames', 'http', 'http_2784', 'http_443',
                     'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell', 'ldap',
                     'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns',
                     'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other',
                     'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i',
                     'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc',
                     'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i',
                     'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']

    flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3',
            'SF', 'SH']

    dimensions = [len(service), len(protocol_type), len(flag)]

    # This will hold the one-hot encoded version of the categorical features
    hot_encode = dict()
    for i in range(len(categorical_features)):
        hot_encode[categorical_features[i]] = np.identity(dimensions[i])
    # Replace the categorical feature levels with their equivalent one-hot code representations
    dataset = []
    for i in range(X.shape[0]):
        temp = X[i]
        # Indices needed to be replaced: 1,2, 3
        s_index = service.index(temp[1])
        s_hot = hot_encode['service'][s_index]
        temp = np.concatenate((np.array([temp[0]]), s_hot, temp[2:]))

        p_index = protocol_type.index(temp[4])
        p_hot = hot_encode['protocol_type'][p_index]
        temp = np.concatenate((temp[0:4], p_hot, temp[5:]))

        f_index = flag.index(temp[74])
        f_hot = hot_encode['flag'][f_index]
        temp = np.concatenate((temp[0:74], f_hot, temp[75:]))

        # Final update
        dataset.append(temp)
    dataset = np.array(dataset)

    dataset = np.array(dataset, dtype=np.float64)
    # Normalize the dataset before segmentation
    maximum = np.max(dataset, axis=0)
    need_to_go = np.where(maximum == 0)[0]
    dataset = np.delete(dataset, need_to_go, axis=1)
    maximum = np.max(dataset, axis=0)
    # As the minimum is z across all features we comment this line out
    # num = np.subtract(dataset, minimum)
    dataset = dataset / maximum


    # Now let's put them in the desired shape
    normal_data = []
    normal_data_label = []
    anomalous_data = []
    anomalous_data_label = []

    for i in range(dataset.shape[0]):
        if Y[i] in normal:
            normal_data.append(dataset[i])
            normal_data_label.append(Y[i])
        elif Y[i] in anomalous:
            anomalous_data.append(dataset[i])
            anomalous_data_label.append(Y[i])

    # Now get rid of the transferred units, in the original training data structures
    normal_data = np.array(normal_data)
    normal_data_label = np.array(normal_data_label)
    anomalous_data = np.array(anomalous_data)
    anomalous_data_label = np.array(anomalous_data_label)

    return normal_data, normal_data_label, anomalous_data, anomalous_data_label