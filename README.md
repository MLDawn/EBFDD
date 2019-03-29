# The Elliptical-Basis-Function-Data-Descriptor network: A One-Class Classification Approach to Anomaly Detector
This repository contains all the code for the EBFDD along its competitors. There are also the supplementary codes for reading the data from the datasets!
# Where are the datasets used in the code? Where can I find them?
The Fault Dataset: http://archive.ics.uci.edu/ml/datasets/steel+plates+faults
The Image Segmentation Dataset: http://archive.ics.uci.edu/ml/datasets/image+segmentation
The Page Dataset: https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification
The LandSat Dataset: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
The Wave Dataset: http://archive.ics.uci.edu/ml/datasets/waveform+database+generator+(version+1)
# What is the DataPreparationScripts folder all about?
It contains some scripts that are used by the EBFDD.py code, so it can normalize the datasets between 0 and 1 and partition them into normal and anomalous. They all return the normal data, anomalous data, normla data label and anomalous data label back to the EBFDD.py, so that EBFDD.py could start the training process for all the algorithms.
# What is the EBFDD.py script all about?
It contains several classes for different state of the algorithms (OCSVM, AEN, iForest, GMM, and RBFDD which is a special case of the EBFDD algorithm) that are used for the task of anomaly detection, including our method that is called the Elliptical Basis Function Data Descriptor (EBFDD) network.
Each class in instantiated by setting up its hyper-parameters, and then in has the train() and test() methods to allow it do apply training and do inference, respectively.
Given a dataset (a binary/multi-class classification dataset), we can consider the instances of ome of the classes as the normal data, and everything else as anomalous. All of these algorithms are trained only and only using the normal instances. During testing, some unseen normal data and anomalous data are shown to the algorithms, and then the raw output and the groundtruth values are computed and stored.
# What is the EBFDD AUC Visualiser.py script all about?
Let's say we used a dataset, and considered a particular scenario in that dataset where all the instances that belong to class 1 are normal and, every other instances are anomalous. Then we will pit all the algorithms against each other on this dataset. We have 6 algorithms so we will get 6 output files. We will put all of these in a separate directory, and set its path to the path variable in the EBFDD AUC Visualiser.py script. The output will be the a boxplot, where you will see 6 boxplots, one for each algorithm, corresponding to the BEST set of hyper-parameters that has resulted in the best average AUC score across the whole experiment. The boxplot, tells us about the variance of the AUC score of the best hyper-parameters per algorithm, which is good!
