from pylab import plot, show, xlim, figure, ylim, legend, boxplot, setp, axes
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sklearn.metrics import roc_curve, auc
import math

###################################################### IMPORTANT #################################################################
# The put the outputs of the EBFDD code in a directory all together. Insert the path to that directory in the variable path below
# For example, if you had 3 algorithms, for a given dataset, and scenario, you will have 3 separate files. Each one starts with the
# Name of one of the algorithms.
# This code will go over all 3 files, and gives you a visual representation of the best set of hyper-parameters, per algorithm, per dataset
path = ""

def setBoxColors(bp):
    '''
    This function is responsible for the appearance of the boxplots
    '''
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['medians'][0], color='blue')


def find_winner_row(final_result, algorithm):
    maximum_auc = 0
    chosen_hyper_parameter = []
    for key in final_result.keys():
        # Grab current row
        row = final_result[key]
        # go through the row, and gather all the macro's
        aucs = []
        FP = []
        for item in row:
            ground_truth = item[1]
            output = item[0]
            # In case we have NAN values for the AUC make sure to SKIP this whole ROW as the hyperparameters
            # seem to have been inappropriate
            if np.isnan(output).any() == True:
                aucs.append(0)
                continue
            if algorithm == 'AEN':
                # As the logic of the AEN is different from the others, we need to multiply the outputs by -1
                # This way the ground truth and the output are monotonically correlated (Necessary for computing AUC scores)
                fpr, tpr, thresholds = roc_curve(ground_truth, -output)
            else:
                fpr, tpr, thresholds = roc_curve(ground_truth, output)
            aucs.append(auc(fpr, tpr))
            FP.append(np.mean(fpr))
        aucs_avg = np.mean(aucs)
        if aucs_avg >= maximum_auc:
            error = np.mean(FP)
            maximum_auc = aucs_avg
            chosen_hyper_parameter = key
    combination = chosen_hyper_parameter
    winner_combination = final_result[chosen_hyper_parameter]
    return combination, winner_combination, maximum_auc, error

algorithms = ['EBFDD','RBFDD', 'OCSVM', 'AEN', 'GMM', 'iForest']

all_files = dict()
for root, dirs, files, in os.walk(path):
    for file in files:
        if file.endswith(".pkl"):
            # load the pickle file
            read_file = open(path + '\\' + file, 'rb')
            object_file = pickle.load(read_file)
            for alg in algorithms:
                if file.startswith(alg):
                    all_files[alg] = object_file

winners = dict()
winner_combination = dict() # This will hold the name of the algorithm and the highest Average AUC score it has had through the experiments
for alg in all_files.keys():
    [winner_combination[alg], winners[alg], max_auc,Error] = find_winner_row(all_files[alg], alg)
    print(str(alg)+': ' + str(max_auc)+' FP rate: '+str(Error))


outputs = dict()
auc_scores = dict()

for alg in winners.keys():
    temp_auc_scores = []
    row = winners[alg]
    for i in range(len(row)):
        if alg == 'AEN':
            fpr, tpr, thresholds = roc_curve(row[i][1], -row[i][0])
        else:
            fpr, tpr, thresholds = roc_curve(row[i][1], row[i][0])

        temp_auc_scores.append(auc(fpr, tpr))
    auc_scores[alg] = temp_auc_scores

AUC = []
AUC_means = []
AUC_labels = []
AUC_std = []
# Get the mean and standard deviation for the AUC scores of each algorithm (Only for the winner hyper-parameters)
#--> Used to give us information regarding the stability of the algorithms
for i in auc_scores.keys():
    AUC_labels.append(i)
    AUC.append(auc_scores[i])
    AUC_means.append(np.mean(auc_scores[i]))
    AUC_std.append(np.std(auc_scores[i]))

fig = plt.figure(figsize=(16, 9))
ylim(0,2)
ax2 = axes()
ax2.set_ylabel('AUC Score Variance', fontsize=18)

p = 5
start_position = []
for pos in range(len(AUC)):
    start_position.append(p)
    p = p + 5

plt.boxplot(AUC, positions=start_position)


lab = []
for l in AUC_labels:
    if l == 'RBFDD':
        lab.append('RBFDD'+'\n'+"(H, K-epoch, BP-epoch, Keta, BPeta, beta, theta)"+'\n'+winner_combination['RBFDD'])
    elif l == 'AEN':
        lab.append('AEN'+'\n'+"(H, Epoch, H-Activation, Final-Activation)"+'\n'+winner_combination['AEN'])
    elif l == 'OCSVM':
        lab.append('OCSVM'+'\n'+"(Nu, Gamma)"+'\n'+winner_combination['OCSVM'])
    elif l == 'EBFDD':
        lab.append('EBFDD' + '\n' + "(H, Epoch, Eta, Beta(Variances), Theta (for weights))" + '\n' + winner_combination['EBFDD'])
    elif l == 'GMM':
        lab.append('GMM' + '\n' + "H" + '\n' + winner_combination[
            'GMM'])
    elif l == 'iForest':
        lab.append('iForest' + '\n' + "# Estimators" + '\n' + winner_combination[
            'iForest'])
print('\n')
print("The mean-AUC scores computed are:", [auc_scores.keys(), AUC_means])
print("Standard Deviation per Algorithm for AUCs:", AUC_std)

ax2.set_xticklabels(lab, fontsize=10)

plt.show()



