import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.mlab as mlab
import pandas as pd
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import KMeans
from sklearn import decomposition
from scipy.stats import friedmanchisquare
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
LABEL = ['RBFDD', 'OCSVM', 'EBFDD','AEN','GMM','iForest']
objects = ('EBFDD', 'RBFDD','AEN','GMM','iForest','OCSVM')
y_pos = np.arange(len(objects))


dataset = pd.read_csv('D:\PhD\Benchmark Experiments\\AllScoresCSV.csv')
RBFDD = np.array(dataset["Rank-RBFDD"])
EBFDD = np.array(dataset["Rank-EBFDD"])
OCSVM = np.array(dataset["Rank-OCSVM"])
GMM = np.array(dataset["Rank-GMM"])
AEN = np.array(dataset["Rank-AEN"])
iForest = np.array(dataset["Rank-iForest"])
ClassLabel = np.array(dataset["DATASET"])
ranks_all = [EBFDD, RBFDD, AEN, GMM, iForest, OCSVM]
# dataset_markers = {'GAMMA':'.','SPAM':',','SKIN':'o','FAULT':'v','IMAGE':'<','LAND':'>','PAGE':'X', 'WAVE':'s'}
map = {0:'RBFDD', 1: 'OCSVM', 2:'EBFDD', 3:'AEN', 4:'GMM', 5:'iForest'}
winner_marker = {'EBFDD':'*','RBFDD':'d','AEN':'o','GMM':'v','OCSVM':'<','iForest':'>'}
fig, axes = plt.subplots(1,3)
# axes[0].set_title('All Scenarios')

axes[0].boxplot(ranks_all, showmeans=True)
# vol_1 = axes[0].violinplot(ranks_all)
# for pc in vol_1['bodies']:
#     pc.set_facecolor('#ffdd75')
#     pc.set_edgecolor('black')
#     pc.set_alpha(1)

axes[0].set_xticklabels(objects, fontsize=12)
axes[0].set_ylim(0,7)

# figure()
# sns.swarmplot(data=ranks_all)
# plt.show()
# ONLY 1vs ALL case

dataset = pd.read_csv('D:\PhD\Benchmark Experiments\\1vsall.csv')
RBFDD = np.array(dataset["Rank-RBFDD"])
EBFDD = np.array(dataset["Rank-EBFDD"])
OCSVM = np.array(dataset["Rank-OCSVM"])
GMM = np.array(dataset["Rank-GMM"])
AEN = np.array(dataset["Rank-AEN"])
iForest = np.array(dataset["Rank-iForest"])

ranks_1vsall = [EBFDD, RBFDD, AEN, GMM, iForest, OCSVM]
# axes[1].set_title('1 vs. All')
axes[1].boxplot(ranks_1vsall, showmeans=True)

# vol_2=axes[1].violinplot(ranks_1vsall)
# for pc in vol_2['bodies']:
#     pc.set_facecolor('#fbede3')
#     pc.set_edgecolor('black')
#     pc.set_alpha(1)

axes[1].set_xticklabels(objects, fontsize=12)
axes[1].set_ylim(0,7)


# ONLY All vs 1 case

dataset = pd.read_csv('D:\PhD\Benchmark Experiments\\allvs1.csv')
RBFDD = np.array(dataset["Rank-RBFDD"])
EBFDD = np.array(dataset["Rank-EBFDD"])
OCSVM = np.array(dataset["Rank-OCSVM"])
GMM = np.array(dataset["Rank-GMM"])
AEN = np.array(dataset["Rank-AEN"])
iForest = np.array(dataset["Rank-iForest"])

ranks_allvs1 = [EBFDD, RBFDD, AEN, GMM, iForest, OCSVM]
# axes[2].set_title('All vs. 1')
axes[2].boxplot(ranks_allvs1, showmeans=True)
# vol_3 = axes[2].violinplot(ranks_allvs1)
# for pc in vol_3['bodies']:
#     pc.set_facecolor('#c6e2ff')
#     pc.set_edgecolor('black')
#     pc.set_alpha(1)

axes[2].set_xticklabels(objects, fontsize=12)
axes[2].set_ylim(0,7)

# Generate some numeric statistics as well
print("(EBFDD, RBFDD, AEN, GMM, iForest, OCSVM)")
print("=========================================ALL EXPERIMENTS======================================")
print('Mean: '+ str(np.mean(ranks_all, axis=1))+'\n'+ 'Variance: '+ str(np.var(ranks_all, axis=1)))
print("=========================================ONE vs ALL EXPERIMENTS======================================")
print('Mean: '+ str(np.mean(ranks_1vsall, axis=1))+'\n'+ 'Variance: '+ str(np.var(ranks_1vsall, axis=1)))
print("=========================================ALL vs ONE EXPERIMENTS======================================")
print('Mean: '+ str(np.mean(ranks_allvs1, axis=1))+'\n'+ 'Variance: '+ str(np.var(ranks_allvs1, axis=1)))

#EBFDD, RBFDD, AEN, GMM, iForest, OCSVM
#
#
# fig = plt.figure()
# # Apply K-means on the ranks
# ranks_allvs1 = np.array(ranks_allvs1).T
# ranks_1vsall = np.array(ranks_1vsall).T
# ranks_all = np.array(ranks_all).T
# RANKS = [ranks_all, ranks_1vsall, ranks_allvs1]
# n_components = 3
# n_clusters = [8, 5, 5]
# counter = 1
# color = ['r','b','y','g','c','k','m','#FF7F50','#BDB76B','#808000','#800000','#708090']
# titles = ['All', 'One vs All', 'All vs One']
# for D in RANKS:
#     ax = fig.add_subplot(1, 3, counter, projection='3d')
#     kmeans = KMeans(n_clusters=n_clusters[counter - 1], random_state=0).fit(np.array(D))
#     ID = kmeans.labels_
#     #Extract the winner's name from D before reducing its dimensionality
#     winners = []
#     for d in D:
#         index = np.where(d == 1)[0][0]
#         winners.append(map[index])
#     # Reduce dimensionality
#     pca_train = decomposition.PCA(n_components=n_components, whiten=True)
#     pca_train.fit(D)
#     R = pca_train.transform(D)
#     for i in range(R.shape[0]):
#         ax.scatter(R[i, 0], R[i, 1], R[i, 2], c=color[ID[i]], marker=winner_marker[winners[i]], alpha=0.5, s=500)
#     ax.set_title(titles[counter - 1]+"\n"+("%d Clusters") % n_clusters[counter - 1])
#     ax.grid()
#     ax.set_xlabel('First PC')
#     ax.set_ylabel('Second PC')
#     ax.set_zlabel('Third PC')
#     counter += 1
#

plt.show()


# print("-----------------------------------------------Friedman Test--------------------------------------------------")
# #D[:, 0], D[:, 1], D[:, 2],D[:, 3], D[:, 4], D[:, 5] where D = (RBFDD, OCSVM, EBFDD,AEN, GMM, iForest)
# stat, p = friedmanchisquare(D[:, 3], D[:, 4],D[:, 0])
# print('Statistics=%f, p=%f' % (stat, p))
# # interpret
# alpha = 0.05
# if p > alpha:
# 	print('Same distributions (fail to reject H0)')
# else:
# 	print('Different distributions (reject H0)')


