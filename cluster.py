'__authour__ == Kanaan'

import os, sys
import time, datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, MDS, SpectralEmbedding
import scipy.stats as ss
import sklearn.metrics as sm

from scipy.cluster.hierarchy import inconsistent, linkage, dendrogram


def plot_dendogram(df, method='ward', metric='euclidean', fname=None):
    sns.set_style('white')
    Z = linkage(df, method=method, metric=metric, optimal_ordering=False)
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    ax = dendrogram(Z, show_contracted=True)  # , truncate_mode='lastp')
    plt.xlabel('Cluster')
    plt.ylabel('Distance')
    plt.axhline(4, linestyle='--', linewidth=3, color='r')
    plt.axhline(10, linestyle='--', linewidth=3, color='r')
    # plt.ylim((0,2500))
    if fname:
        plt.savefig(fname, dpi=300, transparent=True, bbox_inches='tight')


def plot_elbow(X, fname=None, title=None):
    distortions = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    xk = np.array(range(2, 20))
    fig = plt.figure(figsize=(12, 6))
    g = sns.barplot(xk, distortions, palette='autumn')
    # plt.bar(xk, distortions, color='autumn')
    plt.plot(xk - 2, distortions, marker="o", markerfacecolor="k", c='r')
    plt.scatter(xk - 2, distortions, s=100, facecolors='k', edgecolors='r')
    if title:
        plt.title('Optimal Number Of Clusters (Elbow Method)', fontsize=18, weight='bold')
    plt.xlabel('k', fontsize=20)
    plt.ylabel('Distortions', fontsize=20)
    sns.despine(left=False, bottom=False)
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight', transprent=True)


def return_cluster_dfs(df, df_z, n_clusters, features):
    # run h-clustering on z-scored dataframe
    clust = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    clust.fit(df_z)

    # add labels names to original dataframes
    df_L = df.copy(deep=True)
    df_L['label'] = clust.labels_
    df_z['label'] = clust.labels_

    # create cluster dataframe for feature averages/sum
    df_mu = pd.DataFrame(index=[i for i in range(n_clusters)], columns=features)

    for feature in features:
        if feature in ['Speed']:
            df_mu[feature] = df_L.groupby('label').mean()[feature]
        else:#
            df_mu[feature] = df_L.groupby('label').mean()[feature]

    df_mu.index = ['cluster_%s' % (i + 1) for i in range(n_clusters)]

    return df_mu, df_L, df_z

