'__authour__ == Kanaan'

import os,sys
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
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, MDS, SpectralEmbedding
import scipy.stats as ss
import sklearn.metrics as sm

from scipy.cluster.hierarchy import inconsistent, linkage, dendrogram


def plot_dendogram(df, method='ward', metric='euclidean'):
    sns.set_style('white')
    Z = linkage(df, method=method, metric=metric, optimal_ordering=False)
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    ax = dendrogram(Z, show_contracted=True)#, truncate_mode='lastp')
    plt.xlabel('Cluster')
    plt.ylabel('Distance')
    plt.axhline(4, linestyle='--', linewidth=3, color= 'r')
    plt.axhline(10, linestyle='--', linewidth=3, color= 'r')
    #plt.ylim((0,2500))
    
    
def plot_elbow(X):
    distortions = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 20), distortions, color='firebrick')
    plt.bar(range(2, 20), distortions, color='firebrick')
    plt.xticks(range(2, 20))
    plt.title('Elbow curve')
    
    
    
def plot_cluster_components(df, decomposition='tsne', lle_method='standard', plot='2D', n_clusters=3, n_components=3, titlex='XXX'):
    
    title = ''
    n_clusters = n_clusters
    clusterx = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    clusterx.fit(df)

    # decompose data and plot 2Pcs
    n_components = 3
    n_neighbors  = 10
    if decomposition=='isomap':
        data_projected = Isomap(n_components=n_components).fit_transform(df)
    elif decomposition=='tsne':
        data_projected = TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(df)
    elif decomposition == 'mds':
        data_projected = MDS(n_components, max_iter=100, n_init=1).fit_transform(df)
    elif decomposition == 'spectral':
        data_projected = SpectralEmbedding(n_components=n_components,  n_neighbors=n_neighbors).fit_transform(df)
    elif decomposition == 'lle':
        lle_methods  = ['standard', 'ltsa', 'hessian', 'modified']
        data_projected = LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
                                                method=lle_method).fit_transform(df)
    elif decomposition == 'kpca':
        kpca = KernelPCA(n_components=n_components, kernel="rbf", fit_inverse_transform=True, gamma=10,)
        data_projected = kpca.fit_transform(df)
    elif decomposition == 'pca':
        pca = PCA(n_components=n_components)
        data_projected = pca.fit_transform(df)

    colors = ['r','g','b', 'k']
    
    if plot=='2D':
        fig = plt.figure(figsize=(10,10))
        for i in range(n_components):
            ds = data_projected[np.where(clusterx.labels_==i)]
            # select only data observations with cluster label == i
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.plot(ds[:,0],ds[:,1],'o')
            plt.axis('tight')
            #if title:
            plt.title(titlex,fontsize = 20)
    
    
    elif plot=='3D':
        print data_projected.shape
        ds1 = data_projected[np.where(clusterx.labels_==0)]
        ds2 = data_projected[np.where(clusterx.labels_==1)]
        ds3 = data_projected[np.where(clusterx.labels_==2)]
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.scatter(ds1[:,0],ds1[:,1],ds1[:,2], c='r',  s = 200)
        ax.scatter(ds2[:,0],ds2[:,1],ds2[:,2], c='g',  s = 200)
        ax.scatter(ds3[:,0],ds3[:,1],ds3[:,2], c='b',  s = 200)
        plt.title(titlex)

def return_cluster_dfs(df, df_z, n_clusters, features):
    
    #run h-clustering on z-scored dataframe
    clust = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    clust.fit(df_z)

    # add labels names to original dataframes
    df_L = df.copy(deep=True)
    df_L['label'] = clust.labels_
    df_z['label'] = clust.labels_
    
    # create cluster dataframe for feature averages/sum
    df_mu = pd.DataFrame(index = [i for i in range(n_clusters)], columns = features)

    for feature in features:
        if feature in ['Speed']:
            df_mu[feature] =  df_L.groupby('label').mean()[feature]
        else:
            df_mu[feature] =  df_L.groupby('label').mean()[feature]
    
    df_mu.index= ['cluster_%s'%(i+1) for i in range(n_clusters)]    
    
    return  df_mu,df_L, df_z

