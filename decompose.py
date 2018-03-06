

def plot_cluster_components(df, decomposition='tsne', lle_method='standard', plot='2D', n_clusters=3, n_components=3, titlex='XXX', fname =None, azim=90,elev=90):
    
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

    colors = ['g','r','b', 'k']
    
    if plot=='2D':
        fig = plt.figure(figsize=(10,10))
        for i in range(n_components):
            ds = data_projected[np.where(clusterx.labels_==i)]
            # select only data observations with cluster label == i
            plt.xlabel('PC1',fontsize=20, weight='bold',labelpad=15)
            plt.ylabel('PC2',fontsize=20, weight='bold',labelpad=15)
            plt.plot(ds[:,0],ds[:,1],'o', c=colors[i])
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
        ax.set_xlabel('PC1',fontsize=20, weight='bold',labelpad=15)
        ax.set_ylabel('PC2',fontsize=20, weight='bold',labelpad=15)
        ax.set_zlabel('PC3',fontsize=20, weight='bold',labelpad=15)
        ax.scatter  (ds1[:,0],ds1[:,1],ds1[:,2], c=colors[0],  s = 200)
        ax.scatter(ds2[:,0],ds2[:,1],ds2[:,2], c=colors[1],  s = 200)
        ax.scatter(ds3[:,0],ds3[:,1],ds3[:,2], c=colors[2],  s = 200)
        ax.view_init(elev=elev, azim=azim)   
        plt.title(titlex)
    
    plt.xticks(fontsize=15,weight='bold')
    plt.yticks(fontsize=15,weight='bold')
    
     
    
    if fname:
        plt.savefig(fname,dpi=300, bbox_inches='tight', transprent=True)

