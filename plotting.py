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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, MDS, SpectralEmbedding
import scipy.stats as ss
import sklearn.metrics as sm

from scipy.cluster.hierarchy import inconsistent, linkage, dendrogram

sys.path.append('./')
from utils import *
from preprocess import *
from plotting import *

# set some viz options
pd.options.display.max_columns=100
sns.set_style('whitegrid')

import vincent
vincent.core.initialize_notebook()
vincent.initialize_notebook()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Plotting Functions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def plot_bar(df, feature, palette, title, xlabel, ylabel, labelsize=20, rot=70):
    fig, ax = plt.subplots(figsize=(20,12))
    g = sns.barplot(df.index, df[feature],palette= palette)
    sns.despine(top=True,right=True)
    plt.xticks(rotation=rot, fontsize=labelsize)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=35, weight='bold')
    plt.xlabel(xlabel, fontsize=25, labelpad=15, weight='bold')
    plt.ylabel(ylabel, fontsize=25, weight='bold')
    change_width(ax, 0.98)
    return fig, ax, g


def plot_reg(x, y, title, xlabel,ylabel, color, size = 7):
    sns.set_style('white')
    matplotlib.rc("legend", fontsize=20)
    
    f = plt.figure(figsize =(30,10))
    g = sns.jointplot(x, y, kind='reg', size=size, color=color,
                      marginal_kws=dict(bins=30, rug=True, hist=False, kde=True, kde_kws={'shade':1}))
    g.fig.suptitle(title,  fontsize=20, weight='bold')
    plt.xlabel(xlabel, fontsize=20, labelpad=15, weight='bold')
    plt.ylabel(ylabel, fontsize=20, weight='bold')
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    g.fig.subplots_adjust(top=.9)
    
    
#########################
def plt_world_map(df, title = None, fname=None):
    from mpl_toolkits.basemap import Basemap
    sns.set_style('white')
    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'c' means use crude resolution coastlines
    # you can also have 'l' for low, then 'h' for high. Unless coastlines are
    # really important to you, or lakes, you should just use c for crude.

    lat = df.Latitude
    lon = df.Longitude

    fig = plt.figure(figsize=(20, 10), edgecolor='w')
    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=90,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.fillcontinents(color='#FFFFFF',lake_color='#000099', alpha=1) # 66CC66 #0066CC
    m.drawcountries()
    m.drawmapboundary(fill_color='#FFFFFF')

    for lat, lon in zip(df.Latitude, df.Longitude):
        x,y = m(lon,lat)
        m.scatter(lon, lat, marker = 'o', color='r',s=200, zorder=10, latlon=True)

    #plt.annotate('Ouahu, Hawaii', xy=(10,10),xytext =(10,10), fon)
    sns.despine(left=True, bottom=True)
    if title:
        plt.title(title, fontsize=25, weight='bold')
    if fname:
        plt.savefig(fname,bbox_inches='tight',dpi=300, transparent=True)
        
    

########################################################################################################################
##### Colormaps

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import matplotlib.colors as colors
import numpy as np

first = int((128*2)-np.round(255*(1.-0.50)))
second = (256-first)
cmap_gradient = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                  np.vstack((plt.cm.YlOrRd(np.linspace(0.98, 0.25, second)),
                                                            plt.cm.viridis(np.linspace(.98, 0.0, first)))))
cmap_drysdale = colors.ListedColormap(['#00ffff', '#00afff','#0000ff', '#260000', '#530000','#fe0000', '#ff6a00',
                                           '#ffff00'])

cmap_ted = colors.ListedColormap(['#00ffff', '#00afff','#0000ff', '#260000', '#530000','#fe0000', '#ff6a00',
                                      '#ffff00', '#ffffff'])

    
    
    
    