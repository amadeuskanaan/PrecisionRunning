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

# set some viz options
pd.options.display.max_columns=100
sns.set_style('whitegrid')

import vincent
vincent.core.initialize_notebook()
vincent.initialize_notebook()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 iPhone7 Data pre-prpocessing  - ASK March-4-2018 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Define time feature dictionaries 
weekday = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 
           4:'Friday', 5:'Saturday', 6:'Sunday'}

tod = {'04':'04:00-06:00', '05':'04:00-06:00', '06':'06:00-08:00', '07':'06:00-08:00',
       '08':'08:00-10:00', '09':'08:00-10:00', '10':'10:00-12:00', '11':'10:00-12:00',
       '12':'12:00-14:00', '13':'12:00-14:00', '14':'14:00-16:00', '15':'14:00-16:00',
       '16':'16:00-18:00', '17':'16:00-18:00', '18':'18:00-20:00', '19':'18:00-20:00',
       '20':'20:00-22:00', '21':'20:00-22:00', '22':'22:00-04:00', '23':'22:00-04:00',
       '00':'22:00-04:00', '01':'22:00-04:00', '02':'22:00-04:00', '03':'22:00-04:00'}
   
    
def preproc_iphone_data(df, source):
    
    ##########################
    # define dataframe source  and get exact dates 
    
    df = df.copy(deep=True)
    
    if source is 'Nike':
        dates = [date for date in df['Date']]
    elif source is 'Apple':
        dates = [date[0:10] for date in df['startDate']]

    converted_dates = map(datetime.strptime, dates, len(dates)*['%Y-%m-%d'])
    
    ##########################
    # add columns for month and day 
    df['Month'] = [str(date.replace(day=1))[0:7] for date in converted_dates]
    df['Day']   = [str(date)[0:10] for date in converted_dates]
    
    ##########################
    # add columns for weekday
    converted_days =[i.weekday() for i in converted_dates]
    df['Weekday_value'] = [i for i in converted_days]
    for i in df.index:
        df.loc[i,'Weekday'] = weekday[df.loc[i]['Weekday_value']]
    
    ##########################
    # add column for time-of-day
    
    for i in df.index:
        if source is 'Nike':
            start_time = df.loc[i]['Start'][0:2]
        elif source is 'Apple':
            start_time = df.loc[i]['startDate'][11:13]
        
        df.loc[i,'Time_of_day_val'] = start_time
        df.loc[i,'Time_of_dayX'] = '%s:00-%s:00' %(df.loc[i]['Time_of_day_val'], int(df.loc[i]['Time_of_day_val'])+1)
        df.loc[i,'Time_of_day']  = tod[df.loc[i]['Time_of_day_val']]
    
    
    ##########################
    # add column for season 
    ##### Autumn Sep 23 to Dec-20 - Winter Dec-21 to Mar-20  
    ##### Spring Mar-21 to Jun-20 - Summer Jun-21 to Sep-22
    for i in df.index:
        season = get_season(datetime.strptime(df.loc[i]['Day'], '%Y-%m-%d'))
        df.loc[i,'Season']= season
    
    return df

def df_group_timefeature(df, time_feature, groupby, feature_name=None):
    
    # group by time feature
    dft  = df.copy(deep=True)
    dft = dft.groupby(time_feature)
    
    # groupby sum, mean or count
    if groupby is 'sum':
        dft = dft.sum()
    elif groupby is 'mean':
        dft = dft.mean()
    elif groupby is 'count':
        dft = dft.count()
    
    if time_feature is 'Month':
        for i in dft.index:
            dft.loc[i,'Season'] = get_season(datetime.strptime(i, '%Y-%m'))
    
    if feature_name:
        dft[feature_name] = dft.value
    
    return dft