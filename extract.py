import os
import numpy as np
import pandas as pd
import zipfile, json, tcxparser
import json
from urllib2 import urlopen

#IO
datadir   = '/Users/kanaan/Desktop/Run'
nike      = os.path.join(datadir, 'Nike/export')
apple     = os.path.join(datadir, 'apple_health_export_28_02_2018/export.xml')


def getplace(lat, lon):
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, lon)
    v = urlopen(url).read()
    j = json.loads(v)
    components = j['results'][0]['address_components']
    for c in components:
        if "country" in c['types']:
            country = c['long_name']
        if 'political' in c['types'] and 'locality' in c['types']:
            city = c['short_name']
    return city, country


def preproc_nike():

    # Grab all zipfiles
    nike_runs = sorted([os.path.join(nike, i) for i in os.listdir(nike) if i.endswith('zip')])

    # Create Pandas Dataframe
    df_nike = pd.DataFrame(index=range(len(nike_runs)))

    # Loop through zip files, read json data and populate df
    for idx, nike_run in enumerate(nike_runs):

        # load zipfile
        nike_zip = zipfile.ZipFile(nike_run, 'r')

        # load json only
        nike_json = json.load([nike_zip.open(i) for i in nike_zip.namelist() if i.endswith('json')][0])
        nike_tcx = tcxparser.TCXParser([nike_zip.open(i) for i in nike_zip.namelist() if i.endswith('tcx')][0])

        try:
           if nike_tcx.latitude:
               lat = nike_tcx.latitude
               lon = nike_tcx.longitude
               city, country = getplace(lat, lon)
               location = '%s, %s' %(city,country)
               print idx, lat, lon, city, country
           else:
               location = 'indoor'
        except:
            location = 'indoor'

        df_nike.loc[idx, 'Source'] = '%s %s ' % (nike_json['source'], nike_json['appversion'][:-5])
        df_nike.loc[idx,'Location']       = location# nike_json['title']
        df_nike.loc[idx,'Latitude']       = lat
        df_nike.loc[idx,'Longitude']      = lon
        df_nike.loc[idx, 'Source'] = '%s %s ' % (nike_json['source'], nike_json['appversion'][:-5])
        df_nike.loc[idx, 'Speed_Max'] = nike_json['maxSpeed']
        df_nike.loc[idx, 'Date'] = nike_json['startTime']['time'][0:10]
        df_nike.loc[idx, 'Description'] = nike_json['description']
        df_nike.loc[idx, 'Speed'] = nike_json['avgSpeed']
        df_nike.loc[idx, 'Speed_Max'] = nike_json['maxSpeed']
        try:
            if nike_tcx.pace:
                df_nike.loc[idx, 'Tempo'] = nike_tcx.pace
            else:
                df_nike.loc[idx, 'Tempo'] = 'xxxxxxxxxxxx'
        except:
            pass
        df_nike.loc[idx, 'Distance'] = nike_tcx.distance / 1000
        df_nike.loc[idx, 'Duration'] = nike_tcx.duration / 60
        df_nike.loc[idx, 'Start'] = nike_json['startTime']['time'][11:-1]

        df_nike.loc[idx, 'Ascent'] = nike_tcx.ascent
        df_nike.loc[idx, 'Descent'] = nike_tcx.descent
        df_nike.loc[idx, 'Elevation_Gain'] = nike_json['elevationGain']
        df_nike.loc[idx, 'Elevation_Max'] = nike_json['maxElevation']

        df_nike.loc[idx, 'Calories'] = nike_json['calories']

        for lap_idx, lap in enumerate(nike_json['laps']):
            df_nike.loc[idx, 'Km_%s' % (lap_idx + 1)] = 'Duration=%s, Speed=%s' % (np.round(lap['duration'] / 60., 3),
                                                                                   np.round(lap['avgSpeed'], 2))


        df_nike.to_csv('./Data/NikeRunClub.csv', encoding='utf-8')


#extract nike data
preproc_nike()

#extract apple data
os.system('python applehealth.py %s'%apple)
os.system('cp %s/apple_health_export_28_02_2018/*.csv Data'%datadir)