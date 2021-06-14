#!/usr/bin/env python
# coding: utf-8

# ### This Python notebook should sreve as an example of how to use tint
# First import all modules that are needed

import matplotlib
from tint import Cell_tracks, animate
import os, pandas as pd
from itertools import groupby
import numpy as np
from netCDF4 import Dataset as nc, num2date, date2num
from datetime import datetime, timedelta
from tint.helpers import get_times, get_grids
from tint.visualization import embed_mp4_as_gif, plot_traj
import warnings
warnings.filterwarnings("ignore")


dataF = os.path.join('data','CPOL_radar.nc') #NetCDF data file
trackdir = os.path.join('tracks') #Output directory
overwrite = True #Overwirte existing old files
first = '2006-11-16 03:00' #Start-date
last = '2006-11-16 11:00' #End-date


# The application of the tracking algorithm constists of the following steps:
#     1. read data and meta-data (from netCDF data)
#     2. apply the tracking and save the ouput to a pandas-dataframe
#     3. animate the tracking output

# ## The tuning parameters
# 
# The parameters play an important role when the tracking is applied to different datatypes (e.g satellite data).
# The algorithm offers the following tunable parameters:
# 
# 
# * FIELD_THRESH : The threshold used for object detection. Detected objects are connnected pixels above this threshold.
#     
# * ISO_THRESH : Used in isolated cell classification. Isolated cells must not be connected to any other cell by contiguous pixels above this threshold.
#     
# * ISO_SMOOTH : Gaussian smoothing parameter in peak detection preprocessing. See
#     single_max in tint.objects.
# * MIN_SIZE : The minimum size threshold in pixels for an object to be detected.
# * SEARCH_MARGIN : The radius of the search box around the predicted object center.
# * FLOW_MARGIN : The margin size around the object extent on which to perform phase correlation.
# * MAX_DISPARITY : Maximum allowable disparity value. Larger disparity values are sent to LARGE_NUM.
# * MAX_FLOW_MAG : Maximum allowable global shift magnitude.
# * MAX_SHIFT_DISP :Maximum magnitude of difference in meters per second for two shifts to be
#     considered in agreement.
# 

# ### Open the netCDF file and apply the tracking

with nc(str(dataF)) as ncf:
        slices = get_times(ncf.variables['time'], first, last) #Get a subset of the data
        lats = ncf.variables['latitude'][:,0] #The latitude vector - 1D
        lons = ncf.variables['longitude'][0,:] # Tthe longitude vector - 1D
        # Define the centre of the domain
        x = lons[lons.shape[0] // 2]
        y = lats[lats.shape[0] // 2]
        grids = []
        for s in slices:
            ani = False
            #Create an iterator for the data dictionary 
            gr = (i for i in get_grids(ncf, s, lons, lats, varname='radar_estimated_rain_rate'))
            anim = (i for i in get_grids(ncf, s, lons, lats, varname='radar_estimated_rain_rate'))
            #Construct start and end date of slcie
            start = num2date(ncf.variables['time'][s[0]],
                             ncf.variables['time'].units)
            end = num2date(ncf.variables['time'][s[-1]],
                           ncf.variables['time'].units)
            #Filename suffix
            suffix = '%s-%s'%(start.strftime('%Y_%m_%d_%H'), end.strftime('%Y_%m_%d_%H'))
            tracks_obj = Cell_tracks()
            tracks_obj.params['MIN_SIZE'] = 4
            tracks_obj.params['FIELD_THRESH'] = 1
            track_file = os.path.join(trackdir,'tint_tracks_%s.h5'%suffix)
            if not os.path.isfile(track_file) or overwrite:
                ncells = tracks_obj.get_tracks(gr, (x, y))
                if ncells > 2 :
                    # Save tracks in handy hdf5 format for later analysis
                    tracks_obj.tracks.to_hdf(track_file, 'radar_tracks')
                    ani = True
                else:
                    ani = False
            animate(tracks_obj, anim, os.path.join(os.path.abspath(trackdir),'ani', 'tint_tracks_%s.mp4'%suffix), 
                    overwrite=overwrite, dt=9.5, tracers=True, basemap_res='f')


# ### Accessing the tracking data
#  the tracks are saved in a dataframe and can be accessed by the ```.tracks``` instance:

embed_mp4_as_gif(os.path.join(trackdir, 'ani', 'tint_tracks_%s.mp4'%suffix))
ax = plot_traj(tracks_obj.tracks, lons, lats, basemap_res='f', label=True, size=20)

