# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:38:03 2018

@author: mbergemann@unimelb.edu.au
"""


import matplotlib
from tint import Cell_tracks, animate
import os, pandas as pd
from itertools import groupby
import numpy as np
from netCDF4 import Dataset as nc, num2date, date2num
from datetime import datetime, timedelta
from tint.helpers import get_times, get_grids
from tint.visualization import embed_mp4_as_gif, plot_traj
dataF = os.path.join(os.path.abspath('.'),'data','CPOL_radar.nc') #NetCDF data file
trackdir = os.path.join(os.path.abspath('.'),'tracks') #Output directory
overwrite = True #Overwirte existing old files
first = '2006-11-16 03:00' #Start-date
last = '2006-11-17 15:00' #End-date

f = nc('/home/unimelb.edu.au/mbergemann/Data/Darwin/netcdf/Cmorph_1998-2010.nc')
lats = f.variables['lat'][:]
lons = f.variables['lon'][:]
slices = get_times(f.variables['time'], first, last, None)
x = lons[int(len(lons)/2)]
y = lats[int(len(lats)/2)]
s = slices[0]
gr = (i for i in get_grids(f, s, lons, lats, varname='precip'))
anim = (i for i in get_grids(f, s, lons, lats, varname='precip'))
start = num2date(f.variables['time'][s[0]],
                 f.variables['time'].units)
end = num2date(f.variables['time'][s[-1]],
               f.variables['time'].units)
suffix = '%s-%s'%(start.strftime('%Y_%m_%d_%H'), end.strftime('%Y_%m_%d_%H'))
tracks_obj = Cell_tracks()
tracks_obj.params['MIN_SIZE'] = 4
tracks_obj.params['FIELD_THRESH'] = 1
tracks_obj.params['ISO_THRESH'] = 2
tracks_obj.params['ISO_SMOOTH'] = 2
tracks_obj.params['SEARCH_MARGIN'] = 750
tracks_obj.params['FLOW_MARGIN'] = 1550
tracks_obj.params['MAX_DISPARITY'] = 999
tracks_obj.params['MAX_FLOW_MAG ']= 50
tracks_obj.params['MAX_SHIFT_DISP'] = 15
tracks_obj.params['GS_ALT'] = 1500
track_file = os.path.join(trackdir,'cpol_tracks_%s.pkl'%suffix)
ncells = tracks_obj.get_tracks(gr, (x,y))
animate(tracks_obj, anim, os.path.join(trackdir,'ani', 'cmporph_tracks_%s.mp4'%suffix),
        overwrite=overwrite, dt=9.5, tracers=True, basemap_res='f')
f.close()
embed_mp4_as_gif(os.path.join(trackdir,'ani', 'cmporph_tracks_%s.mp4'%suffix))
ax = plot_traj(tracks_obj.tracks, lons, lats, basemap_res='f', label=True, mintrace=2)