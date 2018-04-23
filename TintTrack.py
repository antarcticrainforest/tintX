import Cell_tracks, animate

import os, pandas as pd
from itertools import groupby
import numpy as np
from netCDF4 import Dataset as nc, num2date, date2num
from datetime import datetime, timedelta
from helper import get_times, get_grids
if __name__ == '__main__':
    dataF = os.path.join(os.getenv("HOME"),'Data','Extremes','CPOL','CPOL_1998-2017.nc')
    trackdir = os.path.join(os.path.dirname(dataF),'Tracking')
    overwrite = True
    start = '2006-11-10 00:00'
    end = '2006-11-18 18:00'
    with nc(dataF) as ncf:
        slices = get_times(ncf['10min'].variables['time'], start, end,
                           ncf['10min'].variables['isfile'][:])
        lats = ncf.variables['lat'][:]
        lons = ncf.variables['lon'][:]
        x = lons[int(117/2)]
        y = lats[int(117/2)]
        grids = []
        for s in slices:
            ani = False
            gr = (i for i in get_grids(ncf['10min'], s, lons, lats))
            anim = (i for i in get_grids(ncf['10min'], s, lons, lats))
            start = num2date(ncf['10min'].variables['time'][s[0]],
                             ncf['10min'].variables['time'].units)

            end = num2date(ncf['10min'].variables['time'][s[-1]],
                           ncf['10min'].variables['time'].units)
            suffix = '%s-%s'%(start.strftime('%Y_%m_%d_%H'), end.strftime('%Y_%m_%d_%H'))
            tracks_obj = Cell_tracks()
            tracks_obj.params['MIN_SIZE'] = 4
            tracks_obj.params['FIELD_THRESH'] = 1
            track_file = os.path.join(trackdir,'tint_tracks_%s.pkl'%suffix)
            if not os.path.isfile(track_file) or overwrite:
                ncells = tracks_obj.get_tracks(gr, (x,y))
                if ncells > 2 :
                    tracks_obj.tracks.to_pickle(track_file)
                    ani = True
                else:
                    ani = False
            '''
            else:
                try:
                    tracks_obj.tracks = pd.read_pickle(track_file)
                    tracks_obj.radar_info = {'radar_lat':y, 'radar_lon':x}
                    ani = True
                except FileNotFoundError:
                    ani = False
            '''
            if ani:
                animate(tracks_obj, anim,
                        os.path.join(trackdir,'video', 'tint_tracks_%s.mp4'%suffix),
                        overwrite=overwrite, dt = 9.5)
            #break

