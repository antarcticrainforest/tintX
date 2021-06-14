#!/usr/bin/env python
# coding: utf-8

# ### This Python notebook should sreve as an example of how to use tint
# First import all modules that are needed

from matplotlib import pyplot as plt
from tint import RunDirectory, animate
from tint.visualization import embed_mp4_as_gif, plot_traj
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")


trackdir = Path(__file__).parent / 'tracks' #Output directory
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

RD = RunDirectory('data/*.nc', 'radar_estimated_rain_rate',
                  start=first, end=last, lon_name='longitude',
                  lat_name='latitude')
suffix = '%s-%s'%(RD.start.strftime('%Y_%m_%d_%H'),
                  RD.end.strftime('%Y_%m_%d_%H'))
RD.params['MIN_SIZE'] = 4
RD.params['FIELD_THRESH'] = 1
track_file = trackdir / f'tint_tracks_{suffix}.h5'
ncells = RD.get_tracks()
RD.tracks.to_hdf(track_file, 'radar_tracks')
animate(RD, RD.grids, trackdir / 'ani' / f'tint_tracks_{suffix}.mp4',
        overwrite=True, dt=9.5, tracers=True, basemap_res='f')

#  the tracks are saved in a dataframe and can be accessed by the ```.tracks``` instance:

embed_mp4_as_gif(trackdir / 'ani' / f'tint_tracks_{suffix}.mp4')
fig = plt.figure()
ax = fig.add_subplot(111)
ax = plot_traj(RD.tracks, RD.lons, RD.lats, basemap_res='f', label=True, size=20, ax=ax)
fig.savefig(Path('tracks') / f'tint_tracks_{suffix}.png', bbox_inches='tight', dpi=300)
