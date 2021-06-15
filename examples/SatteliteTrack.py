# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:38:03 2018

@author: mbergemann@unimelb.edu.au
"""

from pathlib import Path
from matplotlib import pyplot as plt
import xarray as xr

from tint import RunDirectory

# In this example we are going to process satellite based observation.
# Sometimes data needs to be processed first. For example if derived
# variables like (density potential temperature) are applied to the tracking,
# or data needs to be remapped first. In this scenario you would create the
# netCDF dataset yourself, rather then letting the code load the data, 
# and apply the tracking on the dataset. Below is an example:

# Read satellite based rainfall estimates and select a sub region.
trackdir = Path(__file__).parent / 'tracks' #Output directory
files = [str(f) for f in Path('data').rglob('CMORPH*.nc')]
# Select a box around the Maritime Continent
dset = xr.open_mfdataset(sorted(files), combine='by_coords').sel(lon=slice(100, 160), lat=slice(-13, 13))

RD = RunDirectory('cmorph', dset.isel(time=slice(0, 20)),
                  dset.lon,
                  dset.lat)
RD.params['MIN_SIZE'] = 8
RD.params['FIELD_THRESH'] = 3
RD.params['ISO_THRESH'] = 10
RD.params['ISO_SMOOTH'] = 10
RD.params['SEARCH_MARGIN'] = 8750
RD.params['FLOW_MARGIN'] = 1750
RD.params['MAX_DISPARITY'] = 999
RD.params['MAX_FLOW_MAG ']= 5000
RD.params['MAX_SHIFT_DISP'] = 1000
suffix = '%s-%s'%(RD.start.strftime('%Y_%m_%d_%H'),
                  RD.end.strftime('%Y_%m_%d_%H'))
track_file = trackdir / f'cmorph_tracks_{suffix}.h5'
ncells = RD.get_tracks()
track_file = trackdir / f'sat_tracks_{suffix}.h5'
RD.tracks.to_hdf(track_file, 'sat_tracks')
RD.animate(trackdir / 'ani' / f'sat_tracks_{suffix}.mp4', vmax=3,
           overwrite=True, dt=9.5, tracers=True, basemap_res='i')
fig = plt.figure()
ax = fig.add_subplot(111)
ax = RD.plot_traj(basemap_res='i', label=True, size=20, ax=ax)
fig.savefig(Path('tracks') / f'sat_tracks_{suffix}.png', bbox_inches='tight', dpi=300)
