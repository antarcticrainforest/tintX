
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .helpers import get_grids, get_times
from .tracks import Cell_tracks
from .visualization import animate, plot_traj, embed_mp4_as_gif

class RunDirectory(Cell_tracks):

    @staticmethod
    def conv_datetime(dates):
        """
        Convert datetime objects in icon format to python datetime objects.

        ::

            time = conf_datetime([20011201.5])

        Parameters
        ----------

        icon_dates: collection
            Collection of date dests

            Returns
            -------

                dates:  pd.DatetimeIndex
        """
        try:
            dates = dates.values
        except AttributeError:
            pass

        try:
            dates = dates[:]
        except TypeError:
            dates = np.array([dates])

        def _convert(in_date):
            frac_day, date = np.modf(in_date)
            frac_day *= 60**2 * 24
            date = str(int(date))
            date_str = datetime.datetime.strptime(date, '%Y%m%d')
            td = datetime.timedelta(seconds=int(frac_day.round(0)))
            return date_str + td

        conv = np.vectorize(_convert)
        try:
            out = conv(dates)
        except TypeError:
            out = dates
        if len(out) == 1:
            return pd.DatetimeIndex(out)[0]
        return pd.DatetimeIndex(out)


    def __enter__(self):
        """
        Create enter method.

        The enter method just returns the object it self. It is used
        to work along the with __exit__ method that closes a distributed
        worker.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the distributed client befor exiting."""
        self.close_client()

    @classmethod
    def open_dataset(cls, input_files, var_name,*,
                     lon_name='lon', lat_name='lat',
                     start=None, end=None, client=None, **kwargs):
        """
        Create an RunDirecotry object from a given input files.

        ::

            run = RunDirectory.open_dataset('/precip-project/3-hourly/CMORPH/*.nc')

        The RunDirectory object gathers all nesseccary information on the
        data that is stored in the run directory. Once loaded the most
        important meta data will be stored in the run directory for faster
        access the second time.

        Parameters
        ----------
        inp_files: str, list
            Input filenames or directory that is opened.
        var_name: str
            Name of the variable that is tracked
        start: str, pandas.Timestamp, datetime.datetime (default: None)
            first time step that is considered, if none given the first
            of the dataset is considered.
        end: str, pandas.Timestamp, datetime.datetime (default: None)
            last time step that is considered, if none given the last
            of the dataset is considered.
        lon_name: str (default: lon)
            The name of the longitude vector/array
        lat_name: str (default: lat)
            The name of the latitude vector/array
        client: dask.distributed cleint, optional (default: None)
            Configuration that is used the create a dask client which recieves
            tasks for multiproccessing. By default (None) a local client will
            be started.
        kwargs:
            Additional keyword arguments that are passed to open the dataset
            with xarray
        """
        for key, value in dict(coords="minimal",
                       data_vars="minimal",
                       compat='override',
                       combine='by_coords',
                       parallel=True).items():
            kwargs.setdefault(key, value)

        _dset = xr.open_mfdataset(input_files, **kwargs)
        start = cls.conv_datetime([start or _dset.isel(time=0).time.values[0]])
        end = cls.conv_datetime([end or _dset.isel(time=-1).time.values[0]])
        data = _dset.sel(time=slice(start, end))
        return cls(var_name, _dset.sel(time=slice(start, end)),
                   _dset[lon_name], _dset[lat_name])

    def __init__(self,
                 var_name,
                 dataset,
                 lons,
                 lats,*,
                 client=None):

        """
        Create an RunDirecotry object from a given xarray/netcdf4 dataset.

        ::
            with netCDF4.Dataset('/precip-project/3-hourly/CMORPH.nc') as f:
                lons = f['lon'][0,:]
                lats = f['lat'][:,0]
                run = RunDirectory('precip', f, lons, lats)

        ::
            dset = xarray.open_mfdataset('/precip-project/3-hourly/CMORPH.nc')
            lons = f['lon'][0,:].values
            lats = f['lats']:,0].values
            run = RunDirectory('precip', dset.isel(time=slice(100, 150), lons, lats)

        The RunDirectory object gathers all nesseccary information on the
        data that is stored in the run directory. Once loaded the most
        important meta data will be stored in the run directory for faster
        access the second time.

        Parameters
        ----------
        inp_files: str, list
            Input filenames or directory that is opened.
        var_name: str
            Name of the variable that is tracked
        dataset: xarray.Dataset, netCDF4.Dataset,
            Dataset that is applied to the tracking algorithm
        lons: numpy.ndarray, xarray.DataArray
            The longitude coordinate array/vector
        lats: numpy.ndarray, xarray.DataArray
            The latitude coordinate array/vector
        client: dask.distributed cleint, optional (default: None)
            Configuration that is used the create a dask client which recieves
            tasks for multiproccessing. By default (None) a local client will
            be started.
        """
        try:
            self.lons = lons.values
        except AttributeError:
            self.lons = lons
        try:
            self.lats = lats.values
        except AttributeError:
            self.lats = lats
        self.data = dataset
        self.var_name = var_name
        self.time = self.conv_datetime(dataset.variables['time']).to_pydatetime()
        self.start = self.time[0]
        self.end = self.time[-1]
        super().__init__(var_name)

    def get_tracks(self, centre=None):
        """ Obtains tracks given a list of data arrays. This is the
            primary method of the tracks class. This method makes use of all of the
            functions and helper classes defined above.

        Parameters
        ----------
            centre: tuple, default: None
                The centre of the radar station
        """

        return self._get_tracks(self.grids, centre)

    @property
    def grids(self):
        """Create dictionary holding longitude and latitude information."""
        yield from get_grids(self.data, (0, self.data.time.shape[0]-1),
                             self.lons,
                             self.lats,
                             varname=self.var_name,
                             times=self.time)

    def animate(self, *args, **kwargs):
        """
        Create a gif animation of tracked cells.

        Parameters
        ----------
        outfile_name : str
            The name of the output file to be produced.
        alt : float
            The altitude to be plotted in meters.
        vmin, vmax : float
            Limit values for the colormap.
        arrows : bool
            If True, draws arrow showing corrected shift for each object. Only used
            in 'full' style.
        isolation : bool
            If True, only annotates uids for isolated objects. Only used in 'full'
            style.
        uid : str
            The uid of the object to be viewed from a lagrangian persepective. Only
            used when style is 'lagrangian'.
        fps : int
            Frames per second for output gif.
        
        kwargs:
            Additional keyword arguments passed to the basemap plotting routine.
        """
        return animate(self, self.grids, *args, **kwargs)

    def plot_traj(self, *args, **kwargs):
        """
        This code is a fork of plot_traj method in the plot module from the
        trackpy project see http://soft-matter.github.io/trackpy fro more details

        Plot traces of trajectories for each particle.
        Optionally superimpose it on a frame from the video.
        Parameters
        ----------
        colorby : {'particle', 'frame'}, optional
        mpp : float, optional
            Microns per pixel. If omitted, the labels will have units of pixels.
        label : boolean, optional
            Set to True to write particle ID numbers next to trajectories.
        basemap_res: str
            Set the resolution of the basemap
        superimpose : ndarray, optional
            Background image, default None
        cmap : colormap, optional
            This is only used in colorby='frame' mode. Default = mpl.cm.winter
        ax : matplotlib axes object, optional
            Defaults to current axes
        t_column : string, optional
            DataFrame column name for time coordinate. Default is 'frame'.
        particles : a preset of stroms (particles) to be drawn, instead of all
            (default)
        size : size of the sctter indicating start and end of the storm
        color : A pre-defined color, if None (default) each track will be assigned
            a different color
        thresh : tuple for thresholds to be applied to the plotted objects. first
            entry of the tuple is the variable (default 'mean') second one the
            the minimum value (default -1)
        create_map: boolean, reate a map object, this can be useful for loops where
            a basemap object has already been created
        pos_columns : list of strings, optional
            Dataframe column names for spatial coordinates. Default is ['x', 'y'].
        plot_style : dictionary
            Keyword arguments passed through to the `Axes.plot(...)` command
        mintrace : int
            Minimum length of a trace to be plotted
        Returns
        -------
        Axes object
        """
        return plot_traj(self.tracks, self.lons, self.lats, *args, **kwargs)
