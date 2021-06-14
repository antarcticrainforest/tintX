
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .helpers import get_grids, get_times
from .tracks import Cell_tracks

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

    def __init__(self,
                 input_files,
                 var_name,*,
                 lon_name='lon',
                 lat_name='lat',
                 client=None,
                 start=None,
                 end=None,
                 **kwargs):
        """
        Create an RunDirecotry object from a given input directory.

        ::

            run = RunDirectory('/work/mh0066/precip-project/3-hourly/CMORPH')

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
        client: dask.distributed cleint, optional (default: None)
            Configuration that is used the create a dask client which recieves
            tasks for multiproccessing. By default (None) a local client will
            be started.
        """

        for key, value in dict(coords="minimal",
                       data_vars="minimal",
                       compat='override',
                       combine='by_coords',
                       parallel=True).items():
            kwargs.setdefault(key, value)

        _dset = xr.open_mfdataset(input_files, **kwargs)
        self.start = self.conv_datetime([start or _dset.isel(time=0).time.values[0]])
        self.end = self.conv_datetime([end or _dset.isel(time=-1).time.values[0]])
        self.data = _dset.sel(time=slice(self.start, self.end))
        self.time = self.conv_datetime(self.data.time).to_pydatetime()
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.var_name = var_name
        self.lons = self.data[lon_name].values
        self.lats = self.data[lat_name].values
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

        self._get_tracks(self.grids, centre)
    
    @property
    def grids(self):
        """Create dictionary holding longitude and latitude information."""
        yield from get_grids(self.data, (0, self.data.time.shape[0]-1),
                             self.lons,
                             self.lats,
                             varname=self.var_name,
                             times=self.time)
