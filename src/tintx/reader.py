from __future__ import annotations
from datetime import datetime
import os
from pathlib import Path
from typing import cast, Any, Optional, Union, Iterator

import pandas as pd
from tqdm.auto import tqdm
import xarray as xr

from .helpers import get_grids, GridType
from .tracks import Cell_tracks
from .visualization import full_domain, plot_traj
from .config import set as set_config


__all__ = ["RunDirectory"]


class RunDirectory(Cell_tracks):
    """
    Create an :class:`RunDirecotry` object from a given xarray dataset.

    The :class:`RunDirectory` object gathers all necessary information on the
    data that is stored in the run directory. Once loaded the most
    important meta data will be stored in the run directory for faster
    access the second time.

    Parameters
    ----------
    var_name: str
        Name of the variable that is tracked
    dataset: xarray.DataArray, xarray.Dataset
        Input dataset/data array
    x_coord: str, default: lon
        Name of the X coordinate array/vector
    y_coord: st, default: lat
        Name of the Y coordinate array/vector
    time_coord: str, default: time
        Name of the time variable.

    Example
    -------

    .. execute_code::

        import xarray
        from tintx import RunDirectory
        dset = xarray.open_mfdataset("source/_static/data/CMORPH*.nc", combine="by_coords")
        run = RunDirectory('precip', dset.isel(time=slice(10, 150)))

    """

    @classmethod
    def from_files(
        cls,
        input_files: Union[os.PathLike, list[os.PathLike]],
        var_name: str,
        *,
        x_coord: str = "lon",
        y_coord: str = "lat",
        time_coord: str = "time",
        start: Optional[Union[str, datetime, pd.Timestamp]] = None,
        end: Optional[Union[str, datetime, pd.Timestamp]] = None,
        **kwargs: Any,
    ):
        """
        Create an :class:`RunDirecotry` object from given input file(s)/ directory.

        The :class:`RunDirectory` object gathers all necessary information on the
        data that is stored in the run directory. Once loaded the most
        important meta data will be stored in the run directory for faster
        access the second time.

        Parameters
        ----------
        inp_files: os.PathLike, list[os.PathLike]
            Input filenames or directory that is opened.
        var_name: str
            Name of the variable that is tracked
        start: str, pandas.Timestamp, datetime.datetime (default: None)
            first time step that is considered, if none given the first
            of the dataset is considered.
        end: str, pandas.Timestamp, datetime.datetime (default: None)
            last time step that is considered, if none given the last
            of the dataset is considered.
        x_coord: str (default: lon)
            The name of the longitude vector/array
        x_coord: str (default: lat)
            The name of the latitude vector/array
        time_coord: str, default: time
            The name of the time variable
        kwargs:
            Additional keyword arguments that are passed to open the dataset
            with xarray

        Example
        -------

        .. execute_code::

            from tintx import RunDirectory
            run = RunDirectory.from_files("source/_static/data/CMORPH*.nc", "cmorph")

        """
        defaults: dict[str, Union[str, bool]] = dict(
            combine="by_coords", use_cftime=True,
        )
        defaults = dict()
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        _dset = xr.open_mfdataset(input_files, **kwargs)
        start_time = start or _dset[time_coord].isel({time_coord: 0})
        end_time = end or _dset[time_coord].isel({time_coord: -1})
        return cls(
            var_name,
            _dset.sel({time_coord: slice(start_time, end_time)}),
            x_coord=x_coord,
            y_coord=y_coord,
            time_coord=time_coord,
        )

    def __init__(
        self,
        var_name: str,
        dataset: Union[xr.Dataset, xr.DataArray],
        *,
        time_coord: str = "time",
        x_coord: str = "lon",
        y_coord: str = "lat",
    ):
        if isinstance(dataset, xr.DataArray):
            self.data = xr.Dataset({var_name: dataset})
        else:
            self.data = dataset
        self.lons = cast(xr.DataArray, dataset[x_coord])
        self.lats = cast(xr.DataArray, dataset[y_coord])
        self.var_name = var_name
        self.time = cast(xr.DataArray, self.data[time_coord])
        self.start = self.time[0]
        self.end = self.time[-1]
        super().__init__(var_name)

    @property
    def tracks(self) -> pd.DataFrame:
        """A pandas.DataFrame representation of the tracked cells."""
        return self._tracks

    def get_tracks(
        self,
        centre: Optional[tuple[float, float]] = None,
        leave_bar: bool = True,
        **tracking_parameters: float,
    ):
        """Obtains tracks given a list of data arrays. This is the
            primary method of the tracks class. This method makes use of all of the
            functions and helper classes defined above.

        Parameters
        ----------
        centre: tuple, default: None
            The centre of the radar station
        leave_bar: bool, default: True
            Leave the progress bar after tracking is finished
        **tracking_parameters: float
            Overwrite the tint tracking parameters with this values for this
            specific tracking. The defaults will be restored afterwards.
            See :py:mod:`tintx.config` for details


        See also
        --------
        :py:mod:`tintx.config`

        Returns
        -------
        int: Number of unique cells identified by the tracking

        Example
        -------

        .. execute_code::

            import xarray
            from tintx import RunDirectory
            dset = xarray.open_mfdataset("source/_static/data/CPOL*.nc",
                                         combine="by_coords")
            run = RunDirectory("radar_estimated_rain_rate",
                               dset,
                               x_coord="longitude",
                               y_coord="latitude")
            num_cells = run.get_tracks(field_thresh=0.01)

        """
        with tqdm(
            self.grids, total=self.time.size - 1, desc="Tracking", leave=leave_bar
        ) as pbar:
            with set_config(**tracking_parameters):
                return self._get_tracks(self.grids, centre, pbar=pbar)

    @property
    def grids(self) -> Iterator[GridType]:
        """Create dictionary holding longitude and latitude information."""
        yield from get_grids(
            self.data,
            (0, self.time.size),
            self.lons,
            self.lats,
            self.time,
            varname=self.var_name,
        )

    def animate(self, **kwargs):
        """
        Create a animation of tracked cells.

        Parameters
        ----------
        altitude : float
            The altitude to be plotted in meters.
        vmin, vmax : float, 0.01
            Minimum values for the colormap.
        vmax : float, 15
            Maximum values for the colormap.
        isolated_only: bool, default: False
            If true, only annotates uids for isolated objects. Only used in 'full'
            style.
        cmap: str, default: Blue
            Colormap used for plotting the tracked fields.
        tracers: bool, default: False
            Plot traces of animated cells
        dt: float, default: 0
            Time shift in hours that is applied to the data, this can be
            useful if data time data in in utc but should be displayed in
            another time zone.
        fps: int, default: 5
            Frames per second for output.
        ax: cartopy.mpl.geoaxes.GeoAxesSubplot, default: None
            Axes object that is used to create the animation. If None (default)
            a new axes object will be created.
        kwargs:
            Additional keyword arguments passed to the basemap plotting routine.

        Returns
        -------
        matplotlib.animation.FuncAnimation:
            See `FuncAnimation <https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_
            for more details.
        """
        return full_domain(self, self.grids, **kwargs)

    def plot_trajectories(self, **kwargs):
        """
        Plot traces of trajectories for each particle.

        This code is a fork of plot_traj method in the plot module from the
        trackpy project see http://soft-matter.github.io/trackpy for more details

        Parameters
        ----------
        label : boolean, default: False
            Set to True to write particle ID numbers next to trajectories.
        cmap : colormap, Default matplotlib.colormap.winter
            Colormap used to color different tracks
        ax: cartopy.mpl.geoaxes.GeoAxesSubplot, default: None
            Axes object that is used to create the animation. If None (default)
            a new axes object will be created.
        uids : list[str], default: None
            a preset of stroms to be drawn, instead of all (default)
        color : str, default: None
            A pre-defined color, if None (default) each track will be assigned a
            different color.
        thresh : float, default: -1
            Plot only trajectories with average intensity above this value.
        mintrace : int, default 2
            Minimum length of a trace to be plotted
        **plot_style :
            Additional keyword arguments passed through to the ``Axes.plot(...)``

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        return plot_traj(self.tracks, self.lons, self.lats, **kwargs)
