"""The :class:`RunDirectory` class is a convenience class to  access and apply
the tint tracking algorithm."""
from __future__ import annotations

import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import geopandas as gpd
import pandas as pd
import pyproj
import xarray as xr
from cartopy.crs import AzimuthalEquidistant, Projection
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib import pyplot as plt
from shapely import wkt
from tqdm.auto import tqdm

from .config import set as set_config
from .helpers import MetaData, convert_to_cftime, get_grids
from .tracks import Cell_tracks
from .types import GridType
from .visualization import full_domain, plot_traj

__all__ = ["RunDirectory"]


class RunDirectory(Cell_tracks):
    """
    Create an :class:`RunDirectory` object from a given xarray dataset.

    The :class:`RunDirectory` object gathers all necessary information of
    data that is stored in a ``xarray`` dataset. Once loaded the most
    important metadata will be stored in the run directory for faster
    access the second time.

    Parameters
    ----------
    dataset: xarray.DataArray, xarray.Dataset
        Input dataset/data array
    var_name: str
        Name of the variable that is tracked
    x_coord: str, default: lon
        Name of the X coordinate array/vector
    y_coord: str, default: lat
        Name of the Y coordinate array/vector
    time_coord: str, default: time
        Name of the time variable.

    Example
    -------
    .. execute_code::
        :hide_headers:

        import os
        import xarray
        from tintx import RunDirectory
        files = os.path.join(os.environ["FILE_PATH"], "CMORPH*.nc")
        dset = xarray.open_mfdataset(files, combine="by_coords")
        run = RunDirectory(dset.isel(time=slice(10, 150)), "prcip")

    """

    _files: Union[str, list[str]] = ""
    _parameters: dict[str, dict[str, float]] = {}
    """Nested dict of tracking parameters that are saved to file."""

    @classmethod
    def from_files(
        cls,
        input_files: Union[Path, str, list[str], list[Path], Iterator[Path]],
        var_name: str,
        *,
        x_coord: str = "lon",
        y_coord: str = "lat",
        time_coord: str = "time",
        start: Optional[Union[str, datetime, pd.Timestamp]] = None,
        end: Optional[Union[str, datetime, pd.Timestamp]] = None,
        crs: Union[str, pyproj.CRS, Projection] = "epsg:4326",
        **kwargs: Any,
    ) -> RunDirectory:
        """
        Create a :class:`RunDirectory` object from input file(s)/directory.

        The :class:`RunDirectory` object gathers all necessary information of
        the data that is stored in the run directory. Once loaded the most
        important metadata will be stored in the run directory for faster
        access the second time.

        Parameters
        ----------
        inp_files: os.PathLike, list[os.PathLike]
            Input filenames or directory that is opened.
        var_name: str
            Name of the variable that is tracked
        start: str, pandas.Timestamp, datetime.datetime (default: None)
            first time step that is considered, if None given the first
            time step in the data is considered.
        end: str, pandas.Timestamp, datetime.datetime (default: None)
            last time step that is considered, if None given the last
            time step in the data is considered.
        x_coord: str (default: lon)
            The name of the longitude vector/array, can be 1D or 2D
        x_coord: str (default: lat)
            The name of the latitude vector/array, can be 1D or 2D
        time_coord: str, default: time
            The name of the time variable
        crs: str, pyproj.Proj, cartopy.crs.Projection (default: "epsg:4326")
            pyproj/cartopy projection object or string defining the Coordinate
            Reference System (CRS). If "aeqd", CRS will be computed as
            AzimuthalEquidistant Projection from the geodetic radar site
            coordinates "longitude" and "latitude".
        kwargs:
            Additional keyword arguments that are passed to open the dataset
            with xarray

        Example
        -------
        .. execute_code::
            :hide_headers:

            import os
            from tintx import RunDirectory
            run = RunDirectory.from_files(
                os.path.join(os.environ["FILE_PATH"], "CMORPH*.nc"),
                "cmorph"
            )

        """
        defaults: dict[str, Union[str, bool]] = dict(
            combine="by_coords",
        )
        _files: Union[str, list[str]] = ""
        if isinstance(input_files, (str, Path)):
            _files = str(input_files)
        else:
            _files = [str(f) for f in input_files]
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        _dset = xr.open_mfdataset(_files, **kwargs)
        start_time = start or _dset[time_coord].isel({time_coord: 0})
        end_time = end or _dset[time_coord].isel({time_coord: -1})

        # create AEQD CRS
        if crs == "aeqd":
            crs = AzimuthalEquidistant(
                central_latitude=_dset.latitude.mean().values.item(),
                central_longitude=_dset.longitude.mean().values.item(),
            )

        return cls(
            _dset.sel({time_coord: slice(start_time, end_time)}),
            var_name,
            x_coord=x_coord,
            y_coord=y_coord,
            time_coord=time_coord,
            crs=crs,
            _files=_files,
        )

    def __init__(
        self,
        dataset: Union[xr.Dataset, xr.DataArray],
        var_name: str,
        *,
        time_coord: str = "time",
        x_coord: str = "lon",
        y_coord: str = "lat",
        crs: Union[str, pyproj.CRS, Projection] = "epsg:4326",
        _files: Union[list[str], str] = "",
    ) -> None:
        if isinstance(dataset, xr.DataArray):
            self.data = xr.Dataset({var_name: dataset})
        else:
            self.data = dataset
        self._files = _files
        self.lons = self.data[x_coord]
        self.lats = self.data[y_coord]
        self.var_name = var_name
        self.time = self.data[time_coord]
        self.start = convert_to_cftime(self.time.values[0])
        self.end = convert_to_cftime(self.time.values[-1])
        # transform crs to wkt in any case
        if isinstance(crs, Projection):
            self.crs = pyproj.CRS(crs.proj4_init).to_wkt()
        elif isinstance(crs, pyproj.CRS):
            self.crs = crs.to_wkt()
        elif isinstance(crs, str):
            self.crs = pyproj.CRS(crs).to_wkt()
        else:
            raise TypeError(
                "crs parameter must be a string or pyproj/cartopy " "projection"
            )
        self._metadata_reader = MetaData(
            self.data,
            self.var_name,
            {"x_coord": x_coord, "y_coord": y_coord, "time_coord": time_coord},
        )
        super().__init__(var_name)

    def get_tracks(
        self,
        centre: Optional[tuple[float, float]] = None,
        leave_bar: bool = True,
        flush: bool = True,
        **tracking_parameters: float,
    ) -> int:
        """Apply the ``tint`` tracking algorithm.

        This is the primary method of the :class:`RunDirectory` class. This
        method applies the tracking algorithm to the data and saves
        the tracked cells to a pandas ``DataFrame`` for analysis.

        Parameters
        ----------
        centre: tuple, default: None
            The centre of the radar station
        leave_bar: bool, default: True
            Leave the progress bar after tracking is finished
        flush: bool, default: True
            Flush old tracking data. If false, the tracks will be added
            to the existing ::method::`tracks` DataFrame.
        **tracking_parameters: float
            Overwrite the tint tracking parameters with these values for this
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
            :hide_headers:

            import os
            import xarray
            from tintx import RunDirectory
            files = os.path.join(os.environ["FILE_PATH"], "CPOL*.nc")
            dset = xarray.open_mfdataset(files, combine="by_coords")
            run = RunDirectory(dset,
                               "radar_estimated_rain_rate",
                               x_coord="longitude",
                               y_coord="latitude")
            num_cells = run.get_tracks(field_thresh=0.01)

        """
        if flush:
            self.reset_tracks()
        parameters = self._parameters.get(self._track_hash(), {})
        parameters.update(tracking_parameters)
        with tqdm(
            self.grids,
            total=self.time.size - 1,
            desc="Tracking",
            leave=leave_bar,
        ) as pbar:
            with set_config(**parameters) as cfg:
                num_tracks = self._get_tracks(self.grids, pbar, centre)
                self._parameters[self._track_hash()] = cfg.config.copy()
            return num_tracks

    @property
    def grids(self) -> Iterator[GridType]:
        """Iterator holding longitude/latitude/time and data."""
        yield from get_grids(
            self.data,
            (0, self.time.size),
            self.lons,
            self.lats,
            self.time,
            varname=self.var_name,
        )

    @property
    def tracks(self) -> pd.DataFrame:
        """Pandas ``DataFrame`` representation of the tracked cells."""
        # todo: cache GeoDataFrame like the normal DataFrame
        # only convert into GeoDataFrame if geometry-column is available
        # for backwards compatibility reasons
        if "geometry" in self._tracks.columns:
            return gpd.GeoDataFrame(self._tracks.copy(), crs=pyproj.CRS(self.crs))
        else:
            return self._tracks

    def save_tracks(self, output: Union[str, Path]) -> None:
        """Save tracked data to hdf5 table.

        Parameters
        ----------
        output: Path, str
            Filename where the output will be saved to.

        Example
        -------
        .. execute_code::
            :hide_headers:

            import os
            import xarray
            from tintx import RunDirectory
            files = os.path.join(os.environ["FILE_PATH"], "CPOL*.nc")
            dset = xarray.open_mfdataset(files, combine="by_coords")
            run = RunDirectory(dset,
                               "radar_estimated_rain_rate",
                               x_coord="longitude",
                               y_coord="latitude")
            num_cells = run.get_tracks(field_thresh=0.01)
            run.save_tracks("/tmp/output.hdf5")

        See Also
        --------
        :class:`from_dataframe`
        """
        output = Path(output).expanduser().absolute()
        metadata = self._metadata.copy()
        metadata["files"] = self._files
        with pd.HDFStore(output, "w") as hdf5:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    action="ignore", category=pd.errors.PerformanceWarning
                )
                tracks = self._tracks.copy()
                # only save track geometries and crs as wkt strings if geometry-column
                # is available for backwards compatibility reasons
                if "geometry" in tracks.columns:
                    tracks["geometry"] = self.tracks.geometry.map(str)
                    metadata["crs"] = self.crs
                hdf5.put("tintx_tracks", tracks)
                self._metadata_reader.save(hdf5)
            table = hdf5.get_storer("tintx_tracks")
            table.attrs.track_meta = metadata
            table.attrs.tracking_parameters = self._parameters[self._track_hash()]

    @classmethod
    def from_dataframe(
        cls, track_file: Union[str, Path], dataset: Optional[xr.Dataset] = None
    ) -> RunDirectory:
        """Create an instance of the :class:`RunDirectory` class from tintx tracks.

        Parameters
        ----------
        track_file: str, pathlib.Path
            Path to the hdf5 file containing the tint tracks
        dataset: xr.Dataset, default: None
            xarray Dataset holding the data of the tracked field.
            if None (default) the dataset will be re opened from the files.
            Note: this only works if the previous instance of the tracking
            class was instanciated with the :class:`from_files` method.

        .. note::
        If the dataset can't be opened an empty data set will be created
        instead.

        Example
        -------
        .. execute_code::
            :hide_code:
            :hide_headers:

            from pathlib import Path
            if not Path("/tmp/output.hdf5").is_file():
                from tintx import RunDirectory
                run = RunDirectory.from_files(
                    os.path.join(os.environ["FILE_PATH"], "CPOL*.nc"),
                    "radar_estimated_rain_rate", x_coord="x", y_coord="y"
                )
                run.get_tracks(min_size=4, field_thresh=0.1)
                run.save_tracks("/tmp/output.hdf5")
                run = RunDirectory.from_dataframe("/tmp/output.hdf5")

        ::

            from tintx import RunDirectory
            run = RunDirectory.from_dataframe("/tmp/output.hdf5")

        See Also
        --------
        :class:`save_tracks`
        """

        track_file = Path(track_file).expanduser().absolute()
        with pd.HDFStore(track_file) as hdf5:
            tracks = pd.read_hdf(hdf5, "tintx_tracks")
            metadata = hdf5.get_storer("tintx_tracks").attrs.track_meta.copy()
            parameters = hdf5.get_storer("tintx_tracks").attrs.tracking_parameters
            coord_dataset = MetaData.dataset_from_coords(hdf5)

        files = metadata.pop("files", "")
        var_name = metadata.pop("var_name", "")
        start = metadata.pop("start", None) or None
        end = metadata.pop("end", None) or None
        try:
            if dataset is None:
                cls_instance = cls.from_files(
                    files,
                    var_name,
                    start=start,
                    end=end,
                    **metadata,
                )
            else:
                cls_instance = cls(dataset, var_name, _files=files, **metadata)
        except Exception as error:
            warnings.warn(
                "Could not access original data, creating empty dataset "
                f"the reason for the failure was:\n{error}"
            )
            cls_instance = cls(coord_dataset, var_name, **metadata)
        cls_instance._metadata[cls_instance._track_hash()] = parameters
        # apply geometry conversion only if geometry-column is available
        # for backwards compatibility reasons
        if "geometry" in tracks.columns:
            tracks["geometry"] = tracks["geometry"].apply(wkt.loads)
            # add crs as wkt
            cls_instance.crs = metadata["crs"]
        cls_instance.reset_tracks(tracks)
        return cls_instance

    def get_parameters(
        self, tracks: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        """Get the parameters of given cell tracks.

        Parameters
        ----------
        tracks: pd.DataFrame, default: None
            The tracks ``DataFrame`` that is the result of the tuning
            parameters in question.

        Returns
        -------
        dict: dictionary holding the tuning parameters for the given tracks.

        Raises
        ------
        ValueError: if no parameters matching the input tracks ``DataFrame``
                    could be found.

        Example
        -------
        .. execute_code::
            :hide_code:
            :hide_headers:

            from pathlib import Path
            if not Path("/tmp/output.hdf5").is_file():
                from tintx import RunDirectory
                run = RunDirectory.from_files(
                    os.path.join(os.environ["FILE_PATH"], "CPOL*.nc"),
                    "radar_estimated_rain_rate", x_coord="x", y_coord="y"
                )
                run.get_tracks(min_size=4, field_thresh=0.1)
                run.save_tracks("/tmp/output.hdf5")
                parameters = run.get_parameters()

        ::

            from tintx import RunDirectory
            run = RunDirectory.from_dataframe("/tmp/output.hdf5")
            parameters = run.get_parameters()
        """

        try:
            return self._parameters[self._track_hash(tracks)]
        except KeyError as error:
            raise ValueError(
                "Could not retrieve parameters for given tracks"
            ) from error

    def _track_hash(self, tracks: Optional[pd.DataFrame] = None) -> str:
        if tracks is None:
            tracks = self._tracks
        df_hash = hashlib.sha512(tracks.to_string().encode("utf-8"))
        return df_hash.hexdigest()

    @property
    def _metadata(self) -> dict[str, Any]:
        return dict(
            x_coord=self.lons.name,
            y_coord=self.lats.name,
            time_coord=self.time.name,
            var_name=self.var_name,
            start=str(self.start),
            end=str(self.end),
        )

    def reset_tracks(self, tracks: Optional[pd.DataFrame] = None) -> None:
        """Override the tack data with a given DataFrame.

        Parameters
        ----------
        tracks: pd.DataFrame, default: None
            Tracking information used to override the current tarcks.
            If None (default) an empty DataFrame will be used.

        Example
        -------
        .. execute_code::
            :hide_headers:

            import os
            import xarray
            from tintx import RunDirectory
            files = os.path.join(os.environ["FILE_PATH"], "CPOL*.nc")
            dset = xarray.open_mfdataset(files, combine="by_coords")
            run = RunDirectory(dset,
                               "radar_estimated_rain_rate",
                               x_coord="longitude",
                               y_coord="latitude")
            num_cells = run.get_tracks(field_thresh=0.01)
            print(len(run.tracks))
            run.reset_tracks()
            print(len(run.tracks))
        """
        self.record = None
        self.counter = None
        self.current_objects = None
        if tracks is None:
            self._tracks = pd.DataFrame()
        else:
            self._tracks = tracks
        self._save()

    def animate(
        self,
        vmin: float = 0.01,
        vmax: float = 15,
        ax: Optional[GeoAxesSubplot] = None,
        cmap: Union[str, plt.cm] = "Blues",
        alt: Optional[float] = None,
        fps: float = 5,
        isolated_only: bool = False,
        tracers: bool = False,
        dt: float = 0,
        plot_style: Optional[dict[str, Union[float, int, str]]] = None,
    ) -> GeoAxesSubplot:
        """
        Create a animation of tracked cells.

        Parameters
        ----------
        vmin : float, default: 0.01
            Minimum values for the colormap.
        vmax : float, default: 15.0
            Maximum values for the colormap.
        isolated_only: bool, default: False
            If true, only annotates uids for isolated objects.
        cmap: str, default: Blues
            Colormap used for plotting the tracked fields.
        alt: float, default: None
            Altitude to be plotted in meters, for 3D data only.
        tracers: bool, default: False
            Plot traces of animated cells
        dt: float, default: 0
            Time shift in hours that is applied to the data, this can be
            useful if time data is in utc but should be displayed in
            another time zone.
        fps: int, default: 5
            Frames per second for output.
        ax: cartopy.mpl.geoaxes.GeoAxesSubplot, default: None
            Axes object that is used to create the animation. If None (default)
            a new axes object will be created.
        plot_style: dict
            Additional keyword arguments passed to the plotting routine.

        Returns
        -------
        matplotlib.animation.FuncAnimation:
            See `FuncAnimation <https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_
            for more details.

        Example
        -------

        .. execute_code::
            :hide_headers:

            from tintx import RunDirectory
            run = RunDirectory.from_files(
                os.path.join(os.environ["FILE_PATH"], "CPOL*.nc"),
                "radar_estimated_rain_rate", x_coord="x", y_coord="y"
            )
            run.get_tracks(min_size=4, field_thresh=2)
            anim = run.animate(vmax=3, fps=2, plot_style={"res": "10m", "lw":1})

        """
        return full_domain(
            self,
            self.grids,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cmap=cmap,
            alt=alt,
            fps=fps,
            isolated_only=isolated_only,
            tracers=tracers,
            dt=dt,
            plot_style=plot_style,
        )

    def plot_trajectories(
        self,
        label: bool = False,
        ax: Optional[GeoAxesSubplot] = None,
        uids: Optional[list[str]] = None,
        mintrace: int = 2,
        size: int = 50,
        thresh: float = -1.0,
        color: Optional[str] = None,
        plot_style: Optional[dict[str, Union[float, int, str]]] = None,
    ) -> GeoAxesSubplot:
        """
        Plot traces of trajectories for each cell track.

        This code is a fork of ``plot_traj`` method in the plot module from the
        trackpy project see http://soft-matter.github.io/trackpy for more
        details

        Parameters
        ----------
        label : boolean, default: False
            Set to True to write cell uids next to trajectories.
        cmap : colormap, Default matplotlib.colormap.winter
            Colormap used to color different tracks
        ax: cartopy.mpl.geoaxes.GeoAxesSubplot, default: None
            Axes object that is used to create the animation. If None (default)
            a new axes object will be created.
        uids : list[str], default: None
            a preset of stroms to be drawn, instead of all (default)
        color : str, default: None
            A pre-defined color, if None (default) each track will be assigned
            a different color.
        thresh : float, default: -1
            Plot only trajectories with average intensity above this value.
        mintrace : int, default 2
            Minimum length of a trace to be plotted
        plot_style: dict
            Additional keyword arguments passed through to the
            ``Axes.plot(...)``

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot


        Example
        -------
        .. execute_code::
            :hide_code:
            :hide_headers:

            from pathlib import Path
            if not Path("/tmp/output.hdf5").is_file():
                from tintx import RunDirectory
                run = RunDirectory.from_files(
                    os.path.join(os.environ["FILE_PATH"], "CPOL*.nc"),
                    "radar_estimated_rain_rate", x_coord="x", y_coord="y"
                )
                run.get_tracks(min_size=4, field_thresh=0.1)
                run.save_tracks("/tmp/output.hdf5")
                run = RunDirectory.from_dataframe("/tmp/output.hdf5")

        ::

            from tintx import RunDirectory
            run = RunDirectory.from_dataframe("/tmp/output.hdf5")
            ax = run.plot_trajectories(thresh=2, plot_style={"ms":25, "lw":1})
        """
        return plot_traj(
            self.tracks,
            self.lons,
            self.lats,
            ax=ax,
            label=label,
            uids=uids,
            mintrace=mintrace,
            size=size,
            thresh=thresh,
            color=color,
            plot_style=plot_style,
        )
