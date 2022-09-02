"""
tint.helpers
============

"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Iterator, Optional, Union, cast

import cftime
from dask import array as dask_array
import numpy as np
import pandas as pd
import xarray as xr
from .grid_utils import parse_grid_datetime, get_grid_size, GridType


class MetaData:
    """Class that writes meta data to an HDF5 file."""

    time_units: str = "seconds since 1970-01-01T00:00:00"

    def __init__(
        self,
        dataset: xr.Dataset,
        variable: str,
        coords: dict[str, str],
    ) -> None:

        self._time = coords["time_coord"]
        self._x_coord = coords["x_coord"]
        self._y_coord = coords["y_coord"]

        self.dataset = dataset
        self.x_coord = dataset[coords["x_coord"]].values
        self.y_coord = dataset[coords["y_coord"]].values
        self.variable = variable

    @property
    def dims(self) -> tuple[str, ...]:
        """Get the dimensions of the DataArray."""
        return tuple(map(str, self.dataset[self.variable].dims))

    @property
    def coords(self) -> pd.DataFrame:
        """Get the coordinates of the DataArray in a pandas DataFrame."""
        coords: dict[str, pd.Series] = {
            str(v): pd.Series(self.dataset[v].values)
            for v in self.dataset[self.variable].dims
            if v != self._time
        }
        coords[self._time] = pd.Series(self.time)
        return pd.DataFrame(coords)

    @property
    def time(self) -> np.ndarray:
        """Get a number representation of the time vector in."""
        return np.array(
            [self._to_num(t) for t in self.dataset[self._time].values]
        )

    def _to_num(self, time: Union[cftime.datetime, np.datetime64]) -> int:
        cf_time = convert_to_cftime(time)
        return cftime.date2num(
            cf_time, units=self.time_units, calendar=cf_time.calendar
        )

    @property
    def calendar(self) -> str:
        cf_time = convert_to_cftime(self.dataset[self._time].values[0])
        return cf_time.calendar

    def to_dataframe(self, var_name: str) -> Union[pd.DataFrame, pd.Series]:
        """Convert the values of a dataset variable to a dataframe."""
        if len(self.dataset[var_name].shape) == 1:
            return pd.Series(self.dataset[var_name].values)
        return pd.DataFrame(self.dataset[var_name].values)

    def save(self, buffer: pd.io.pytables.Table) -> None:
        """Save attrs to a hdf5 store object."""
        self.coords.to_hdf(buffer, "dims")
        self.to_dataframe(self._x_coord).to_hdf(buffer, "x_coord")
        self.to_dataframe(self._y_coord).to_hdf(buffer, "y_coord")
        x_attrs = self.dataset[self._x_coord].attrs
        y_attrs = self.dataset[self._y_coord].attrs
        x_attrs["short_name"] = self._x_coord
        y_attrs["short_name"] = self._y_coord
        y_attrs["dims"] = self.dataset[self._y_coord].dims
        x_attrs["dims"] = self.dataset[self._x_coord].dims
        buffer.get_storer("x_coord").attrs.attrs = x_attrs
        buffer.get_storer("y_coord").attrs.attrs = y_attrs
        buffer.get_storer("x_coord").attrs.attrs = x_attrs
        buffer.get_storer("y_coord").attrs.attrs = y_attrs
        dim_attrs = {v: self.dataset[v].attrs for v in self.dims}
        dim_attrs[self._time].setdefault("calendar", self.calendar)
        dim_attrs[self._time]["units"] = self.time_units
        var_attrs = self.dataset[self.variable].attrs
        var_attrs["time_coord"] = self._time
        var_attrs["short_name"] = self.variable
        buffer.get_storer("dims").attrs.dim_attrs = dim_attrs
        buffer.get_storer("dims").attrs.var_attrs = var_attrs
        buffer.get_storer("dims").attrs.dims = self.dims

    @staticmethod
    def dataset_from_coords(buffer: pd.io.pytables.Table) -> xr.Dataset:
        """Create a xarray Dataset from metadata save in a pytable."""

        dims_df = buffer.get("dims")
        storer_obj = buffer.get_storer("dims")
        x_coords_attrs = buffer.get_storer("x_coord").attrs.attrs.copy()
        y_coords_attrs = buffer.get_storer("y_coord").attrs.attrs.copy()
        var_attrs = storer_obj.attrs.var_attrs.copy()
        var_attrs[
            "history"
        ] = f"{datetime.now().isoformat()}: Created empty dataset"
        coord_attrs = storer_obj.attrs.dim_attrs.copy()
        time_coord = var_attrs.pop("time_coord")
        metadata = buffer.get_storer("tintx_tracks").attrs.track_meta.copy()
        shape = []
        data = {}
        coords = [x_coords_attrs["short_name"], y_coords_attrs["short_name"]]
        for dim in storer_obj.attrs.dims:
            data[dim] = xr.DataArray(
                dims_df[dim].dropna().values,
                name=dim,
                dims=(dim,),
                attrs=coord_attrs[dim],
            )
            shape.append(data[dim].shape[0])
        data[time_coord].data = cftime.num2date(
            data[time_coord].data,
            coord_attrs[time_coord]["units"],
            coord_attrs[time_coord]["calendar"],
        )
        data[metadata["var_name"]] = xr.DataArray(
            dask_array.zeros(shape),
            name=metadata["var_name"],
            attrs=var_attrs,
            dims=storer_obj.attrs.dims,
        ).chunk({time_coord: 1})
        print(buffer.get("x_coord").values.shape, x_coords_attrs["dims"])
        data[coords[0]] = xr.DataArray(
            buffer.get("x_coord").values,
            dims=x_coords_attrs.pop("dims"),
            name=coords[0],
            attrs=x_coords_attrs,
        )
        data[coords[1]] = xr.DataArray(
            buffer.get("y_coord").values,
            dims=y_coords_attrs.pop("dims"),
            name=coords[1],
            attrs=y_coords_attrs,
        )
        return xr.Dataset(data).set_coords(coords)


class Counter:
    """
    Counter objects generate and keep track of unique cell ids.
    Currently only the uid attribute is used, but this framework can
    accomodate further development of merge/split detection.

    Attributes
    ----------
    uid : int
        Last uid assigned.
    cid : dict
        Record of cell genealogy.

    """

    def __init__(self) -> None:
        """uid is an integer that tracks the number of independently formed
        cells. The cid dictionary keeps track of 'children' --i.e., cells that
        have split off from another cell."""
        self.uid: int = -1
        self.cid: dict[str, int] = {}

    def next_uid(self, count: int = 1) -> np.ndarray:
        """Incremented for every new independently formed cell."""
        new_uids = self.uid + np.arange(count) + 1
        self.uid += count
        return np.array([str(uid) for uid in new_uids])


class Record:
    """
    Record objects keep track of information related to the shift correction
    process.

    Attributes
    ----------
    scan : int
        Index of the current scan.
    time : datetime
        Time corresponding to scan.
    interval : timedelta
        Temporal difference between the next scan and the current scan.
    interval_ratio : float
        Ratio of current interval to previous interval.
    grid_size : array of floats
        Length 3 array containing z, y, and x mesh size in meters.
    shifts : dataframe
        Records inputs of shift correction process. See matching.correct_shift.
    new_shfits : dataframe
        Row of new shifts to be added to shifts dataframe.
    correction_tally : dict
        Tallies correction cases for performance analysis.


    Shift Correction Case Guide:
    case0 - new object, local_shift and global_shift disagree, returns global
    case1 - new object, returns local_shift
    case2 - local disagrees with last head and global, returns last head
    case3 - local disagrees with last head, returns local
    case4 - local and last head agree, returns average of both
    case5 - flow regions empty or at edge of frame, returns global_shift

    """

    def __init__(self, grid_obj: GridType) -> None:
        self.scan = -1
        self.time: Optional[Union[np.datetime64, cftime.datetime]] = None
        self.interval: Optional[Union[timedelta, np.timedelta64]] = None
        self.interval_ratio: Optional[float] = None
        self.grid_size = get_grid_size(grid_obj)
        self.shifts = pd.DataFrame()
        self.new_shifts = pd.DataFrame()
        self.correction_tally = {
            "case0": 0,
            "case1": 0,
            "case2": 0,
            "case3": 0,
            "case4": 0,
            "case5": 0,
        }

    def count_case(self, case_num: int) -> None:
        """Updates correction_tally dictionary. This is used to monitor the
        shift correction process."""
        self.correction_tally["case" + str(case_num)] += 1

    def record_shift(
        self,
        corr: float,
        gl_shift: float,
        l_heads: Optional[float],
        local_shift: float,
        case: int,
    ) -> None:
        """Records corrected shift, phase shift, global shift, and last
        heads per object per timestep. This information can be used to
        monitor and refine the shift correction algorithm in the
        correct_shift function."""
        if l_heads is None:
            l_heads = np.ma.array([-999, -999], mask=[True, True])

        new_shift_record = pd.DataFrame()
        new_shift_record["scan"] = [self.scan]
        new_shift_record["uid"] = ["uid"]
        new_shift_record["corrected"] = [corr]
        new_shift_record["global"] = [gl_shift]
        new_shift_record["last_heads"] = [l_heads]
        new_shift_record["phase"] = [local_shift]
        new_shift_record["case"] = [case]

        self.new_shifts = pd.concat([self.new_shifts, new_shift_record])

    def add_uids(self, current_objects: pd.DataFrame) -> None:
        """Because of the chronology of the get_tracks process, object uids
        cannot be added to the shift record at the time of correction, so they
        must be added later in the process."""
        if len(self.new_shifts) > 0:
            self.new_shifts["uid"] = current_objects["uid"]
            self.new_shifts.set_index(["scan", "uid"], inplace=True)
            self.shifts = pd.concat([self.shifts, self.new_shifts])
            self.new_shifts = pd.DataFrame()

    def update_scan_and_time(
        self, grid_obj1: GridType, grid_obj2: Optional[GridType] = None
    ) -> None:
        """Updates the scan number and associated time. This information is
        used for obtaining object properties as well as for the interval ratio
        correction of last_heads vectors."""
        self.scan += 1
        self.time = parse_grid_datetime(grid_obj1)
        if grid_obj2 is None:
            # tracks for last scan are being written
            return
        time2 = parse_grid_datetime(grid_obj2)
        old_diff = self.interval
        self.interval = cast(Union[timedelta, np.timedelta64], time2 - self.time)
        inter_val_s = get_interval(self.interval)
        if old_diff is not None:
            diff = get_interval(old_diff)
            self.interval_ratio = inter_val_s / diff


def get_interval(interval: Union[timedelta, np.timedelta64]) -> float:
    """Get a timeinterval in total seconds."""

    if isinstance(interval, np.timedelta64):
        return float(interval / np.timedelta64(1, "s"))
    return interval.total_seconds()


def convert_to_cftime(
    time: Union[np.datetime64, cftime.datetime]
) -> cftime.datetime:
    """Convert a dateim to a cftime object."""
    if isinstance(time, np.datetime64):
        t = datetime.fromisoformat(str(time).partition(".")[0])
        return cftime.DatetimeGregorian(
            t.year, t.month, t.day, t.hour, t.minute, t.second
        )
    return time


def get_grids(
    group: xr.Dataset,
    slices: tuple[int, int],
    lon: xr.DataArray,
    lat: xr.DataArray,
    times: xr.DataArray,
    varname: str = "rain_rate",
) -> Iterator[GridType]:

    dims = group[varname].dims
    for s in range(slices[0], slices[-1]):
        time = convert_to_cftime(
            cast(Union[np.datetime64, cftime.datetime], times[s].values)
        )
        mask_data = np.ma.masked_invalid(group[varname][s].values)
        if len(mask_data.shape) < 3:
            mask_data = mask_data[np.newaxis, :]
        yield GridType(
            x=group[dims[-1]],
            y=group[dims[-2]],
            lon=lon,
            lat=lat,
            data=mask_data,
            time=time,
        )
