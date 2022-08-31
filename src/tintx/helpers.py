"""
tint.helpers
============

"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Iterator, Optional, Union, cast

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from .grid_utils import parse_grid_datetime, get_grid_size, GridType


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

    def __init__(self):
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


def convert_to_cftime(time: Union[np.datetime64, cftime.datetime]) -> cftime.datetime:
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

    dims = group.variables[varname].dims
    for s in range(slices[0], slices[-1]):
        time = convert_to_cftime(
            cast(Union[np.datetime64, cftime.datetime], times[s].values)
        )
        try:
            data = group.variables[varname][s].values
        except AttributeError:
            data = group.variables[varname][s]
        if len(data.shape) < 3:
            data = data[np.newaxis, :]
        mask_data = np.ma.masked_invalid(data)
        yield GridType(
            x=group[dims[-1]],
            y=group[dims[-2]],
            lon=lon,
            lat=lat,
            data=mask_data,
            time=time,
        )
