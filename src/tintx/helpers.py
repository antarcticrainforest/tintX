"""
tint.helpers
============

"""
from __future__ import annotations
from datetime import datetime
import string
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

    def next_cid(self, pid: str) -> str:
        """Returns parent uid with appended letter to denote child."""
        if pid in self.cid:
            self.cid[pid] += 1
        else:
            self.cid[pid] = 0
        letter = string.ascii_lowercase[self.cid[pid]]
        return pid + letter


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
        self.time = None
        self.interval = None
        self.interval_ratio = None
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

    def record_shift(self, corr, gl_shift, l_heads, local_shift, case):
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

    def add_uids(self, current_objects):
        """Because of the chronology of the get_tracks process, object uids
        cannot be added to the shift record at the time of correction, so they
        must be added later in the process."""
        if len(self.new_shifts) > 0:
            self.new_shifts["uid"] = current_objects["uid"]
            self.new_shifts.set_index(["scan", "uid"], inplace=True)
            self.shifts = pd.concat([self.shifts, self.new_shifts])
            self.new_shifts = pd.DataFrame()

    def update_scan_and_time(self, grid_obj1, grid_obj2=None):
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
        self.interval = time2 - self.time
        try:
            inter_val_s = self.interval.total_seconds()
        except AttributeError:
            inter_val_s = self.interval / np.timedelta64(1, "s")
        if old_diff is not None:
            try:
                diff = old_diff.total_seconds()
            except AttributeError:
                diff = old_diff / np.timedelta64(1, "s")
            self.interval_ratio = inter_val_s / diff


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
        time = cast(Union[np.datetime64, cftime.datetime], times[s].values)
        try:
            data = group.variables[varname][s].values
        except AttributeError:
            data = group.variables[varname][s]
        if len(data.shape) < 3:
            data = data[np.newaxis, :]
        if isinstance(time, np.datetime64):
            t = datetime.fromtimestamp(time.astype("O") / 1e9)
            time = cftime.DatetimeGregorian(
                t.year, t.month, t.day, t.hour, t.minute, t.second
            )
        yield GridType(
            x=group[dims[-1]],
            y=group[dims[-2]],
            lon=lon,
            lat=lat,
            data=np.ma.masked_invalid(data),
            time=time,
        )
