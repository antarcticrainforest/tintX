"""
tint.helpers
============

"""

import string

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num
from .grid_utils import parse_grid_datetime, get_grid_size


class Counter(object):
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
        """ uid is an integer that tracks the number of independently formed
        cells. The cid dictionary keeps track of 'children' --i.e., cells that
        have split off from another cell. """
        self.uid = -1
        self.cid = {}

    def next_uid(self, count=1):
        """ Incremented for every new independently formed cell. """
        new_uids = self.uid + np.arange(count) + 1
        self.uid += count
        return np.array([str(uid) for uid in new_uids])

    def next_cid(self, pid):
        """ Returns parent uid with appended letter to denote child. """
        if pid in self.cid.keys():
            self.cid[pid] += 1
        else:
            self.cid[pid] = 0
        letter = string.ascii_lowercase[self.cid[pid]]
        return pid + letter


class Record(object):
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

    def __init__(self, grid_obj):
        self.scan = -1
        self.time = None
        self.interval = None
        self.interval_ratio = None
        self.grid_size = get_grid_size(grid_obj)
        self.shifts = pd.DataFrame()
        self.new_shifts = pd.DataFrame()
        self.correction_tally = {'case0': 0, 'case1': 0, 'case2': 0,
                                 'case3': 0, 'case4': 0, 'case5': 0}

    def count_case(self, case_num):
        """ Updates correction_tally dictionary. This is used to monitor the
        shift correction process. """
        self.correction_tally['case' + str(case_num)] += 1

    def record_shift(self, corr, gl_shift, l_heads, local_shift, case):
        """ Records corrected shift, phase shift, global shift, and last
        heads per object per timestep. This information can be used to
        monitor and refine the shift correction algorithm in the
        correct_shift function. """
        if l_heads is None:
            l_heads = np.ma.array([-999, -999], mask=[True, True])

        new_shift_record = pd.DataFrame()
        new_shift_record['scan'] = [self.scan]
        new_shift_record['uid'] = ['uid']
        new_shift_record['corrected'] = [corr]
        new_shift_record['global'] = [gl_shift]
        new_shift_record['last_heads'] = [l_heads]
        new_shift_record['phase'] = [local_shift]
        new_shift_record['case'] = [case]

        self.new_shifts = self.new_shifts.append(new_shift_record)

    def add_uids(self, current_objects):
        """ Because of the chronology of the get_tracks process, object uids
        cannot be added to the shift record at the time of correction, so they
        must be added later in the process. """
        if len(self.new_shifts) > 0:
            self.new_shifts['uid'] = current_objects['uid']
            self.new_shifts.set_index(['scan', 'uid'], inplace=True)
            self.shifts = self.shifts.append(self.new_shifts)
            self.new_shifts = pd.DataFrame()

    def update_scan_and_time(self, grid_obj1, grid_obj2=None):
        """ Updates the scan number and associated time. This information is
        used for obtaining object properties as well as for the interval ratio
        correction of last_heads vectors. """
        self.scan += 1
        self.time = parse_grid_datetime(grid_obj1)
        if grid_obj2 is None:
            # tracks for last scan are being written
            return
        time2 = parse_grid_datetime(grid_obj2)
        old_diff = self.interval
        self.interval = time2 - self.time
        if old_diff is not None:
            self.interval_ratio = self.interval.total_seconds()/old_diff.total_seconds()


def spl(present, time):
    out = []
    start = True
    a = ''.join(list(present.astype(int).astype(str)))
    b = list(present.astype(int).astype(str))
    ii = 0
    for k, g in groupby(a):
        gg = list(g)
        if len(gg) == 1:
            b[ii] = str(1 - int(gg[0]))
        ii += len(gg)
    ii = 0
    for k, g in groupby(''.join(b)):
        G = list(g)
        if k == '1':
            kk = 0
            for i in G:
                d1 = time[kk+ii]
                try:
                    d2 = time[kk+ii+1]
                except IndexError:
                    break
                if (d2 - d1) > 24*60**2:
                    break
                kk += 1
            if (ii+kk-1) - ii > 5:
                out.append((ii, ii+kk-1))
        ii += len(G)
    return out


def get_grids(group, slices, lon, lat, varname='rain_rate', times=None):

    x = group.variables[varname].shape[1]
    y = group.variables[varname].shape[2]

    try:
        dims = group.variables[varname].dimensions[1:]
    except AttributeError:
        dims = group.variables[varname].dims[1:]
    for s in range(slices[0], slices[-1]+1):
        try:
            time = times[s]
        except TypeError:
            time = num2date(group.variables['time'][s],
                            group['time'].units)
        try:
            data = group.variables[varname][s].values
        except AttributeError:
            data = group.variables[varname][s]
        
        yield {'x': group.variables[dims[-1]],
               'y': group.variables[dims[-2]],
               'lon': lon, 'lat': lat,
               'data': np.ma.masked_invalid(data.reshape(1, x, y)),
               'time': time}


def get_times(time, start=None, end=None, isfile=None):
    '''Get the start and end index for a given period'''

    if type(end) == type('a') and type(start) == type('a'):
        end = pd.DatetimeIndex([end]).to_pydatetime()[0]
        start = pd.DatetimeIndex([start]).to_pydatetime()[0]
        start = date2num([start], time.units)
        end = date2num([end], time.units)
        e_idx = np.argmin(np.fabs(time[:] - end))+1
        s_idx = np.argmin(np.fabs(time[:] - start))
        return [(s_idx, e_idx)]
    elif type(isfile) == type(None):
        isifle = np.ones_like(time.shape[0])
        return spl(isfile, time)
