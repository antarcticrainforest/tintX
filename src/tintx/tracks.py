"""
tint.tracks
===========

Cell_tracks class.

"""

from __future__ import annotations
import copy
from typing import cast, Dict, Iterator, Optional

import pandas as pd
import numpy as np
from tqdm.std import tqdm

from .grid_utils import get_grid_size, get_radar_info, extract_grid_data
from .helpers import Record, Counter, GridType
from .phase_correlation import get_global_shift
from .matching import get_pairs
from .objects import init_current_objects, update_current_objects
from .objects import get_object_prop, write_tracks
from .config import config as tint_config
from .types import ConfigType


class Cell_tracks:
    """
    This is the main class in the module. It allows tracks
    objects to be built using lists of data arrays.

    Attributes
    ----------
    params : dict
        Parameters for the tracking algorithm.
    field : str
        String specifying data variable to be used for tracking. Default is
        'reflectivity'.
    grid_size : array
        Array containing z, y, and x mesh size in meters respectively.
    last_grid : Grid
        Contains the most recent grid object tracked. This is used for dynamic
        updates.
    counter : Counter
        See Counter class.
    record : Record
        See Record class.
    current_objects : dict
        Contains information about objects in the current scan.
    _tracks : DataFrame

    _saved_record : Record
        Deep copy of Record at the penultimate scan in the sequence. This and
        following 2 attributes used for link-up in dynamic updates.
    _saved_counter : Counter
        Deep copy of Counter.
    _saved_objects : dict
        Deep copy of current_objects.

    """

    def __init__(self, field: str = "reflectivity"):

        self.field = field
        self.grid_size: Optional[np.ndarray] = None
        self.radar_info: Optional[dict[str, float]] = None
        self.last_grid: Optional[GridType] = None
        self.counter: Optional[Counter] = None
        self.record: Optional[Record] = None
        self.current_objects: Optional[dict[str, np.ndarray]] = None
        self._tracks = pd.DataFrame()

        self._saved_record: Optional[Record] = None
        self._saved_counter: Optional[Counter] = None
        self._saved_objects: Optional[dict[str, np.ndarray]] = None

    @property
    def params(self) -> ConfigType:
        """Get the tracking parameters."""
        return cast(ConfigType, tint_config)

    def _save(self) -> None:
        """Saves deep copies of record, counter, and current_objects."""
        self._saved_record = copy.deepcopy(self.record)
        self._saved_counter = copy.deepcopy(self.counter)
        self._saved_objects = copy.deepcopy(self.current_objects)

    def _load(self) -> None:
        """Loads saved copies of record, counter, and current_objects. If new
        tracks are appended to existing tracks via the get_tracks method, the
        most recent scan prior to the addition must be overwritten to link up
        with the new scans. Because of this, record, counter and
        current_objects must be reverted to their state in the penultimate
        iteration of the loop in get_tracks. See get_tracks for details."""
        self.record = self._saved_record
        self.counter = self._saved_counter
        self.current_objects = self._saved_objects

    @property
    def tracks(self) -> pd.DataFrame:
        """A pandas.DataFrame representation of the tracked cells."""
        return self._tracks

    def _get_tracks(
        self,
        grids: Iterator[GridType],
        pbar: tqdm,
        centre: Optional[tuple[float, float]] = None,
    ) -> int:
        raw2: Optional[np.ndarray] = None
        if self.record is None:
            # tracks object being initialized
            grid_obj2 = next(grids)
            self.grid_size = get_grid_size(grid_obj2)
            if centre is None:
                xgrid = grid_obj2.x.values
                ygrid = grid_obj2.y.values
                if len(xgrid.shape) == 2:
                    x_c = xgrid[xgrid.shape[0] // 2][xgrid.shape[1] // 2]
                else:
                    x_c = xgrid[xgrid.shape[0] // 2]
                if len(ygrid.shape) == 2:
                    y_c = ygrid[ygrid.shape[0] // 2][ygrid.shape[1] // 2]
                else:
                    y_c = ygrid[ygrid.shape[0] // 2]
                x_c = cast(float, x_c)
                y_c = cast(float, y_c)
                self.radar_info = get_radar_info((x_c, y_c))
            else:
                self.radar_info = get_radar_info(centre)
            self.counter = Counter()
            self.record = Record(grid_obj2)
        else:
            # tracks object being updated
            grid_obj2 = cast(GridType, self.last_grid)
            self._tracks.drop(self.record.scan + 1)  # last scan is overwritten

        new_rain = bool(self.current_objects is None)
        stop_iteration = bool(grid_obj2 is None)
        raw2, frame2 = extract_grid_data(grid_obj2, self.params)
        while not stop_iteration:
            pbar.update()
            grid_obj1 = grid_obj2
            raw1 = raw2
            frame1 = frame2

            try:
                grid_obj2 = next(grids)
            except StopIteration:
                stop_iteration = True

            if not stop_iteration:
                self.record.update_scan_and_time(grid_obj1, grid_obj2)
                raw2, frame2 = extract_grid_data(grid_obj2, self.params)
            else:
                # setup to write final scan
                self._save()
                self.last_grid = grid_obj1
                self.record.update_scan_and_time(grid_obj1)
                raw2 = None
                frame2 = np.zeros_like(frame1)

            if np.nanmax(frame1) == 0:
                new_rain = True
                self.current_objects = None
                continue
            global_shift = cast(float, get_global_shift(raw1, raw2))
            pairs = cast(
                np.ndarray,
                get_pairs(
                    frame1,
                    frame2,
                    global_shift,
                    self.current_objects,
                    self.record,
                    self.params,
                ),
            )
            if new_rain:
                # first nonempty scan after a period of empty scans
                self.current_objects, self.counter = init_current_objects(
                    frame1, frame2, pairs, cast(Counter, self.counter)
                )
                new_rain = False
            else:
                self.current_objects, self.counter = update_current_objects(
                    frame1,
                    frame2,
                    pairs,
                    cast(Dict[str, np.ndarray], self.current_objects),
                    cast(Counter, self.counter),
                )
            obj_props = get_object_prop(
                frame1, grid_obj1, self.record, self.params
            )

            self.record.add_uids(self.current_objects)
            self._tracks = write_tracks(
                self._tracks, self.record, self.current_objects, obj_props
            )
            del grid_obj1, raw1, frame1, global_shift, pairs, obj_props
            # scan loop end
        self._load()
        ncells = 0
        if len(self._tracks):
            ncells = self._tracks.index.get_level_values(1).astype(int).max() + 1
        return ncells
