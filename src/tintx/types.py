"""Collection of all type definitions used for type hinting."""
from __future__ import annotations
from typing import Union, NamedTuple, List
from typing_extensions import TypedDict

import cftime
import numpy as np
import xarray as xr

SearchPredictType = TypedDict(
    "SearchPredictType",
    {
        "x1": int,
        "x2": int,
        "y1": int,
        "y2": int,
        "center_pred": np.ndarray,
        "valid": bool,
    },
)
"""Type to store predictions for the objects in the next time step."""

ObjectPropType = TypedDict(
    "ObjectPropType",
    {
        "id1": List[int],
        "center": List[float],
        "grid_x": List[float],
        "grid_y": List[float],
        "area": List[float],
        "field_max": List[float],
        "field_mean": List[float],
        "lon": List[float],
        "lat": List[float],
        "isolated": np.ndarray,
        "ok_items": List[bool],
    },
)

CurrentObjectType = TypedDict(
    "CurrentObjectType",
    {
        "id1": np.ndarray,
        "uid": np.ndarray,
        "id2": np.ndarray,
        "obs_num": np.ndarray,
        "origin": np.ndarray,
    },
)
"""Parameter type describing properties of a tracked object."""

ConfigType = TypedDict(
    "ConfigType",
    {
        "FIELD_THRESH": float,
        "ISO_THRESH": float,
        "ISO_SMOOTH": float,
        "MIN_SIZE": float,
        "FLOW_MARGIN": float,
        "MAX_DISPARITY": float,
        "MAX_FLOW_MAG": float,
        "MAX_SHIFT_DISP": float,
        "SEARCH_MARGIN": float,
        "GS_ALT": float,
    },
)
"""The Parameter type defines the default values and types of the tint
tracking tuning parameters."""

GridType = NamedTuple(
    "GridType",
    [
        ("x", xr.DataArray),
        ("y", xr.DataArray),
        ("lon", xr.DataArray),
        ("lat", xr.DataArray),
        ("time", Union[np.datetime64, cftime.datetime]),
        ("data", np.ma.core.MaskedArray),
    ],
)
"""Type that describes the grid/time/data information for a time step."""
