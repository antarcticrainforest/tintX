"""
tint.grid_utils
===============

Tools for pulling data from reading data.


"""

from __future__ import annotations
from typing import Union

import cftime
import numpy as np
import pandas as pd
from scipy import ndimage

from .types import ConfigType, GridType


def parse_grid_datetime(
    grid_obj: GridType,
) -> Union[np.datetime64, cftime.datetime]:
    """Obtains datetime object from the data dictionary."""
    return grid_obj.time


def get_grid_size(grid_obj: GridType) -> np.ndarray:
    """Calculates grid size per dimension given a grid object."""
    return np.array(grid_obj.data.shape)


def get_radar_info(radar: tuple[float, float]) -> dict[str, float]:
    """Return the longitude/latitude of the radar."""
    info = {"radar_lon": radar[0], "radar_lat": radar[1]}
    return info


def get_grid_alt(
    grid_size: tuple[float, float], alt_meters: Union[float, int] = 1500.0
) -> int:
    """Returns z-index closest to alt_meters."""
    return int(np.round(alt_meters / grid_size[0]))


def get_vert_projection(
    grid: np.ndarray, thresh: Union[float, int] = 40
) -> np.ndarray:
    """Returns boolean vertical projection from grid."""
    return np.any(grid > thresh, axis=0)


def get_filtered_frame(
    grid: np.ndarray, min_size: float, thresh: Union[int, float]
) -> np.ndarray:
    """Returns a labeled frame from gridded radar data. Smaller objects
    are removed and the rest are labeled."""
    echo_height = get_vert_projection(grid, thresh)
    labeled_echo = ndimage.label(echo_height)[0]
    frame = clear_small_echoes(labeled_echo, min_size)
    return frame


def clear_small_echoes(label_image: np.ndarray, min_size: float) -> np.ndarray:
    """Takes in binary image and clears objects less than min_size."""
    flat_image = pd.Series(label_image.flatten())
    flat_image = flat_image[flat_image > 0]
    size_table = flat_image.value_counts(sort=False)
    small_objects = size_table.keys()[size_table < min_size]

    for obj in small_objects:
        label_image[label_image == obj] = 0
    label_image = ndimage.label(label_image)
    return label_image[0]


def extract_grid_data(
    grid_obj: GridType, params: ConfigType
) -> tuple[np.ndarray, np.ndarray]:
    """Get filtered grid frame and raw grid slice at global shift altitude."""
    masked = grid_obj.data.filled(0)
    raw = masked[0, :, :]
    frame = get_filtered_frame(masked, params["MIN_SIZE"], params["FIELD_THRESH"])
    return raw, frame
