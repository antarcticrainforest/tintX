"""
tint.objects
============

Functions for managing and recording object properties.

"""
from __future__ import annotations
from typing import Union

import numpy as np
import pandas as pd
from scipy import ndimage
from .grid_utils import get_filtered_frame
from .helpers import Counter, Record
from .types import ConfigType, GridType, ObjectPropType


def get_object_center(
    obj_id: Union[int, float, str], labeled_image: np.ndarray
) -> int:
    """Returns index of center pixel of the given object id from labeled
    image. The center is calculated as the median pixel of the object extent;
    it is not a true centroid."""
    obj_index = np.argwhere(labeled_image == obj_id)
    center = np.median(obj_index, axis=0).astype("i")
    return center


def get_obj_extent(
    labeled_image: np.ndarray, obj_label: float
) -> dict[str, np.ndarray]:
    """Takes in labeled image and finds the radius, area, and center of the
    given object."""
    obj_index = np.argwhere(labeled_image == obj_label)

    xlength = np.max(obj_index[:, 0]) - np.min(obj_index[:, 0]) + 1
    ylength = np.max(obj_index[:, 1]) - np.min(obj_index[:, 1]) + 1
    obj_radius = np.max((xlength, ylength)) / 2
    obj_center = np.round(np.median(obj_index, axis=0), 0)
    obj_area = len(obj_index[:, 0])

    obj_extent = {
        "obj_center": obj_center,
        "obj_radius": obj_radius,
        "obj_area": obj_area,
        "obj_index": obj_index,
    }
    return obj_extent


def init_current_objects(
    first_frame: np.ndarray,
    second_frame: np.ndarray,
    pairs: np.ndarray,
    counter: Counter,
) -> tuple[dict[str, np.ndarray], Counter]:
    """Returns a dictionary for objects with unique ids and their
    corresponding ids in frame1 and frame1. This function is called when
    echoes are detected after a period of no echoes."""
    nobj = np.max(first_frame)

    id1 = np.arange(nobj) + 1
    uid = counter.next_uid(count=nobj)
    id2 = pairs
    obs_num = np.zeros(nobj, dtype="i")
    origin = np.array(["-1"] * nobj)

    current_objects = {
        "id1": id1,
        "uid": uid,
        "id2": id2,
        "obs_num": obs_num,
        "origin": origin,
    }
    current_objects = attach_last_heads(
        first_frame, second_frame, current_objects
    )
    return current_objects, counter


def update_current_objects(
    frame1: np.ndarray,
    frame2: np.ndarray,
    pairs: np.ndarray,
    old_objects: dict[str, np.ndarray],
    counter: Counter,
) -> tuple[dict[str, np.ndarray], Counter]:
    """Removes dead objects, updates living objects, and assigns new uids to
    new-born objects."""
    nobj = np.max(frame1)
    id1 = np.arange(nobj) + 1
    uid = np.array([], dtype="str")
    obs_num = np.array([], dtype="i")
    origin = np.array([], dtype="str")

    for obj in np.arange(nobj) + 1:
        if obj in old_objects["id2"]:
            obj_index = old_objects["id2"] == obj
            uid = np.append(uid, old_objects["uid"][obj_index])
            obs_num = np.append(obs_num, old_objects["obs_num"][obj_index] + 1)
            origin = np.append(origin, old_objects["origin"][obj_index])
        else:
            origin = np.append(origin, -1)
            uid = np.append(uid, counter.next_uid())
            obs_num = np.append(obs_num, 0)

    id2 = pairs
    current_objects = {
        "id1": id1,
        "uid": uid,
        "id2": id2,
        "obs_num": obs_num,
        "origin": origin,
    }
    return attach_last_heads(frame1, frame2, current_objects), counter


def attach_last_heads(
    frame1: np.ndarray, frame2: np.ndarray, current_objects: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Attaches last heading information to current_objects dictionary."""
    nobj = len(current_objects["uid"])
    heads = np.ma.empty((nobj, 2))
    for obj in range(nobj):
        if (current_objects["id1"][obj] > 0) and (
            current_objects["id2"][obj] > 0
        ):
            center1 = get_object_center(current_objects["id1"][obj], frame1)
            center2 = get_object_center(current_objects["id2"][obj], frame2)
            heads[obj, :] = center2 - center1
        else:
            heads[obj, :] = np.ma.array([-999, -999], mask=[True, True])

    current_objects["last_heads"] = heads
    return current_objects


def check_isolation(
    raw: np.ndarray,
    filtered: np.ndarray,
    grid_size: np.ndarray,
    params: ConfigType,
) -> np.ndarray:
    """Returns list of booleans indicating object isolation. Isolated objects
    are not connected to any other objects by pixels greater than ISO_THRESH,
    and have at most one peak."""
    nobj = np.max(filtered)
    min_size = params["MIN_SIZE"] / np.prod(grid_size[1:] / 1000)
    iso_filtered = get_filtered_frame(raw, min_size, params["ISO_THRESH"])
    nobj_iso = np.max(iso_filtered)
    iso = np.empty(nobj, dtype="bool")

    for iso_id in np.arange(nobj_iso) + 1:
        obj_ind = np.where(iso_filtered == iso_id)
        objects = np.unique(filtered[obj_ind])
        objects = objects[objects != 0]
        if len(objects) == 1 and single_max(obj_ind, raw, params):
            iso[objects - 1] = True
        else:
            iso[objects - 1] = False
    return iso


def single_max(
    obj_ind: tuple[np.ndarray, ...], raw: np.ndarray, params: ConfigType
) -> bool:
    """Returns True if object has at most one peak."""
    max_proj = np.max(raw, axis=0)
    smooth = ndimage.gaussian_filter(max_proj, params["ISO_SMOOTH"])
    padded = np.pad(smooth, 1, mode="constant")
    obj_ind = tuple([axis + 1 for axis in obj_ind])  # adjust for padding
    maxima = 0
    for pixel in range(len(obj_ind[0])):
        ind_0 = obj_ind[0][pixel]
        ind_1 = obj_ind[1][pixel]
        neighborhood = padded[
            (ind_0 - 1) : (ind_0 + 2), (ind_1 - 1) : (ind_1 + 2)
        ]
        max_ind = np.unravel_index(neighborhood.argmax(), neighborhood.shape)
        if max_ind == (1, 1):
            maxima += 1
            if maxima > 1:
                return False
    return True


def get_object_prop(
    image1: np.ndarray,
    grid1: GridType,
    record: Record,
    params: ConfigType,
) -> ObjectPropType:
    """Returns dictionary of object properties for all objects found in
    image1."""
    id1: list[int] = []
    center: list[float] = []
    grid_x: list[float] = []
    grid_y: list[float] = []
    area: list[float] = []
    longitude: list[float] = []
    latitude: list[float] = []
    field_max: list[float] = []
    field_mean: list[float] = []
    nobj = np.max(image1)

    unit_area = 1  # (unit_dim[1]*unit_dim[2])/(1000**2)

    raw3D = grid1.data
    get_items = []
    for obj in np.arange(nobj) + 1:
        try:

            obj_index = np.argwhere(image1 == obj)
            this_centroid = np.round(np.mean(obj_index, axis=0), 3)
            rounded = np.round(this_centroid).astype("i")
            lon, lat = grid1.lon.values, grid1.lat.values
            if len(lon.shape) == 2:
                lon = lon[rounded[0], rounded[1]]
            else:
                lon = lon[rounded[1]]
            if len(lat.shape) == 2:
                lat = lat[rounded[0], rounded[1]]
            else:
                lat = lat[rounded[0]]
            longitude.append(np.round(lon, 4))
            latitude.append(np.round(lat, 4))

            id1.append(obj)

            # 2D frame stats
            center.append(np.median(obj_index, axis=0))
            grid_x.append(this_centroid[1])
            grid_y.append(this_centroid[0])
            area.append(obj_index.shape[0] * unit_area)

            # raw 3D grid stats
            obj_slices = [raw3D[:, ind[0], ind[1]] for ind in obj_index]
            field_max.append(float(np.nanmax(obj_slices)))
            field_mean.append(float(np.nanmean(obj_slices)))
            get_items.append(obj - 1)
        except IndexError:
            pass
    # cell isolation
    isolation = check_isolation(raw3D, image1, record.grid_size, params)
    objprop: ObjectPropType = {
        "id1": id1,
        "center": center,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "area": area,
        "field_max": field_max,
        "field_mean": field_mean,
        "lon": longitude,
        "lat": latitude,
        "isolated": isolation,
        "ok_items": get_items,
    }
    return objprop


def write_tracks(
    old_tracks: pd.DataFrame,
    record: Record,
    current_objects: dict[str, np.ndarray],
    obj_props: ObjectPropType,
) -> pd.DataFrame:
    """Writes all cell information to tracks dataframe."""

    nobj = len(obj_props["id1"])
    scan_num = [record.scan] * nobj
    gi_obj = obj_props["ok_items"]
    uid = current_objects["uid"][gi_obj]
    new_tracks = pd.DataFrame(
        {
            "scan": scan_num,
            "uid": uid,
            "time": record.time,
            "grid_x": obj_props["grid_x"],
            "grid_y": obj_props["grid_y"],
            "lon": obj_props["lon"],
            "lat": obj_props["lat"],
            "area": obj_props["area"],
            "max": obj_props["field_max"],
            "mean": obj_props["field_mean"],
            "isolated": obj_props["isolated"][gi_obj],
        }
    )
    new_tracks.set_index(["scan", "uid"], inplace=True)
    tracks = pd.concat([old_tracks, new_tracks])
    return tracks
