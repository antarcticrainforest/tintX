"""pytest definitions to run the unittests."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import pytest
import numpy as np
import xarray as xr


def get_blobs(blobs: int, size: int) -> np.ndarray:
    """Create an array with blobs."""
    y_coords, x_coords = np.meshgrid(np.arange(100), np.arange(100))
    x_blob = np.arange(0, 100, 100 / max(blobs**2, 2))[1:]
    data_array = np.zeros((100, 100))
    centres: list[tuple[int, int]] = []
    for i in x_blob:
        centres.append((i, i))
    for centre in centres:
        is_smaller = (x_coords - centre[1]) ** 2 + (y_coords - centre[0]) ** 2 < (
            size / 2
        ) ** 2
        data_array[is_smaller] = 1
    return data_array * 10


def create_data(variable_name: str, blobs: int, size: int) -> xr.Dataset:
    """Create a netcdf dataset."""
    coords: dict[str, np.ndarray] = {}
    coords["x"] = np.linspace(-10, -5, 100)
    coords["y"] = np.linspace(120, 125, 100)
    lat, lon = np.meshgrid(coords["y"], coords["x"])
    Lon = xr.DataArray(lon, name="Lg", coords=coords, dims=("y", "x"))
    Lat = xr.DataArray(lat, name="Lt", coords=coords, dims=("y", "x"))
    coords["time"] = np.array(
        [np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-01T00:10")]
    )
    dims = (2, 100, 100)
    data_array = np.empty(dims)
    for time in range(dims[0]):
        data_array[time] = get_blobs(blobs, size)
    dset = xr.DataArray(
        data_array,
        dims=("time", "y", "x"),
        coords=coords,
        name=variable_name,
    )
    data_array = np.zeros(dims)
    return xr.Dataset({variable_name: dset, "Lt": Lat, "Lg": Lon}).set_coords(
        list(coords.keys())
    )


@pytest.fixture(scope="session")
def save_dir() -> Generator[Path, None, None]:
    """Crate a temporary directory."""
    with TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture(scope="session")
def data_with_a_blob() -> Generator[xr.Dataset, None, None]:
    """Define a simple dataset with a blob in the middle."""
    data = create_data("precip", 1, 20)
    yield data


@pytest.fixture(scope="session")
def real_data_files() -> Generator[Path, None, None]:
    """Return the path where real data resides."""
    this_dir = Path(__file__).absolute().parent
    data_dir = this_dir.parent.parent.parent
    yield data_dir / "docs" / "source" / "_static" / "data"


@pytest.fixture(scope="session")
def netcdf_files_with_blob(
    data_with_a_blob: xr.Dataset,
) -> Generator[Path, None, None]:
    """Save data with a blob to file."""

    with TemporaryDirectory() as td:
        out_file = Path(td) / "out_f.nc"
        data_with_a_blob.to_netcdf(out_file)
        yield out_file
