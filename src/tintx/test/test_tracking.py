"""Test the highlevel tracking results."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def test_tracking_of_dataset(data_with_a_blob: xr.Dataset) -> None:
    """Test the tracking of fake data with one field of rainfall."""

    from tintx import RunDirectory

    run_dir = RunDirectory(data_with_a_blob, "precip", x_coord="Lg", y_coord="Lt")
    run_dir2 = RunDirectory(
        data_with_a_blob["precip"], "precip", x_coord="x", y_coord="x"
    )
    assert isinstance(run_dir2.data, xr.Dataset)
    assert len(run_dir.tracks) == 0
    tracks = run_dir.get_tracks(field_thresh=0.0)
    assert tracks == 1
    assert len(run_dir.tracks) == 2
    tracks = run_dir.get_tracks(field_thresh=1000)
    assert tracks == 0


def test_get_parameters(data_with_a_blob: xr.DataArray) -> None:
    """Test getting the tuning parameters."""

    from tintx import RunDirectory

    run_dir = RunDirectory(data_with_a_blob, "precip", x_coord="Lg", y_coord="Lt")
    _ = run_dir.get_tracks(field_thresh=0.0)
    assert run_dir.get_parameters(run_dir.tracks) == run_dir.get_parameters()
    with pytest.raises(ValueError):
        run_dir.get_parameters(pd.DataFrame({"foo": ["bar"]}))


def test_tracking_of_datafile(netcdf_files_with_blob: Path) -> None:
    """Test tracking from saved netcdf files."""

    from tintx import RunDirectory

    run_dir = RunDirectory.from_files(
        netcdf_files_with_blob, "precip", x_coord="x", y_coord="y"
    )
    assert len(run_dir.tracks) == 0
    tracks = run_dir.get_tracks(field_thresh=0.0, iso_thresh=0)
    assert tracks == 1
    _ = run_dir.get_tracks(field_thresh=1000, flush=False)
    assert len(run_dir.tracks) == tracks + 1


def test_save_dataset(save_dir: Path, netcdf_files_with_blob: Path) -> None:
    """Test saving the dataset."""

    from tintx import RunDirectory

    run_dir = RunDirectory.from_files(
        netcdf_files_with_blob, "precip", x_coord="x", y_coord="y"
    )
    _ = run_dir.get_tracks(field_thresh=0.0, iso_thresh=0)
    run_dir.save_tracks(save_dir / "test.h5")
    assert len(run_dir.tracks) == len(
        pd.read_hdf(save_dir / "test.h5", "tintx_tracks")
    )


def test_load_dataset(save_dir: Path, netcdf_files_with_blob: Path) -> None:
    """Test loading the data from dataframe."""

    from tintx import RunDirectory

    save_file = save_dir / "test.h5"
    run_dir2 = RunDirectory.from_files(
        netcdf_files_with_blob, "precip", x_coord="x", y_coord="y"
    )
    _ = run_dir2.get_tracks(field_thresh=0.0)
    run_dir2.save_tracks(save_file)
    run_dir1 = RunDirectory.from_dataframe(save_file)
    run_dir3 = RunDirectory.from_dataframe(save_file, run_dir2.data)
    run_dir3._files = ""
    run_dir3.save_tracks(save_file)
    assert len(run_dir1.tracks) == len(run_dir2.tracks) == len(run_dir3.tracks)


def test_load_empty(data_with_a_blob: xr.DataArray) -> None:
    """Test loading empty datasets."""
    from tintx import RunDirectory

    run_dir = RunDirectory(data_with_a_blob, "precip", x_coord="Lg", y_coord="Lt")
    _ = run_dir.get_tracks(field_thresh=0.0)
    with NamedTemporaryFile(suffix=".h5") as save_file:
        run_dir.save_tracks(save_file.name)
        with pytest.warns(UserWarning):
            run_dir2 = RunDirectory.from_dataframe(save_file.name)
    data1 = run_dir.data
    data2 = run_dir2.data
    for dim in data1.dims:
        assert dim in data2.dims
        if dim != "time":
            assert np.allclose(data1[dim].values, data2[dim].values)
    assert "Lg" in data2.coords
    assert "Lt" in data2.coords
    assert "precip" in data2.data_vars
    assert data2["precip"].shape == data1["precip"].shape
