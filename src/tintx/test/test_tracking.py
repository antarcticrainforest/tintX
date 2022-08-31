"""Test the highlevel tracking results."""

from pathlib import Path

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
    assert len(run_dir.tracks) == len(pd.read_hdf(save_dir / "test.h5", "tintx_tracks"))


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
    with pytest.raises(ValueError):
        RunDirectory.from_dataframe(save_file)
