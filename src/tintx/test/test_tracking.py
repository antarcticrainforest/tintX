"""Test the highlevel tracking results."""

from pathlib import Path

import pandas as pd
import xarray as xr


def test_tracking_of_dataset(data_with_a_blob: xr.Dataset) -> None:
    """Test the tracking of fake data with one field of rainfall."""

    from tintx import RunDirectory, config

    run_dir = RunDirectory("precip", data_with_a_blob, x_coord="Lg", y_coord="Lt")
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
