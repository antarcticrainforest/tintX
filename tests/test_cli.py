"""Test for the command line interface."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


def test_help() -> None:
    """Test the help."""

    from tintx.cli import tintx

    with pytest.raises(SystemExit):
        tintx()


def test_tracking_and_plotting(real_data_files: Path) -> None:
    """Test plotting and tracking."""

    from tintx.cli import tintx

    with TemporaryDirectory() as td:
        save_file = Path(td) / "test.h5"
        tintx(
            [
                "track",
                "radar_estimated_rain_rate",
                str(real_data_files / "CPOL_radar.nc"),
                "--x-coord=longitude",
                "--y-coord=latitude",
                "--field-thresh=0.1",
                "--start=2006-11-16T03:00:00",
                "--end=2006-11-16T11:00:00",
                f"--output={save_file}",
            ],
            standalone_mode=False,
        )
        assert save_file.is_file()
        tintx(
            [
                "plot",
                str(save_file),
                "--animate",
                f"-o={save_file.with_suffix('.mp4')}",
            ],
            standalone_mode=False,
        )
        tintx(
            [
                "plot",
                str(save_file),
                f"-o={save_file.with_suffix('.png')}",
            ],
            standalone_mode=False,
        )


def test_get_files(netcdf_files_with_blob: Path) -> None:
    """Test getting the files."""

    from tintx.cli import _get_file_names

    wild_card = (netcdf_files_with_blob.parent / "*.nc",)
    files1 = tuple(_get_file_names((netcdf_files_with_blob,)))
    files2 = tuple(_get_file_names((netcdf_files_with_blob.parent,)))
    files3 = tuple(_get_file_names(wild_card))
    assert files1 == files2 == files3
