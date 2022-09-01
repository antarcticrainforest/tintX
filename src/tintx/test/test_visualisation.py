"""Test data vis."""

from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
import xarray as xr


def test_animation(data_with_a_blob: xr.Dataset) -> None:
    """Simple test for creating animations."""

    from tintx import RunDirectory

    rd = RunDirectory(data_with_a_blob, "precip", x_coord="x", y_coord="y")
    _ = rd.get_tracks(field_thresh=0)
    animation = rd.animate(tracers=True, plot_style={"title": "test"})
    with TemporaryDirectory() as temp_dir:
        animation.save(Path(temp_dir) / "test.gif")
        assert (Path(temp_dir) / "test.gif").is_file()


def test_plot_tracks(data_with_a_blob: xr.Dataset) -> None:
    """Simple test for plotting tracks."""
    from tintx import RunDirectory

    rd = RunDirectory(data_with_a_blob, "precip", x_coord="x", y_coord="y")
    _ = rd.get_tracks(field_thresh=1000)
    with pytest.raises(TypeError):
        _ = rd.plot_trajectories(ax=1)
    with pytest.raises(ValueError):
        _ = rd.plot_trajectories()


def test_plot_real_data(real_data_files: Path) -> None:
    """Run the tracking with real data and plot it."""
    from tintx import RunDirectory
    from cartopy.mpl.geoaxes import GeoAxesSubplot

    files = list(real_data_files.rglob("*CPOL*.nc"))
    first = "2006-11-16 03:00"
    last = "2006-11-16 11:00"
    rd = RunDirectory.from_files(
        files,
        "radar_estimated_rain_rate",
        x_coord="longitude",
        y_coord="latitude",
        start=first,
        end=last,
    )
    _ = rd.get_tracks(field_thresh=0.001, min_size=0)
    ax = rd.plot_trajectories(label=True)
    assert isinstance(ax, GeoAxesSubplot)


def test_plot_kwargs() -> None:
    """Test the keyword correction of plotting."""
    from tintx.visualization import _normalize_kwargs

    kwargs = _normalize_kwargs(
        {
            "c": "yellow",
        }
    )
    assert "color" in kwargs
    assert kwargs["color"] == "yellow"

    kwargs = _normalize_kwargs({"ms": 20}, kind="line2d")
    assert "markersize" in kwargs
    assert kwargs["markersize"] == 20
