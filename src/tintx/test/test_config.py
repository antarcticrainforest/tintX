"""Tests for the tint configuration."""

import xarray as xr


def test_set_config(data_with_a_blob: xr.Dataset) -> None:
    """Test the set class."""

    from tintx import config, RunDirectory

    with config.set(field_thresh=0.0, foo=2) as cfg:
        run_dir = RunDirectory(
            data_with_a_blob, "precip", x_coord="Lg", y_coord="Lt"
        )

        assert run_dir.params["FIELD_THRESH"] == 0
        assert cfg.get("foo") == 2
    assert config.get("foo", 3) == 3
    assert run_dir.params["FIELD_THRESH"] != 0


def test_get_config() -> None:

    from tintx import config

    assert config.get("FIELD_THRESH") == config.get("field_thresh") == 32.0
    assert config.get("foo") is None
    assert config.get("foo", default=23) == 23
