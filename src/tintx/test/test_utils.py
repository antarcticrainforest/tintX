"""Tests for various utilities."""

from datetime import timedelta

import pandas as pd
import numpy as np
import xarray as xr


def test_time_interval() -> None:
    """Test for getting the time interval."""

    from tintx.helpers import get_interval

    dt_1 = timedelta(seconds=3600)
    time1 = np.datetime64("2020-01-01T00:00")
    time2 = np.datetime64("2020-01-01T01:00")
    dt_2 = time2 - time1
    assert get_interval(dt_1) == get_interval(dt_2) == 3600


def test_shift() -> None:
    """Test the phase corrleation function."""

    from tintx.phase_correlation import get_global_shift

    shift_array = np.ones((10, 10))
    assert get_global_shift(None, None) is None
    assert get_global_shift(None, shift_array) is None
    assert get_global_shift(shift_array, None) is None
    assert isinstance(get_global_shift(shift_array, shift_array), np.ndarray)


def test_meta_reader(data_with_a_blob: xr.Dataset) -> None:
    """Test the metdata reader."""
    from tintx.helpers import MetaData

    metdata = MetaData(
        data_with_a_blob,
        "precip",
        {"x_coord": "Lg", "y_coord": "Lt", "time_coord": "time"},
    )
    assert isinstance(metdata.to_dataframe("time"), pd.Series)
    assert isinstance(metdata.to_dataframe("Lg"), pd.DataFrame)
