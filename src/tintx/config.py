"""
Tracking tuning parameters
--------------------------

The following parameter can be set to tune the cell tracking algorithm

* field_thresh : units of 'field' attribute, default: 32
    The threshold used for object detection. Detected objects are connected
    pixels above this threshold.
* iso_thresh : units of 'field' attribute, default: 4
    Used in isolated cell classification. Isolated cells must not be connected
    to any other cell by contiguous pixels above this threshold.
* iso_smooth : pixels, default: 4
    Gaussian smoothing parameter in peak detection preprocessing. See
    single_max in tint.objects.
* min_size : square kilometers, default: 8
    The minimum size threshold in pixels for an object to be detected.
* search_margin : meters, default: 250
    The radius of the search box around the predicted object center.
* flow_margin : meters, default: 750
    The margin size around the object extent on which to perform phase
    correlation.
* max_disparity : float, default: 999
    Maximum allowable disparity value. Larger disparity values are sent to a
    large number.
* max_flow_mag : meters per second, default: 50
    Maximum allowable global shift magnitude. See get_global_shift in
    tint.phase_correlation.
* max_shift_disp : meters per second, default: 15
    Maximum magnitude of difference in meters per second for two shifts to be
    considered in agreement. See correct_shift in tint.matching.
* gs_alt : meters, default: 1500
    Altitude in meters at which to perform phase correlation for global shift
    calculation. See correct_shift in tint.matching.


Setting and getting the tuning parameters
-----------------------------------------
The parameters can the set using the :py:class:`tintx.config.set` class.
Getting the values of the currently set parameters can be done by the
:py:func:`tintx.config.get` method.
"""

from __future__ import annotations
from typing import cast, Any, Optional
from typing_extensions import Literal

config: dict[str, float] = dict(
    ISO_THRESH=4.0,
    FIELD_THRESH=32.0,
    ISO_SMOOTH=4.0,
    MIN_SIZE=8.0,
    SEARCH_MARGIN=250.0,
    FLOW_MARGIN=750.0,
    MAX_DISPARITY=999.0,
    MAX_FLOW_MAG=50.0,
    MAX_SHIFT_DISP=15.0,
    GS_ALT=1500.0,
)
global_config: dict[str, float] = config  # alias


def get(key: str, default: Optional[float] = None) -> Optional[float]:
    """
    Get elements from global tinitX config


    Examples
    --------
    .. execute_code::
        :hide_headers:

        from tintx import config
        print(config.get('min_size'))

    .. execute_code::
        :hide_headers:

        from tintx import config
        print(config.get('bar', default=123.))


    See Also
    --------
    tintx.config.set
    """

    return config.get(key.upper(), default)


class set:
    """Set tint configuration values within a context manager

    Parameters
    ----------
    **kwargs : float
        key-value pairs of the config values to set.

    Example
    -------
    Set ``field_thresh`` parameter in a context. After the ``with`` block
    the parameter will be set back to its original value.

    .. execute_code::
        :hide_headers:

        from tintx import config
        with config.set(field_thresh=5):
            print(config.get("field_thresh"))
        print(config.get("field_thresh"))

    To make the configuration changes persistent you can use ``set`` without
    the ``with`` block:

    .. execute_code::
        :hide_headers:

        import tintx
        tintx.config.set(field_thresh=5)


    See Also
    --------
    tintx.config.get
    """

    config: dict[str, float]
    _record: list[tuple[Literal["insert", "replace"], str, Optional[float]]]

    def __init__(
        self,
        **kwargs: float,
    ):
        self.config = config
        self._record = []

        for key, value in kwargs.items():
            self._assign(key.upper(), float(value), config)

    def __enter__(self) -> set:
        return self

    def __exit__(self, *args: Any) -> None:
        for op, key, value in reversed(self._record):
            if op == "replace":
                self.config[key] = cast(float, value)
            else:  # insert
                self.config.pop(key, None)

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Convenience method to retrieve elements from the ``tinitx`` configuration


        Examples
        --------
        >>> from tintx import config
        >>> config.get('min_size')
        8.0

        >>> config.get('bar', default=123.)
        123
        """
        return get(key, default)

    def _assign(
        self,
        key: str,
        value: float,
        cfg: dict[str, float],
        record: bool = True,
    ) -> None:
        """Assign value into a nested configuration dictionary

        Parameters
        ----------
        keys : Sequence[str]
            The nested path of keys to assign the value.
        value : object
        d : dict
            The part of the nested dictionary into which we want to assign the
            value
        path : tuple[str], optional
            The path history up to this point.
        record : bool, optional
            Whether this operation needs to be recorded to allow for rollback.
        """
        if record:
            if key in cfg:
                self._record.append(("replace", key, cfg[key]))
            else:
                self._record.append(("insert", key, None))
        cfg[key] = value
