#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TintX main tracking utility.

TintX offers a simple :class:``RunDirectory`` class that uses the ``xarray``
library to read datasets such as model output data or observational datasets.

The cell tracking as well as visualisation of cell tracks is realised through
instances of this :class:`RunDirectory` class.
"""

from tintx import config  # noqa: F401

from .reader import RunDirectory

__all__ = ["RunDirectory", "config"]

__version__ = "2025.05.1"
