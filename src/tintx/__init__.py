#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TintX main tracking utility.

TintX offers a simple ``RunDirectory`` class that uses the ``xarray`` library
to read datasets such as model output data or observational datasets.

The cell tracking as well as visualisation of cell tracks is realised through
instances of the ``RunDirectory`` class.
"""

from .reader import RunDirectory
from tintx import config

__all__ = ["RunDirectory"]

__version__ = "2022.8.0"
