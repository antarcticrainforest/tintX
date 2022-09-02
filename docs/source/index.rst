.. tintX documentation master file, created by
   sphinx-quickstart on Wed Aug 24 16:33:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tintX's documentation!
=================================

.. image:: https://readthedocs.org/projects/tintx/badge/?version=latest
    :target: https://tintx.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://codecov.io/gh/antarcticrainforest/tintX/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/antarcticrainforest/tintX
.. image:: https://github.com/antarcticrainforest/tintX/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/antarcticrainforest/tintX/actions
.. image:: https://badge.fury.io/py/tintx.svg
    :target: https://badge.fury.io/py/tintx

TintX is an adaptation of the tint `tracking algorithm <https://github.com/openradar/TINT>`_.
Tint and ``tintX`` are easy-to-use storm cell tracking package based on the
TITAN methodology by Dixon and Wiener. While Tint is meant to be applied to
radar data using the `py-ART toolkit <http://arm-doe.github.io/pyart/>`_ tintX can
be applied with any data - for example output from numerical weather prediction
models.

How does the tint algorithm work?
---------------------------------

The tracking algorithm is designed to track storm cells using `phase
correlation <https://en.wikipedia.org/wiki/Phase_correlation>`_
between two consecutive time steps and an application of the
`Hungarian Maximum Matching Algorithm
<https://en.wikipedia.org/wiki/Hungarian_algorithm>`_
to identify cells that are connected in time and space. The algorithm assigns
every identified storm cell a unique identifier (uid).
The Hungarian Matching Algorithm decides whether a new uid
(a new storm cell appears or splits from another system) or an existing uid
is assigned (a storm cell from the previous time step). This unique
identifiers allow for a connection of individual storm cells in time and space.

The `original tint package <https://github.com/openradar/TINT>`_, wich has
been developed for radar data only, is adopted to be able to track model and
radar based rainfall data alike.

Installation
------------

The ``tintX`` package can be installed using pip:

.. code-block:: console

   python3 -m pip install tintx

if you don't have root access add the ``--user`` flag for a local installation.

Citation
--------

The original version of the Tint tracking algorithm can be found under:

Raut, B. A., Jackson, R., Picel, M., Collis, S. M., Bergemann, M.,
& Jakob, C. (2021). An Adaptive Tracking Algorithm for Convection in Simulated
and Remote Sensing Data. Journal of Applied Meteorology and Climatology,
60(4), 513-526. [1]_

The TintX tracking package is introduces in:

Bergemann, M. Lane T. P., Wales, S., Narsey, S., Louf, V. (2022), High
Resolution Simulations of Tropical Island Thunderstorms: Does an Increase
in Resolution Improve the Representation of Extreme Rainfall?. [2]_

Acknowledgements
----------------

This work is the adaptation of tracking code in R created by Bhupendra Raut
who was working at Monash University, Australia in the Australian Research
Council's Centre of Excellence for Climate System Science led by
Christian Jakob. This work was supported by the Department of
Energy, Atmospheric Systems Research (ASR) under Grant DE-SC0014063,
“The vertical structure of convective mass-flux derived from modern radar
systems - Data analysis in support of cumulus parametrization”

The development of this software was funded by the Australian Research
Council's Centre of Excellence for Climate Extremes under the fundering
number CE170100023.

.. toctree::
   :maxdepth: 2
   :caption: Documentation content:

   api
   I_Tracking_data_from_files
   II_Tracking_already_loaded_datasets
   III_Using_the_command_line_interface

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [1] `doi: 10.1175/JAMC-D-20-0119.1 <https://doi.org/10.1175/JAMC-D-20-0119.1>`_
.. [2] `doi: 10.1002/qj.4360 <https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.4360>`_
