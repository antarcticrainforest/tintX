.. tintX documentation master file, created by
   sphinx-quickstart on Wed Aug 24 16:33:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tintX's documentation!
=================================

TintX is an adaptation of the tint `tracking algorithm <https://github.com/openradar/TINT>`_.
Tint and TintX are easy-to-use storm cell tracking package based on the
TITAN methodology by Dixon and Wiener. While Tint is meant to be applied to
radar data using the `py-ART toolkit <http://arm-doe.github.io/pyart/>`_ tintX can
be applied with any data - for example output from numerical weather prediction
models.

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
   :caption: Contents:

   api
   RainTracking

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [1] `doi: 10.1175/JAMC-D-20-0119.1 <https://doi.org/10.1175/JAMC-D-20-0119.1>`_
.. [2] `doi: 10.1002/qj.4360 <https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.4360>`_
