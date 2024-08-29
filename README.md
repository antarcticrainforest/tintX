# TintX (Tint is not TITAN) tracking algorithm for any kind of input data

[![Documentation Status](https://readthedocs.org/projects/tintx/badge/?version=latest)](https://tintx.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD-purple.svg)](LICENSE)
[![tests](https://github.com/antarcticrainforest/tintX/actions/workflows/tests.yml/badge.svg)](https://github.com/antarcticrainforest/tintX/actions)
[![codecov](https://codecov.io/gh/antarcticrainforest/tintX/branch/main/graph/badge.svg)](https://codecov.io/gh/antarcticrainforest/tintX)
[![PyPI version](https://badge.fury.io/py/tintx.svg)](https://badge.fury.io/py/tintx)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/tintx/badges/version.svg)](https://anaconda.org/conda-forge/tintx)
[![Latest](https://anaconda.org/conda-forge/tintx/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/tintx)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/antarcticrainforest/tintX/main?labpath=Readme.ipynb)

TintX is an adaptation of the tint [tracking algorithm](https://github.com/openradar/TINT).
Tint and tintX are easy-to-use storm cell tracking packages. While Tint is meant
to be applied to radar data using the
[py-ART toolkit](http://arm-doe.github.io/pyart/) tintX can
be applied with any data - for example output from numerical weather prediction
models. The original tracking algorithm has been developed by a team of
researchers at Monash University [Raut et al. 2020](http://dx.doi.org/10.1175/JAMC-D-20-0119.1).

## Installation
The `tintX` package can be installed using the `conda-froge` conda channel:

```console
conda install -c conda-forge tintx
```

Alternatively the package can be installed with pip:
```console
python -m pip install tintx
```


## Usage
Documentation can be found on the
[official documentation page](https://tintx.readthedocs.io/en/latest/) of this
library.

If you just want to try the usage and play with tracking data you can follow
[this link](https://mybinder.org/v2/gh/antarcticrainforest/tintX/main?labpath=Readme.ipynb)
to launch and familiarise yourself with the tracking by executing one of the
example notebooks.


## Contributing
Any contributions to improve this software in any way are welcome. Below is a
check list that makes sure your contributions can be added as fast as
possible to tintX:

- [ ] Create a fork of [this repository](https://github.com/antarcticrainforest/tintX)
     and clone this fork (not the original code)
- [ ] Install the code in development mode: `python3 -m pip install -e .[dev]`
- [ ] Create a new branch in the forked repository `git checkout -b my-new-branch`
- [ ] Add your changes
- [ ] Make sure all tests are sill running. To do so run the following commands
    - tox
- [ ] Create a new pull request to the `main` branch of the
     [original repository](https://github.com/antarcticrainforest/tintX).
### Adding new Jupyter examples
You can add more examples to the
[docs/source documentation folder](https://github.com/antarcticrainforest/tintX/tree/main/docs/source).
Because notebooks are executed automatically by the unit tests GitHub workflow,
you should make sure that any additional dependencies imported in the notebook
are added to the `docs` section in the
[setup.py](https://github.com/antarcticrainforest/tintX/blob/main/setup.py).
All notebooks should be run with a kernel called `tintx` to install a new
kernel named `tintx` run the following command in the root directory
of the cloned repository:
```console
python -m ipykernel install --name tintx --display-name "tintX kernel"\
    --env DATA_FIELS $PWD/docs/source/_static/data --user
```
Make also sure to add additional link(s) to the
[notebook readme file](https://github.com/antarcticrainforest/tintX/blob/main/.Readme.ipynb).


## Acknowledgements
This work is the adaptation of tracking code in R created by Bhupendra Raut
who was working at Monash University, Australia in the Australian Research
Council's Centre of Excellence for Climate System Science led by
Christian Jakob. This work was supported by the Department of
Energy, Atmospheric Systems Research (ASR) under Grant DE-SC0014063,
“The vertical structure of convective mass-flux derived from modern radar
systems - Data analysis in support of cumulus parametrization”

The development of this software was funded by the Australian Research
Council's Centre of Excellence for Climate Extremes under the funding
number CE170100023.
