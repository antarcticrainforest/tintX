# TintX (Tint is not TITAN) tracking algorithm for any kind of input data

TintX is an adaptation of the tint (tracking algorithm)[https://github.com/openradar/TINT].
Tint and TintX are easy-to-use storm cell tracking package based on the
TITAN methodology by Dixon and Wiener. While Tint is meant to be applied to
radar data using the [py-ART toolkit](http://arm-doe.github.io/pyart/) tintX can
be applied with any data - for example output from numerical weather prediction
models. The original tracking algorithm that has been developed by a team of
researchers at Monash University [Raut et al. 2020](http://dx.doi.org/10.1175/JAMC-D-20-0119.1).

## Installation
To install the packages run :
```bash
$: python -m pip install tintx
```
if you don't have root access add the ```--user``` flag for a local installation

## Usage
Documentation can be found on the official document page of this library.

## Acknowledgements
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
