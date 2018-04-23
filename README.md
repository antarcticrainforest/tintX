# TintV2 (Tint is not TITAN tracking algorithm version 2)

This source code is python implementation of a tracking algorithm that has been 
developed by Bhupentra Raut at Monash University. The original version 
fo tint is designed to only work with pyart radar-data object. The aim of TintV2 
is therefore the application of the algorithm to arbitrary binary data. 

## Pre-requesites
TintV2 needs the following packages to be installed:
* [pandas](http://pandas.pydata.org)
* [numpy](http://www.numpy.org)
* [netCDF4](https://pypi.org/project/netCDF4/)
* [scipy](https://www.scipy.org)
* [matplotlib-basemap](https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap) (for visualisation)
## Installation
To install the packages run :
```bash
$: python setup.py install
```
if you don't have root access add the ```--user``` flag for a local installation
## Usage
For an example see the [ipython Notbook](https://github.com/antarcticrainforest/tintV2/blob/master/examples/RainTracking.ipynb)
in the example section.
