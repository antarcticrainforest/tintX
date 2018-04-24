from setuptools import setup
import numpy.distutils.misc_util, os

meta = dict(\
        description="Module to apply tracking algorithm of rainfall data",
        url = 'https://github.com/antarcticrainforest/tintV2',
        author = 'Martin Bergemann',
        author_email = 'martin.bergemann@met.fu-berlin.de',
        license = 'GPL',
        version = '1.0')

setup(name='tint', packages=['tint'], **meta)
