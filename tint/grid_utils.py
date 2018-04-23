"""
tint.grid_utils
===============

Tools for pulling data from reading data.


"""

import datetime

import numpy as np
import pandas as pd
from scipy import ndimage


def parse_grid_datetime(grid_obj):
    """ Obtains datetime object from the data dictionary. """
    #dt_string = grid_obj.time['units'].split(' ')[-1]
    #date = dt_string[:10]
    ##time = dt_string[11:19]
    #dt = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
    return grid_obj['time']


def get_grid_size(grid_obj):
    """ Calculates grid size per dimension given a grid object. """
    return np.array(grid_obj['data'].shape)


def get_radar_info(radar):
    info = {'radar_lon': radar[0],
            'radar_lat': radar[0]}
    return info


def get_grid_alt(grid_size, alt_meters=1500):
    """ Returns z-index closest to alt_meters. """
    return np.int(np.round(alt_meters/grid_size[0]))


def get_vert_projection(grid, thresh=40):
    """ Returns boolean vertical projection from grid. """
    return np.any(grid > thresh, axis=0)


def get_filtered_frame(grid, min_size, thresh):
    """ Returns a labeled frame from gridded radar data. Smaller objects
    are removed and the rest are labeled. """
    echo_height = get_vert_projection(grid, thresh)
    labeled_echo = ndimage.label(echo_height)[0]
    frame = clear_small_echoes(labeled_echo, min_size)
    return frame


def clear_small_echoes(label_image, min_size):
    """ Takes in binary image and clears objects less than min_size. """
    flat_image = pd.Series(label_image.flatten())
    flat_image = flat_image[flat_image > 0]
    size_table = flat_image.value_counts(sort=False)
    small_objects = size_table.keys()[size_table < min_size]

    for obj in small_objects:
        label_image[label_image == obj] = 0
    label_image = ndimage.label(label_image)
    return label_image[0]


def extract_grid_data(grid_obj, field, grid_size, params):
    """ Returns filtered grid frame and raw grid slice at global shift
    altitude. """
    try:
        masked = grid_obj['data'].filled(0)
    except AttributeError:
        masked = grid_obj['data']
    gs_alt = params['GS_ALT']
    raw = masked[0 , :, :]
    frame = get_filtered_frame(masked, params['MIN_SIZE'],
                               params['FIELD_THRESH'])
    return raw, frame
