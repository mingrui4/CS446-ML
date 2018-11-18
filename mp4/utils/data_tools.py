"""Implements feature extraction and other data processing helpers.
(This file will not be graded).
"""

import numpy as np
import skimage
from skimage import color


def preprocess_data(data, process_method='default'):
    """Preprocesses dataset.

    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1] by dividing by 255.
          2. Remove dataset mean. Average the images across the batch dimension.
             This will result in a mean image of dimension (8,8,3).
          3. Flatten images, data['image'] is converted to dimension (N, 8*8*3)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Convert from rgb to gray then back to rgb. Use skimage
          3. Take the absolute value of the difference with the original image.
          4. Remove dataset mean. Average the absolute value differences across
             the batch dimension. This will result in a mean of dimension (8,8,3).
          5. Flatten images, data['image'] is converted to dimension (N, 8*8*3)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'raw':
        data['image']=data['image']/255
        remove_data_mean(data)
        length = data['image'].shape[0]
        x = np.zeros((length, 8 * 8 * 3))
        for i in range(length):
            x[i] = data['image'][i].flatten()
        data['image'] = x
        pass

    elif process_method == 'default':
        data['image'] = data['image'] / 255
        gray = color.rgb2gray(data['image'])
        colorful = color.gray2rgb(gray)
        data['image'] = np.abs(data['image']-colorful)
        remove_data_mean(data)
        length = data['image'].shape[0]
        x = np.zeros((length, 8 * 8 * 3))
        for i in range(length):
            x[i] = data['image'][i].flatten()
        data['image'] = x


    elif process_method == 'custom':
        data['image'] = data['image'] / 255
        gray = color.rgb2gray(data['image'])
        colorful = color.gray2rgb(gray)
        data['image'] = np.abs(data['image']-colorful)
        remove_data_mean(data)
        length = data['image'].shape[0]
        x = np.zeros((length, 8 * 8 * 3))
        for i in range(length):
            x[i] = data['image'][i].flatten()
        data['image'] = x
        # Design your own feature!
        pass
    return data


def compute_image_mean(data):
    """ Computes mean image.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        image_mean(numpy.ndarray): Average across the example dimension.
    """
    mean_add = np.zeros((8, 8, 3))
    length = data['image'].shape[0]
    for i in range(length):
        mean_add = mean_add + data['image'][i]
    image_mean = mean_add/length
    return image_mean


def remove_data_mean(data):
    """Removes data mean.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    length = data['image'].shape[0]
    image_mean = compute_image_mean(data)
    for i in range(length):
        data['image'][i] = data['image'][i] - image_mean
    pass
    return data
