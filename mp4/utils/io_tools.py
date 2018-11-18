"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    file = open(data_txt_file, 'r')
    T = []
    A = []
    B = []
    train = file.readline()
    train_0 = train.strip('\n')
    train_1 = train_0.split(",")
    photo_id = str(train_1[0])
    sample = os.path.join(image_data_path, photo_id + '.jpg')
    picture = io.imread(sample)
    A.append(picture)
    T.append(train_1[1])
    while train:
        train = file.readline()
        train_0 = train.strip('\n')
        train_1 = train_0.split(",")
        try:
            photo_id = str(train_1[0])
            sample = os.path.join(image_data_path, photo_id + '.jpg')
            picture = io.imread(sample)
            A.append(picture)
            T.append(train_1[1])
        except:
            pass
    file.close()

    data['image'] = np.array(A)
    data['label'] = np.array(T).astype(float)

    return data
