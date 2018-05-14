"""Input and output helpers to load in data.
"""

import pickle
import numpy as np
from numpy import genfromtxt


def read_dataset(input_file_path):
    """Read input file in csv format from file.
    In this csv, each row is an example, stored in the following format.
    label, pixel1, pixel2, pixel3...

    Args:
        input_file_path(str): Path to the csv file.
    Returns:
        (1) label (np.ndarray): Array of dimension (N,) containing the label.
        (2) feature (np.ndarray): Array of dimension (N, ndims) containing the
        images.
    """
    # Imeplemntation here.
    data = genfromtxt(input_file_path, delimiter=',')

    fw = open('data/dataFile.txt', 'wb')
    # Pickle the list using the highest protocol available.
    pickle.dump(data, fw, -1)
    fw.close()

    # Use load to read data from dataFile
    fr = open('data/dataFile.txt', 'rb')
    dataList = pickle.load(fr)
    fr.close()

    length = dataList.shape[1]
    features = dataList[:,1:length]
    labels = dataList[:,0]
    return labels, features