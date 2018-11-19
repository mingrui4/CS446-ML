"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1], 
                                                     [1, x2], 
                                                     [1, x3],
                                                     .......] 
                                where xi is the 16-dimensional feature of each sample
            
        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...] 
                             where yi is +1/-1, the label of each sample 
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    file = open(path_to_dataset_folder + '/' + index_filename, 'r')
    T = []
    A = []
    data = file.readline()
    line = data.split()
    sample = open(path_to_dataset_folder +'/' + line[1].strip('\n'), 'r')
    x = sample.read().split()
    A.append(x)
    T.append(line[0])
    sample.close()
    while data:
        data = file.readline()
        line = data.split()
        try:
            sample = open(path_to_dataset_folder + '/' + line[1].strip('\n'), 'r')
            x = sample.read().split()
            A.append(x)
            T.append(line[0])
            sample.close()
        except:
            pass
    A0 = np.array(A).astype(float)
    A1 = np.ones(A0.shape[0])
    A = np.c_[A1, A0]
    T = np.array(T).astype(float)
    file.close()
    return A,T