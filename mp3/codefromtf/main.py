"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.001
max_iters = 300

def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')

    # Initialize model.

    # Build TensorFlow training graph
    
    # Train model via gradient descent.

    # Compute classification accuracy based on the return of the "fit" method
    A, T = read_dataset_tf('../data/trainset', 'indexing.txt')
    ndims = A.shape[1] - 1
    model = LogisticModel_TF(ndims, W_init='zeros')
    model.build_graph(learn_rate)
    Y = model.fit(T, A, max_iters)
    for i in range(len(Y)):
        if Y[i] <0.5:
            Y[i] = 0
        elif Y[i] >= 0.5:
            Y[i] = 1

    accuracy = np.mean(Y == T) * 100
    print(accuracy)


if __name__ == '__main__':
    tf.app.run()
