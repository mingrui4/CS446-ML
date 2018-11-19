"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *
import tensorflow as tf

""" Hyperparameter for Training """
learn_rate = 0.001
max_iters = 300

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')

    # Initialize model.

    # Train model via gradient descent.

    # Save trained model to 'trained_weights.np'

    # Load trained model from 'trained_weights.np'

    # Try all other methods: forward, backward, classify, compute accuracy
    A,T = read_dataset('../data/trainset','indexing.txt')
    ndims = A.shape[1]-1
    model = LogisticModel(ndims, W_init='zeros')
    model.fit(T, A, learn_rate, max_iters)

    model.save_model('trained_weights.np')
    model.load_model('trained_weights.np')

    Y = model.classify(A)

    accuracy = np.mean(Y == T)*100

    print(accuracy)




