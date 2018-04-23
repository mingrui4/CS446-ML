"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models import linear_regression
from models import linear_model
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    x_data = processed_dataset[0]
    y_data = processed_dataset[1]

    # _samples = x.shape[0]
    # _x0 = np.ones(_samples)
    # x_data = np.c_[x, _x0]
    index = list(range(0,1000))

    for k in range(num_steps):
        if shuffle == True:
            np.random.shuffle(index)
            for i in range(int(1000/16)):
                num = index[i*16:(i+1)*16]
                x_batch = x_data[num]
                y_batch = y_data[num]
                for j in range(batch_size):
                    update_step(x_batch, y_batch, model, learning_rate)


    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(x_batch)
    model.w = model.w -learning_rate * model.backward(f,y_batch)


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """

    x = processed_dataset[0]
    y = processed_dataset[1]

    x1 = np.ones((len(x),1))
    x0 = np.c_[x,x1]

    model.w = np.linalg.inv(x0.T .dot(x0)+ model.w_decay_factor * np.eye(x0.shape[1])) .dot(x0.T) .dot(y)
    pass


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    x = processed_dataset[0]
    y = processed_dataset[1]
    x1 = np.ones((len(x),1))
    x0 = np.c_[x,x1]
    f = np.dot(x0,model.w)
    loss = model.total_loss(f,y)

    return loss
