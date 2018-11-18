"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    x_data = data['image']
    y_data = data['label']
    l = x_data.shape[0]
    index = list(range(0,l))
    for k in range(num_steps):
        if shuffle == True:
            np.random.shuffle(index)
            for i in range(int(l/16)):
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
    # Implementation here. (This function will not be graded.)
    f = model.forward(x_batch)
    model.w = model.w - learning_rate * model.backward(f, y_batch)




def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    x = data['image']
    y = data['label']
    n_samples= data['image'].shape[0]
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)

    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    pass
    # Set model.w

    x0 = np.ones((len(x),1))

    add0_x = np.append(x0,x,1)
    x = add0_x
    z0 = np.ravel(z)

    sv_index = z0 > 1e-5
    a = z[sv_index]
    sv= x[sv_index]
    sv_y =y[sv_index]

    for n in range(len(a)):
        model.w +=a[n] * sv_y[n] * sv[n]
    return  model



def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    # Gram matrix
    n_samples = data['image'].shape[0]
    x = data['image']
    y = data['label']
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.dot(x[i], x[j])
    P = np.outer(y, y) * K
    q = np.ones((n_samples,1)) * -1
    tmp1 = np.diag(np.ones(n_samples) * -1)
    tmp2 = np.identity(n_samples)
    G = np.vstack((tmp1, tmp2))
    tmp1 = np.zeros((n_samples,1))
    tmp2 = np.ones((n_samples,1)) * model.w_decay_factor
    h = np.r_[tmp1, tmp2]
    #     np.hstack((tmp1, tmp2)).astype(float)



    # Implementation here.
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    x = data['image']
    y = data['label']
    f = model.forward(x)
    loss = model.total_loss(f,y)
    y_predict = model.predict(f)
    acc = np.mean(y == y_predict)*100
    return loss, acc
