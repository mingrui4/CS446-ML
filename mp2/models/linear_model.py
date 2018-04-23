"""Linear model base class."""

import abc
import numpy as np
import six
import random


@six.add_metaclass(abc.ABCMeta)
class LinearModel(object):
    """Abstract class for linear models."""

    def __init__(self, ndims, w_init='zeros', w_decay_factor=0.001):
        """Initialize a linear model.

        This function prepares an uninitialized linear model.
        It will initialize the weight vector, self.w, based on the method
        specified in w_init.

        We assume that the last index of w is the bias term, self.w = [w,b]

        self.w(numpy.ndarray): array of dimension (n_dims+1,1)

        w_init needs to support:
          'zeros': initialize self.w with all zeros.
          'ones': initialze self.w with all ones.
          'uniform': initialize self.w with uniform random number between [0,1)

        Args:
            ndims(int): feature dimension
            w_init(str): types of initialization.
            w_decay_factor(float): Weight decay factor for regularization.
        """
        self.ndims = ndims
        self.w_init = w_init
        self.w_decay_factor = w_decay_factor

        if self.w_init == 'zeros':
            init = 0
        elif self.w_init == 'ones':
            init = 1
        elif self. w_init == 'uniforms':
            init = random.uniform(0,1)

        self.w = np.array([[init for a in range(1)] for b in range(self.ndims+1)])

        pass

    def forward(self, x):
        """Performs forward operation for linear models.

        Performs the forward operation. Appends 1 to x then compute
        f=w^Tx, and return f.

        Args:
            x(numpy.ndarray): Dimension of (N, ndims), N is the number
              of examples.

        Returns:
            (numpy.ndarray): Dimension of (N,1)
        """
        x1 = np.ones(x.shape[0])
        x0 = np.c_[x, x1]
        self.x = x0
        f = self.x.dot(self.w)
        pass
        return f

    @abc.abstractmethod
    def backward(self, f, y):
        """Do not need to be implemented here."""
        pass

    @abc.abstractmethod
    def total_loss(self, f, y):
        """Do not need to be implemented here."""
        pass

    def predict(self, f):
        """Do not need to be implemented here."""
        pass
