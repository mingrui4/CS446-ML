import tensorflow as tf


def toy_fn_1(arg1, arg2):
    '''Given two tensors of arbitrary (but same) rank and size, build a computation
    graph for the following function, which should be computed element-wise:

    arg1^3 + 4*arg2^2 - 10*arg1

    Args:
        arg1(tf.Tensor): A tensor of arbitrary rank
        arg2(tf.Tensor): A tensor of the same rank as arg1
    Returns:
        (tf.Tensor): the result of the computation (same rank as inputs)
    '''
    # Input your code here
    ans_1 = tf.pow(arg1, 3) + 4 * tf.square(arg2) - 10 * arg1
    return ans_1


def toy_fn_2(arg1, arg2):
    '''Given a rank-two tensor and a rank-one tensor, build a computation graph
    that computes the following:

    first, it sums over the first dimension of the rank-two tensor
    (zero-indexed - i.e. sum over the rows). It then subtracts the maximum
    value of the rank-1 tensor from each element of the result.

    Args:
        arg1(tf.Tensor): A rank-2 tensor with dimensions (m, n)
        arg2(tf.Tensor): A rank-1 tensor with dimension p
    Returns:
        (tf.Tensor): the result of the computation, which is a rank-1 tensor
          with dimension m
    '''
    # Input your code here
    add_2 = tf.reduce_sum(arg1,reduction_indices=1)
    max = tf.reduce_max(arg2)
    ans_2 = add_2-max
    return ans_2


def toy_fn_3(arg1, arg2):
    '''
    Given two rank-one tensors of the same size, build a computation graph that
    builds a rank-one tensor by interleaving the two original tensors. For
    example, given the following inputs:

    arg1 = [1, 2]
    arg2 = [10, 20]

    The result should be [1, 10, 2, 20]

    Hint: this can be accomplished by first creating a rank-two tensor whose
    columns are the two original tensors and then reshaping it. Make sure the
    final tensor is rank-1!

    Args:
        arg1(tf.Tensor): A rank-1 tensor with dimension m
        arg2(tf.Tensor): A rank-1 tensor with dimension m
    Returns:
        (tf.Tensor): the result of the computation, which is a rank-1 tensor
          with dimension 2*m
    '''
    # Input your code here
    a = tf.expand_dims(arg1, 1)
    b = tf.expand_dims(arg2, 1)
    c = tf.concat([a, b], 1)
    ans_3 = tf.reshape(c, [1, -1])
    return ans_3
