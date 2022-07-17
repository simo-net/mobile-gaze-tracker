import math

def convolution_output_shape(in_shape: (int, int),
                             kernel: (int, int), stride: (int, int) = (1, 1), padding: (int, int) = (0, 0)) -> (int, int):
    """
    Let in_shape be the size of the input and kernel be the size of the kernel.
    The size of the output is given by the following formula: out_shape = (in_shape - kernel - 2*padding) / stride + 1
    """
    out_shape = tuple([int((i - k + 2*p) / s + 1) for i, k, s, p in zip(in_shape, kernel, stride, padding)])
    return out_shape


def pooling_output_shape(in_shape: (int, int),
                         pool: (int, int), stride: (int, int) = (1, 1)) -> (int, int):
    """
    Let in_shape be the size of the input and kernel be the size of the kernel.
    The size of the output is given by the following formula: out_shape = (in_shape - kernel - 2*padding) / stride + 1
    """
    assert all([i>p for i, p in zip(in_shape, pool)])
    out_shape = tuple([math.floor((i-p)/s)+1 for i,p,s in zip(in_shape, pool, stride)])
    return out_shape
