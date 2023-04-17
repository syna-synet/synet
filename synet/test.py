#!/usr/bin/env python
from argparse import ArgumentParser
def parse_opt():
    """dummy parse_opt entry func"""
    return ArgumentParser().parse_args()

from numpy import absolute
def test_arr(out1, out2):
    """compare two arrays.  Return the max difference."""
    assert all(s1 == s2 for s1, s2 in zip(out1.shape, out2.shape))
    return absolute(out1 - out2).max()


from .base import askeras
t_actv_to_k = lambda actv: actv.detach().numpy().transpose(0, 2, 3, 1)
def test_layer(layer, torch_inp):
    """Given synet layer, test on some torch input activations and
return max error between two output activations

    """
    tout = layer(torch_inp)
    with askeras(train=True):
        kout = layer(t_actv_to_k(torch_inp))
    if isinstance(tout, dict):
        assert len(tout) == len(kout)
        return max(test_arr(t_actv_to_k(tout[key]), kout[key])
                   for key in tout)
    elif isinstance(tout, list):
        assert len(tout) == len(kout)
        return max(test_arr(t_actv_to_k(t), k)
                   for t, k in zip(tout, kout))
    return test_arr(t_actv_to_k(tout), kout)


from torch import rand
from torch.nn.init import uniform_
def test_sizes(layer, batch_size, in_channels, shapes):
    """Run test_layer on a set of random input shapes.  Prints the max
difference between all configurations.

    """
    for param in layer.parameters():
        uniform_(param, -1)
    max_diff = max(test_layer(layer,rand(batch_size,in_channels,*shape)*2-1)
                   for shape in shapes)
    print("max_diff:", max_diff)
    return max_diff


from .base import Conv2d, ReLU, BatchNorm
def run():
    """Run all test cases.  Print errors.  Throw error if max error
encountered is greater than 1e-5

    """
    print("running tests")
    batch_size = 2
    in_channels = 5
    out_channels = 7
    expansion_factor = 11
    shapes = [(i, i) for i in range(5,10)]
    max_diff = -1

    print("testing Conv2d")
    for bias in True, False:
        print("use bias:", bias)
        for kernel, stride in ((1, 1), (3, 1), (3, 2)):
            print("kernel, stride:", kernel, stride)
            tconv = Conv2d(in_channels, out_channels, kernel, stride, bias)
            max_diff = max(test_sizes(tconv, batch_size, in_channels, shapes),
                           max_diff)

    print("testing ReLU")
    relu = ReLU(.6)
    max_diff = max(test_sizes(relu, batch_size, in_channels, shapes),
                   max_diff)

    print("testing BatchNorm")
    batchnorm = BatchNorm(in_channels)
    max_diff = max(test_sizes(batchnorm, batch_size, in_channels, shapes),
                   max_diff)


    print("OVERALL MAXIMUM DIFFERENCE:", max_diff)
    tolerance = 1e-5
    if max_diff > tolerance:
        print(f"maximum difference greater than tolerance ({tolerance}).")
        print("Tests failed")
        exit(1)
    print(f"maximum difference less than tolerance ({tolerance}).")
    print("Tests passed.")
