#!/usr/bin/env python
from numpy import absolute
def test_arr(out1, out2):
    assert all(s1 == s2 for s1, s2 in zip(out1.shape, out2.shape))
    return absolute(out1 - out2).max()


from models import t_actv_to_k, askeras
def test_layer(layer, torch_inp):
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
def test_sizes(layer, batch_size, in_channels, shapes):
    layer.rand_init()
    max_diff = max(test_layer(layer,rand(batch_size,in_channels,*shape)*2-1)
                   for shape in shapes)
    print("max_diff:", max_diff)
    return max_diff


from models import (Conv2d, ReLU, BatchNorm, InvertedResidule,
                    Backbone, Head, PersonDetect)
def run_tests():
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

    for stride in 1, 2:
        print(f"testing InvRes (stride={stride})")
        invres = InvertedResidule(in_channels, expansion_factor, stride=stride)
        max_diff = max(test_sizes(invres, batch_size, in_channels, shapes),
                       max_diff)

    print("testing Backbone")
    backbone = Backbone()
    max_diff = max(test_sizes(backbone, batch_size, in_channels=1,
                              shapes=shapes),
                   max_diff)

    print("testing Head")
    head = Head(in_channels, out_channels)
    max_diff = max(test_sizes(head, batch_size, in_channels, shapes),
                   max_diff)

    print("testing PersonDetect")
    persondetect = PersonDetect()
    max_diff = max(test_sizes(persondetect, batch_size,
                              in_channels=1, shapes=shapes),
                   max_diff)

    print("OVERALL MAXIMUM DIFFERENCE:", max_diff)
    tolerance = 1e-2
    if max_diff > tolerance:
        print(f"maximum difference greater than tolerance ({tolerance}).")
        print("Tests failed")
        exit(1)
    print(f"maximum difference less than tolerance ({tolerance}).")
    print("Tests passed.")


if __name__ == "__main__":
    run_tests()
