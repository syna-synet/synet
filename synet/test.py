#!/usr/bin/env python
from argparse import ArgumentParser
from math import ceil

from numpy import absolute
from torch import rand
from torch.nn.init import uniform_

from .base import askeras, Conv2d, ReLU, BatchNorm


def parse_opt():
    """dummy parse_opt entry func"""
    return ArgumentParser().parse_args()


def test_arr(out1, out2):
    """compare two arrays.  Return the max difference."""
    assert all(s1 == s2 for s1, s2 in zip(out1.shape, out2.shape)), \
        (out1.shape, out2.shape)
    return absolute(out1 - out2).max()


def t_actv_to_k_helper(actv):
    if len(actv.shape) == 4:
        tp = 0, 2, 3, 1
    elif len(actv.shape) == 3:
        tp = 0, 2, 1
    return actv.detach().numpy().transpose(*tp)


def t_actv_to_k(actv):
    return [t_actv_to_k_helper(a) for a in actv] if isinstance(actv, list) \
        else t_actv_to_k_helper(actv)


def test_layer(layer, torch_inp):
    """Given synet layer, test on some torch input activations and
return max error between two output activations

    """
    tout = layer(torch_inp[:])
    with askeras(train=True, imgsz=torch_inp[0].shape[-2:],
                 xywh=True):
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


def test_sizes(layer, batch_size, in_channels, shapes):
    """Run test_layer on a set of random input shapes.  Prints the max
difference between all configurations.

    """
    for param in layer.parameters():
        uniform_(param, -1)
    max_diff = max(test_layer(layer,
                              [rand(batch_size, in_channels, *s)*2-1
                               for s in shape]
                              if isinstance(shape[0], tuple)
                              else rand(batch_size, in_channels, *shape)*2-1)
                   for shape in shapes)
    print("max_diff:", max_diff)
    return max_diff


def run():
    """Run all test cases.  Print errors.  Throw error if max error
encountered is greater than 1e-5

    """
    print("running tests")
    batch_size = 2
    in_channels = 5
    out_channels = 7
    shapes = [(i, i) for i in range(5, 10)]
    max_diff = -1

    print("testing Conv2d")
    for bias in True, False:
        print("use bias:", bias)
        for kernel, stride in ((1, 1), (2, 1), (3, 1), (3, 2), (4, 1),
                               (4, 2), (4, 3)):
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

    from .ultralytics_patches import Detect
    print("testing Ultralytics Detect")
    for sm_split in None, 1, 2:
        print("sm_split:", sm_split)
        detect = Detect(nc=13, ch=(in_channels, in_channels), sm_split=sm_split)
        detect.eval()
        detect.export = True
        detect.format = 'tflite'
        detect.stride[0], detect.stride[1] = 1, 2
        # need to only test even sizes for sm_split = 2
        dshapes = [((4, 4), (2, 2)), ((8, 8), (4, 4)), ((12, 12), (6, 6))]
        max_diff = max(test_sizes(detect, batch_size, in_channels, dshapes),
                       max_diff)

    from .ultralytics_patches import Pose
    print("testing Ultralytics Pose, 2kpt")
    pose = Pose(nc=13, kpt_shape=(17, 2), ch=(in_channels, in_channels))
    pose.eval()
    pose.export = True
    pose.format = 'tflite'
    pose.stride[0], pose.stride[1] = 1, 2
    dshapes = [(s, (ceil(s[0]/2), ceil(s[1]/2))) for s in shapes]
    max_diff = max(test_sizes(pose, batch_size, in_channels, dshapes),
                   max_diff)

    print("testing Ultralytics Pose, 3kpt")
    pose = Pose(nc=13, kpt_shape=(17, 3), ch=(in_channels, in_channels))
    pose.eval()
    pose.export = True
    pose.format = 'tflite'
    pose.stride[0], pose.stride[1] = 1, 2
    dshapes = [(s, (ceil(s[0]/2), ceil(s[1]/2))) for s in shapes]
    max_diff = max(test_sizes(pose, batch_size, in_channels, dshapes),
                   max_diff)

    print("OVERALL MAXIMUM DIFFERENCE:", max_diff)
    tolerance = 2e-4
    if max_diff > tolerance:
        print(f"maximum difference greater than tolerance ({tolerance}).")
        print("Tests failed")
        exit(1)
    print(f"maximum difference less than tolerance ({tolerance}).")
    print("Tests passed.")
