#!/usr/bin/env python
from argparse import ArgumentParser
from copy import deepcopy
from math import ceil
from os import environ
from sys import argv

from numpy import absolute
from torch import rand, no_grad
from torch.nn.init import uniform_

from synet.base import askeras, Conv2d, ReLU, BatchNorm
from synet.__main__ import main as synet_main


def parse_opt():
    """dummy parse_opt entry func"""
    parser = ArgumentParser()
    parser.add_argument("mode", nargs="?", default='all')
    return parser.parse_args()


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
                 xywh=True, test=True):
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


def init(module):
    for param in module.parameters():
        uniform_(param, -1)


def test_sizes(layer, batch_size, in_channels, shapes):
    """Run test_layer on a set of random input shapes.  Prints the max
difference between all configurations.

    """
    init(layer)
    max_diff = max(test_layer(layer,
                              [rand(batch_size, in_channels, *s)*2-1
                               for s in shape]
                              if isinstance(shape[0], tuple)
                              else rand(batch_size, in_channels, *shape)*2-1)
                   for shape in shapes)
    print("max_diff:", max_diff)
    return max_diff


def run_tf():
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

    from .backends.ultralytics import Detect
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

    from .backends.ultralytics import Pose
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
        return 1
    print(f"maximum difference less than tolerance ({tolerance}).")
    print("Tests passed.")
    return 0


def test_nni_layer(float_model, quant_model, batch, in_channels, shapes):
    with no_grad():
        return max(test_arr(float_model(inp), quant_model(inp))
                   for inp in (rand(batch, in_channels, *shape)*2-1
                               for shape in shapes))


def run_nni():
    batch = 2
    in_channels = 5
    out_channels = 7
    shapes = [(i, i) for i in range(5, 10)]
    max_diff = -1

    print("testing conv with naive quant")
    from nni.algorithms.compression.pytorch.quantization import NaiveQuantizer
    conv = Conv2d(in_channels, out_channels, 3, stride=2)
    init(conv)
    conv2 = deepcopy(conv)
    quant_conf = [{"quant_types": ['weight'],
                   'quant_bits': 8,
                   "op_types": ["Conv2d"]}]
    quantizer = NaiveQuantizer(conv2, quant_conf)
    quantizer.compress()
    max_diff = max(test_nni_layer(conv, conv2, batch, in_channels, shapes),
                   max_diff)
    print(f"maximum difference {max_diff}.")

    print("testing conv with LSQ quant")
    from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
    from torch.optim import SGD

    config_list = [{'quant_types': ['input', 'output', 'weight'],
                    'quant_bits': {'input': 8, 'weight': 8, 'output': 8},
                    'op_types': ['Conv2d']}]

    for stride in (1, 2):
        conv = Conv2d(in_channels, out_channels, 3, stride=stride)
        conv2 = deepcopy(conv)
        optim = SGD(conv2.parameters(), 1e-2)
        print('quantizing...')
        quantizer = QAT_Quantizer(conv2, config_list, optim,
                                  rand(batch, in_channels, 8, 7)*2-1)
        quantizer.compress()
        print("Quantized!!")

    print("quantizing ultralytics model with QAT...")
    from synet import get_model
    from synet.ultralytics_patches import patch_ultralytics
    patch_ultralytics("sabre")
    model = get_model('/mnt/bpd_architecture/zaccy/models/synet/svga3-5.1/skp2_vga/weights/best.pt', raw=False)
    # get rid of head
    model.model.model = model.model.model
    optim = SGD(model.model.parameters(), 1e-2)
    config_list = [{'quant_types': [],  # 'weight', 'input', 'output'],
                    'quant_bits': {},  # 'weight': 8, 'input': 8, 'output': 8},
                    'op_types': ['Conv2d']}]
    model.model.train()
    quantizer = QAT_Quantizer(model.model, config_list, optim,
                              rand(batch, 3, 64, 64)*2-1)
    quantizer.compress()
    print("quantized!!")


def run_quantize():
    print("running quantize on skp2.yaml with ultralytics backend")
    environ["CUDA_VISIBLE_DEVICES"] = ''
    argv[1:] = ["quantize", "--backend", "ultralytics", "--cfg", "sabre-keypoint-qvga.yaml"]
    synet_main()
    return 0

def run_ultralytics():
    environ["CUDA_VISIBLE_DEVICES"] = ''
    argv[1:] = ["ultralytics", "train", "model=sabre-keypoint-qvga.yaml", "data=coco-pose.yaml"]
    synet_main()
    return 0

tests = {
    "tf": run_tf,
    #"nni": run_nni,
    "quantize": run_quantize,
    #"ultralytics": run_ultralytics,
}


def run(mode):
    if mode == "all":
        ret = False
        for run_test in tests.values():
            ret |= run_test()
        exit(ret)
    else:
        exit(tests[mode]())

def main():
    run(**vars(parse_opt()))
