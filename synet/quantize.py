#!/usr/bin/env python

from argparse import ArgumentParser
from os.path import splitext, commonpath, dirname, abspath
from keras import Input, Model
from torch import no_grad
from .base import askeras
from . import get_model
from tensorflow import lite, int8
from cv2 import imread, resize
from glob import glob
from numpy import float32
from os.path import isdir, join, isabs
from random import shuffle
from numpy import float32
from numpy.random import rand
try:
    from .yolov5_patches import get_yolov5_model
    from yolov5.utils.general import check_dataset
except ImportError:
    pass


def parse_opt():
    """parse_opt() is used to make it compatible with how yolov5
obtains arguments.

    """
    parser = ArgumentParser()
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--image-shape", nargs=2, type=int)
    parser.add_argument("--data")
    parser.add_argument("--kwds", nargs="+", default=[])
    parser.add_argument("--channels", "-c", default=1, type=int)
    parser.add_argument("--number", "-n", default=500, type=int)
    return parser.parse_args()


def run(image_shape, weights, cfg, data, number, channels, kwds):
    """Entrypoint to quantize.py.  Quantize the model specified by
weights (falling back to cfg), using samples from the data yaml with
image shape image_shape, using only number samples.

    """
    # obtain the pytorch model from weights or cfg, prioritizing weights
    model = get_model(weights or cfg, raw=True)

    # generate keras model
    inp = Input(image_shape+[channels], batch_size=1)
    with askeras(imgsz=image_shape, **dict(s.split("=") for s in kwds)), \
         no_grad():
        kmodel = Model(inp, model(inp))

    # determine output file path
    if not weights and dirname(__file__) == commonpath((__file__,
                                                        abspath(cfg))):
        # if pulling from model zoo, just place a model.tflite in cwd
        out = "model.tflite"
    else:
        # otherwise, use input name, but with swapped out extension
        out = splitext(weights or cfg)[0]+".tflite"

    # quantize the model
    quantize(kmodel, data, image_shape, number, out, channels)


def quantize(kmodel, data, image_shape, N=500, out_path=None, channels=1,
             generator=None):
    """Given a keras model, kmodel, and data yaml at data, quantize
using N samples reshaped to image_shape and place the output model at
out_path.

    """
    # more or less boilerplate code
    converter = lite.TFLiteConverter.from_keras_model(kmodel)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.inference_input_type = int8
    converter.inference_output_type = int8

    if generator:
        converter.representative_dataset = generator
    # use our data if given
    elif data is None:
        converter.representative_dataset = \
            lambda: phony_data(image_shape, channels)
    else:
        converter.representative_dataset = \
            lambda: representative_data(data, image_shape, N, channels)
    # quantize
    tflite_quant_model = converter.convert()
    if out_path is None: return tflite_quant_model
    # write out tflite
    with open(out_path, "wb") as f:
        f.write(tflite_quant_model)


def representative_data(data, image_shape, N, channels):
    """Obtains dataset from data, samples N samples, and returns those
samples reshaped to image_shape.

    """
    data_dict = check_dataset(data)
    path = data_dict.get('test', data_dict['val'])
    f = []
    for p in path if isinstance(path, list) else [path]:
        if isdir(p):
            f += glob(join(p, "**", "*.*"), recursive=True)
        else:
            f += [t if isabs(t) else join(dirname(p), t)
                  for t in open(p).read().splitlines()]
    shuffle(f)
    for fpth in f[:N]:
        im = imread(fpth)
        if im.shape[-1] != channels:
            assert channels == 1
            im = im.mean(-1, keepdims=True)
        if im.shape[0] != image_shape[0] or im.shape[1] != image_shape[1]:
            im = resize(im, image_shape)
        yield [im.reshape((1, *image_shape, channels)).astype(float32) / 255]


def phony_data(image_shape, channels):
    for _ in range(2):
        yield [rand(1, *image_shape, channels).astype(float32)]
