#!/usr/bin/env python
from argparse import ArgumentParser
from glob import glob
from os.path import abspath, commonpath, dirname, isabs, isdir, join, splitext
from random import shuffle

from cv2 import imread, resize
from keras import Input, Model
from numpy import float32
from numpy.random import rand
from tensorflow import int8, lite
from torch import no_grad

from .base import askeras
from .backends import get_backend


def parse_opt():
    """parse_opt() is used to make it compatible with how yolov5
obtains arguments.

    """
    parser = ArgumentParser()
    parser.add_argument("--backend", type=get_backend)
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--image-shape", nargs=2, type=int)
    parser.add_argument("--data")
    parser.add_argument("--kwds", nargs="+", default=[])
    parser.add_argument("--channels", "-c", default=3, type=int)
    parser.add_argument("--number", "-n", default=500, type=int)
    parser.add_argument("--val-post",
                        help="path to sample image to validate on.")
    parser.add_argument("--tflite",
                        help="path to existing tflite (for validating).")
    return parser.parse_args()


def run(backend, image_shape, weights, cfg, data, number, channels, kwds,
        val_post, tflite):
    """Entrypoint to quantize.py.  Quantize the model specified by
weights (falling back to cfg), using samples from the data yaml with
image shape image_shape, using only number samples.

    """
    backend.patch()

    if tflite is None:
        tflite = get_tflite(backend, image_shape, weights, cfg, data,
                            number, channels, kwds)

    if val_post:
        backend.val_post(weights, tflite, val_post)


def get_tflite(backend, image_shape, weights, cfg, data, number,
               channels, kwds):

    # obtain the pytorch model from weights or cfg, prioritizing weights
    model = backend.get_model(weights or cfg)

    # maybe get image shape
    if image_shape is None:
        image_shape = backend.get_shape(model)

    # generate keras model
    inp = Input(image_shape+[channels], batch_size=1)
    with askeras(imgsz=image_shape, quant_export=True,
                 **dict(s.split("=") for s in kwds)), \
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
    return quantize(kmodel, data, image_shape, number, out, channels)


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
    elif data is None:
        converter.representative_dataset = \
            lambda: phony_data(image_shape, channels)
    else:
        converter.representative_dataset = \
            lambda: representative_data(data, image_shape, N, channels)

    # quantize
    tflite_quant_model = converter.convert()

    # write out tflite
    if out_path:
        with open(out_path, "wb") as f:
            f.write(tflite_quant_model)

    return tflite_quant_model


def representative_data(data, image_shape, N, channels):
    """Obtains dataset from data, samples N samples, and returns those
samples reshaped to image_shape.

    """
    from yolov5.utils.general import check_dataset
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


def main():
    return run(**vars(parse_opt()))
