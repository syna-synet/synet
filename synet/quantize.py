#!/usr/bin/env python

from argparse import ArgumentParser
def parse_opt():
    parser = ArgumentParser()
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--image-shape")
    return parser.parse_args()

from os.path import splitext
from keras import Input, Model
from torch import no_grad
from .base import askeras
from .yolov5_patches import get_yolov5_model
def run(image_shape, weights, cfg):
    model = get_yolov5_model(weights if weights else cfg, raw=True)
    data_shape = image_shape+[1]
    inp = Input(data_shape, batch_size=1)
    with askeras(imgsz=image_shape), no_grad():
        kmodel = Model(inp, model(inp))
    if not weights:
        out = "model.tflite"
    else:
        out = splitext(weights)[0]+".tflite"
    quantize(kmodel, data_shape, out)

from numpy import float32
from numpy.random import rand
def phony_data(data_shape):
    for _ in range(2):
        yield [rand(1, *data_shape).astype(float32)]

from tensorflow import lite, int8
def quantize(kmodel, data_shape, out_path):
    converter = lite.TFLiteConverter.from_keras_model(kmodel)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS,
                                           lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = int8
    converter.inference_output_type = int8
    converter.representative_dataset = lambda:phony_data(data_shape)
    tflite_quant_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_quant_model)


if __name__ == "__main__":
    from keras import Input, Model
    from models import PersonDetect, askeras
    data_shape = (480, 640, 1)
    tmodel = PersonDetect()
    input = Input(data_shape, batch_size=1)
    with askeras:
        kmodel = Model(input, tmodel(input))
    kmodel.save("model.h5")
    quantize(kmodel, data_shape, "model.tflite")

