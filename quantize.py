#!/usr/bin/env python

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

