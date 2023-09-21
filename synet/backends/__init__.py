from importlib import import_module
from shutil import copy

from ..zoo import in_zoo, get_config, get_configs


class Backend:
    def get_model(self, model_path):
        """Load model from config, or pretrained save."""
        raise NotImplementedError("Please subclass and implement")

    def get_shape(self, model):
        """Get shape of model."""
        raise NotImplementedError("Please subclass and implement")

    def patch(self):
        """Initialize backend to utilize Synet Modules"""
        raise NotImplementedError("Please subclass and implement")

    def val_post(self, weights, tflite, val_post, conf_thresh=.25,
                 iou_thresh=.7):
        """Default conf_thresh and iou_thresh (.25 and .75 resp.)
        taken from ultralytics/cfg/default.yaml.

        """
        raise NotImplementedError("Please subclass and implement")

    def tf_post(self, tflite, val_post, conf_thresh, iou_thresh):
        """Loads the tflite, loads the image, preprocesses the image,
        evaluates the tflite on the pre-processed image, and performs
        post-processing on the tflite output with a given confidence
        and iou threshold.

        :param tflite: Path to tflite file, or a raw tflite buffer
        :param val_post: Path to image to evaluate on.
        :param conf_thresh: Confidence threshould.  See val_post docstring
        above for default value details.
        :param iou_thresh: IoU threshold for NMS.  See val_post docstring
        above for default value details.

        """
        raise NotImplementedError("Please subclass and implement")

    def get_chip(self, model):
        """Get chip of model."""
        raise NotImplementedError("Please subclass and implement")

    def maybe_grab_from_zoo(self, config_path):
        if in_zoo(config_path, self.name):
            copy(get_config(config_path, self.name), config_path)
        return config_path

    def get_configs(self):
        return get_configs(self.name)


def get_backend(name):
    return import_module(f".{name}", __name__).Backend()
