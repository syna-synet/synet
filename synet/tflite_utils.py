from typing import Optional, List

from cv2 import imread
from numpy import (newaxis, ndarray, int8, float32,
                   concatenate as cat, max as npmax, argmax)
from tensorflow import lite
from torch import tensor
from torchvision.ops import nms


def tf_post(tflite, val_post, conf_thresh, iou_thresh):
    """Loads the tflite, loads the image, preprocesses the image,
    evaluates the tflite on the pre-processed image, and performs
    post-processing on the tflite output with a given confidence and
    iou threshold.

    :param tflite: Path to tflite file, or a raw tflite buffer
    :param val_post: Path to image to evaluate on.
    :param conf_thresh: Confidence threshold.  See val_post docstring
    above for default value details.
    :param iou_thresh: IoU threshold for NMS.  See val_post docstring
    above for default value details.

    """

    # initialize tflite interpreter.
    interpreter = lite.Interpreter(**{"model_path"
                                      if isinstance(tflite, str) else
                                      "model_content"
                                      : tflite})

    # make image RGB (not BGR) channel order, BCHW dimensions, and
    # in the range [0, 1].  cv2's imread reads in BGR channel
    # order, with dimensions in Height, Width, Channel order.
    # Also, imread keeps images as integers in [0,255].  Normalize
    # to floats in [0, 1].  Also, model expects a batch dimension,
    # so add a dimension at the beginning
    im = imread(val_post)[newaxis, ..., ::-1] / 255

    # Run tflite interpreter on the input image
    tout = tflite_interpreter_runner(interpreter, im)

    # Procces the tflite output to be one tensor
    preds = tflite_process_output(tout, xywh=False, classes_to_index=True)

    # perform nms
    preds = apply_post_process(preds, conf_thresh, iou_thresh)
    return preds


def tflite_interpreter_runner(interpreter: Optional[lite.Interpreter],
                              input_arr: ndarray) -> List[ndarray]:
    """
    Evaluating tflite interpreter on input data
    :param interpreter: tflite interpreter
    :param input_arr: model input
    :return: list [bbox_x1y1, kpts_visibility, kpts_xy, bbox_x2y2, classes]
    """
    interpreter.allocate_tensors()
    in_scale, in_zero = interpreter.get_input_details()[0]['quantization']
    out_scale_zero_index = [(*detail['quantization'], detail['index'])
                            for detail in interpreter.get_output_details()]
    # run tflite on image
    assert interpreter.get_input_details()[0]['index'] == 0
    assert interpreter.get_input_details()[0]['dtype'] is int8
    interpreter.set_tensor(0, (input_arr / in_scale + in_zero).astype(int8)
                           )
    interpreter.invoke()
    # indexing below with [0] removes the batch dimension
    return [
        (interpreter.get_tensor(index)[0].astype(float32) - zero) * scale
        for scale, zero, index in out_scale_zero_index]


def tflite_process_output(model_output: List[ndarray],
                          xywh: Optional[bool] = False,
                          classes_to_index: Optional[bool] = True
                          ) -> ndarray:
    """
    This method reordering the tflite output structure to be fit to run
    post process such as NMS etc. See the description below
    :param model_output: [bbox_x1y1, kpts_visibility, kpts_xy, bbox_x2y2,
        classes]
    :param xywh: flag for convert xyxy to xywh
    :param classes_to_index: flag for convert the classes output logits
        to single class index
    :return: ndarray of (x1y1x2y2 or xywh, conf,
        class_index or classes logits, kpts_xyv,...]
    """

    b1, kpresence, kcoord, b2, bclass = model_output

    if xywh:
        bbox_xy_center = (b1 + b2) / 2
        bbox_wh = b2 - b1
        bbox = cat([bbox_xy_center, bbox_wh], -1)
    else:
        bbox = cat([b1, b2], -1)

    # the number of keypoints and classes can be found from output shapes
    _, num_kpts, _ = kcoord.shape
    _, num_classes = bclass.shape

    # find the box class confidence and number.  class_num is only
    # relevant for multiclass.
    conf = npmax(bclass, axis=1, keepdims=True)

    if classes_to_index:
        bclass = argmax(bclass, axis=1, keepdims=True)

    # Combine results AFTER dequantization (so values in 0-1 and
    # 0-255 can be combined).
    return cat((bbox, conf, bclass, cat(
        (kcoord, kpresence), -1).reshape(-1, num_kpts * 3)), axis=-1)


def apply_post_process(preds: ndarray, conf_thresh: float,
                       iou_thresh: float):
    """
    Apply NMS on ndarray of: [x1y1x2y2, conf, class_index, kpts_xyv,...]
    :param preds:
    :return: same structure [x1y1x2y2, conf, class_index, kpts_xyv,...]
    """
    # perform confidence thresholding, and convert to tensor for nms.
    preds = tensor(preds[preds[:, 4] > conf_thresh])
    # Perform NMS
    # https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html
    return preds[nms(preds[:, :4], preds[:, 4], iou_thresh)]
