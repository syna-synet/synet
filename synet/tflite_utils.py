"""This module exists to hold all tflite related processing.  The main
benefit of keeping this in a seperate modules is so that large
dependencies (like ultralytics) need not be imported when simulating
tflite execution (like for demos).  However, visualization
(interpretation} of the model is left to ultralytics.  This module
also serves as a reference for C and/or other implementations;
however, do read any "Notes" sections in any function docstrings

"""

from typing import Optional, List

from cv2 import imread
from numpy import (newaxis, ndarray, int8, float32 as npfloat32,
                   concatenate as cat, max as npmax, argmax, moveaxis)
from tensorflow import lite
from torch import tensor, float32 as torchfloat32
from torchvision.ops import nms


def tf_run(tflite, img, conf_thresh, iou_thresh, backend, task):
    """Run a tflite model on an image, including post-processing.

    Loads the tflite, loads the image, preprocesses the image,
    evaluates the tflite on the pre-processed image, and performs
    post-processing on the tflite output with a given confidence and
    iou threshold.

    Parameters
    ----------
    tflite : str or buffer
        Path to tflite file, or a raw tflite buffer
    img : str or ndarray
        Path to image to evaluate on, or the image as read by cv2.imread.
    conf_thresh : float
        Confidence threshold applied before NMS
    iou_thresh : float
        IoU threshold for NMS
    backend : {"ultralytics"}
        The backend which is used.  For now, only "ultralytics" is supported.
    task : {"classify", "detect", "segment", "pose"}
        The computer vision task to which the tflite model corresponds.

    Returns
    -------
    ndarray or tuple of ndarrys
        Return the result of running preprocessing, tflite evaluation,
        and postprocessing on the input image.  Segmentation models
        produce two outputs as a tuple.

    """

    assert backend == "ultralytics", "only supports ultralytics"

    # initialize tflite interpreter.
    interpreter = lite.Interpreter(**{"model_path"
                                      if isinstance(tflite, str) else
                                      "model_content"
                                      : tflite})

    # read the image if given as path
    if isinstance(img, str):
        img = imread(img)

    # make image RGB (not BGR) channel order, BCHW dimensions, and
    # in the range [0, 1].  cv2's imread reads in BGR channel
    # order, with dimensions in Height, Width, Channel order.
    # Also, imread keeps images as integers in [0,255].  Normalize
    # to floats in [0, 1].  Also, model expects a batch dimension,
    # so add a dimension at the beginning
    img = img[newaxis, ..., ::-1] / 255
    # FW TEAM NOTE: It might be strange converting to float here, but
    # the model might have been quantized to use a subset of the [0,1]
    # range, i.e. 220 could map to 255

    # Run tflite interpreter on the input image
    tout = run_interpreter(interpreter, img)

    # Procces the tflite output to be one tensor
    preds = concat_reshape(tout, task)

    # perform nms
    return apply_nms(preds, conf_thresh, iou_thresh)


def run_interpreter(interpreter: Optional[lite.Interpreter],
                    input_arr: ndarray) -> List[ndarray]:
    """Evaluating tflite interpreter on input data

    Parameters
    ----------
    interpreter : Interpreter
        the tflite interpreter to run
    input_arr : 4d ndarray
        tflite model input with shape (batch, height, width, channels)

    Returns
    -------
    list
        List of output arrays from running interpreter.  The order and
        content of the output is specific to the task and if model
        outputs xywh or xyxy.
    """

    interpreter.allocate_tensors()
    in_scale, in_zero = interpreter.get_input_details()[0]['quantization']
    out_scale_zero_index = [(*detail['quantization'], detail['index'])
                            for detail in
                            sorted(interpreter.get_output_details(),
                                   key=lambda x:x['name'])]
    # run tflite on image
    assert interpreter.get_input_details()[0]['index'] == 0
    assert interpreter.get_input_details()[0]['dtype'] is int8
    interpreter.set_tensor(0, (input_arr / in_scale + in_zero).astype(int8))
    interpreter.invoke()
    # indexing below with [0] removes the batch dimension, which is always 1.
    return [(interpreter.get_tensor(index)[0].astype(npfloat32) - zero) * scale
            for scale, zero, index in out_scale_zero_index]


def concat_reshape(model_output: List[ndarray],
                   task: str,
                   xywh: Optional[bool] = False,
                   classes_to_index: Optional[bool] = True
                   ) -> ndarray:
    """Concatenate, reshape, and transpose model output to match pytorch.

    This method reordering the tflite output structure to be fit to run
    post process such as NMS etc.

    Parameters
    ----------
    model_output : list
        Output from running tflite.
    task : {"classify", "detect", "segment", "pose"}
        The task the model performs.
    xywh : bool, default=False
        If true, model output should be converted to xywh
    classes_to_index : bool, default=True
        If true, convert the classes output logits to single class index

    Returns
    -------
    ndarray or list
        Final output after concatenating and reshaping input.  Returns
        an ndarray for every task except "segment" which returns a
        tupule of two arrays.

    Notes
    -----
    The python implementation here concats all output before applying
    nms.  This is to mirror the original pytorch implementation.  For
    a more efficient implementation, you may want to perform
    confidence thresholding and nms on the boxes and scores, masking
    other tensor appropriately, before reshaping and concatenating.

    Also, for segmentation, the proto mask is transposed (moveaxis())
    to match the pytorch convention.  When transcribing this code for
    other implementations, you may not want this behavior.

    """

    # interperate input tuple of tensors based on task.  Though
    # produced tflite always have output names like
    # "StatefulPartitionedCall:0", the numbers at the end are infact
    # alphabetically ordered by the final layer name for each output,
    # even though those names are discarded.  Hence, the following
    # variables are nammed to match the corresponding output layer
    # name and always appear in alphabetical order.
    if task == "pose":
        box1, box2, cls, kpts, pres = model_output
        _, num_kpts, _ = kpts.shape
    if task == "segment":
        box1, box2, cls, proto, seg = model_output
    if task == "detect":
        box1, box2, cls = model_output

    _, num_classes = cls.shape
    assert num_classes == 1, "apply_nms() hardcodes for num_classes=1"
    # obtain class confidences
    conf = npmax(cls, axis=1, keepdims=True),
    if classes_to_index:  # important for multi-class
        conf = (*conf, argmax(cls, axis=1, keepdims=True))

    # possibly convert to xywh if desired
    # FW TEAM NOTE: see comment below
    if xywh:
        bbox_xy_center = (box1 + box2) / 2
        bbox_wh = box2 - box1
        bbox = cat([bbox_xy_center, bbox_wh], -1)
    else:
        bbox = cat([box1, box2], -1)

    # return final concatenated output
    # FW TEAM NOTE: Though this procedure creates output consistent
    # with the original pytorch behavior of these models, you probably
    # want to do something more clever, i.e. perform NMS reading from
    # the arrays without concatenating.  At the very least, maybe do a
    # confidence filter before trying to copy the full tensors.  Also,
    # future models might have several times the output size, so keep
    # that in mind.
    if task == "segment":
        # FW TEAM NOTE: the second element here move channel axis to
        # beginning in line with pytorch behavior.  Maybe not relevent.

        # FW TEAM NOTE: the proto array is HUGE (HxWx64).  You
        # probably want to compute individual instance masks for your
        # implementation.  See the YOLACT paper on arxiv.org:
        # https://arxiv.org/abs/1904.02689.  Basically, for each
        # instance that survives NMS, generate the segmentation (only
        # HxW for each instance) by taking the iner product of seg
        # with each pixel in proto.
        return cat((bbox, *conf, seg), axis=-1), moveaxis(proto, -1, 0)
    if task == 'pose':
        return cat((bbox, *conf, cat((kpts, pres), -1
                                     ).reshape(-1, num_kpts * 3)),
                   axis=-1)
    if task == 'detect':
        return cat((bbox, *conf), axis=-1)


def apply_nms(preds: ndarray, conf_thresh: float, iou_thresh: float):
    """Apply NMS on ndarray prepared model output

    preds : ndarray or tuple of ndarray
        prepared model output.  Is a tuple of two arrays for "segment" task
    conf_thresh : float
        confidence threshold applied before NMS.

    Returns
    -------
    ndarray or tuple of ndarray
        same structure as preds, but with some values suppressed (removed).

    Notes
    -----
    This function converts ndarrays to pytorch tensors for two reasons:
     - the nms code requires torch tensor inputs
     - the output format becomes identical to that used by
       ultralytics, and so can be passed to an ultralytics visualizer.

    Also, as mentioned in the concat_reshape function, you may want to
    perform nms and thresholding before combining all the output.

    """

    # segmentation task returns a tuple of (preds, proto)
    if isinstance(preds, tuple):
        is_tuple = True
        preds, proto = preds
    else:
        is_tuple = False

    # perform confidence thresholding, and convert to tensor for nms.
    preds = tensor(preds[preds[:, 4] > conf_thresh], dtype=torchfloat32)

    # Perform NMS
    # https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html
    preds = preds[nms(preds[:, :4], preds[:, 4], iou_thresh)]
    return (preds, tensor(proto)) if is_tuple else preds
