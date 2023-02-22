from numpy import array

from os.path import join, dirname
from json import load
from tensorflow.keras.models import load_model
def get_katananet_model(model_path, input_shape, low_thld, **kwds):
    """Load katananet model.

model_dir: str
    path to directory with model.h5.
input_shape: iterable of ints
    shape of the cell run.

    """
    raw_model = load_model(model_path, compile=False)
    anchor_params = load(open(join(dirname(model_path), "anchors.json")))
    anchors = gen_anchors(input_shape, **anchor_params)
    def model(image):
        (deltas,), (scores,) = raw_model.predict_on_batch(preproc(image))
        # low thld
        keep = scores.max(1) > low_thld
        deltas, anchor_keep, scores = deltas[keep], anchors[keep], scores[keep]
        # get_abs_coords only get coordinates relative to cell
        boxes = get_abs_coords(deltas, anchor_keep,
                               training_scale=.2, training_shift=0,
                               maxx=image.shape[-1], maxy=image.shape[-2])
        # apply nms
        boxes, scores = nms(boxes, scores, threshold=.3)
        return boxes, scores.squeeze(-1)

    return model

from numpy import float32, expand_dims
def preproc(image):
    """Convert image values from integer range [0,255] to float32
    range [-1,1)."""
    if len(image.shape) < 3:
        image = expand_dims(image, 0)
    return image.astype(float32) / 128 - 1

from numpy import zeros, arange, concatenate
from math import ceil
def gen_anchors(image_shape, strides, sizes, ratios, scales):
    imy, imx = image_shape
    all_anchors = []
    scales = array(scales).reshape(-1, 1)
    ratios = array(ratios).reshape(-1, 1, 1)**.5
    for stride, size in zip(strides, sizes):
        py, px = ceil(imy/stride), ceil(imx/stride)
        anchors = zeros((py, px, len(ratios), len(scales), 4))
        # anchors as (xc, yc, w, h)
        anchors[...,2:] = size * scales
        # apply ratios
        anchors[...,2] /= ratios[...,0]
        anchors[...,3] *= ratios[...,0]
        # convert to xyxy
        anchors[...,:2] -= anchors[...,2:]/2
        anchors[...,2:] /= 2
        # add offsets for xy position
        anchors[...,0::2] += ((arange(px) + 0.5) * stride).reshape(-1,1,1,1)
        anchors[...,1::2] += ((arange(py) + 0.5) * stride).reshape(-1,1,1,1,1)
        all_anchors.append(anchors.reshape(-1, 4))
    return concatenate(all_anchors)

from numpy import clip, newaxis
def get_abs_coords(deltas, anchors, training_scale, training_shift,
                   maxx, maxy):
    """Convert model output (deltas) into "absolute" coordinates.
    Note: absolute coordinates here are still relative to the grid
    cell being run.

deltas: ndarray
    nx4 array of xyxy values.
anchors: ndarray
    nx4 array of ofsets.
training_scale: float
    scale specific to our training code.  For us always set to .2.
training_shift: float
    shift specific to our training code.  For us is always 0.
maxx: float
    Max x value. Used to clip final results to fit in cell.
maxy: float
    Max y value. Used to clip final results to fit in cell.

    """
    width, height = (anchors[:, 2:4] - anchors[:, 0:2]).T
    deltas = deltas * training_scale + training_shift
    deltas[:,0::2] *= width [...,newaxis]
    deltas[:,1::2] *= height[...,newaxis]
    boxes = deltas + anchors
    boxes[:, 0::2] = clip(boxes[:, 0::2], 0, maxx)
    boxes[:, 1::2] = clip(boxes[:, 1::2], 0, maxy)
    return boxes

from numpy import argsort, maximum, minimum
def nms(boxes, score, threshold):
    """
    Non-maxima supression to remove redundant boxes
    :param bounding_boxes: Input box coordinates
    :param confidence_score: Confidence scores for each box
    :param labels: Class label for each box
    :param threshold: Only boxes above this threshold are selected
    :return:
    Final detected boxes
    """
    if not len(boxes):
        return boxes, score

    # coordinates of bounding boxes
    all_x1 = boxes[:, 0]
    all_y1 = boxes[:, 1]
    all_x2 = boxes[:, 2]
    all_y2 = boxes[:, 3]

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (all_y2 - all_y1 + 1) * (all_x2 - all_x1 + 1)

    # Sort by confidence score of bounding boxes
    order = argsort(-score.max(-1))

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[0]
        order = order[1:]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(boxes[index])
        picked_score.append(score[index])

        # Compute ordinates of intersection-over-union(IOU)
        y1 = maximum(all_y1[index], all_y1[order])
        x1 = maximum(all_x1[index], all_x1[order])
        y2 = minimum(all_y2[index], all_y2[order])
        x2 = minimum(all_x2[index], all_x2[order])

        # Compute areas of intersection-over-union
        w = maximum(0.0, x2 - x1 + 1)
        h = maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order] - intersection)

        order = order[ratio < threshold]

    return array(picked_boxes), array(picked_score)
