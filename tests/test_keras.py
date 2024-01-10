from numpy import absolute
from torch import rand, no_grad
from torch.nn.init import uniform_

from synet.base import askeras


BATCH_SIZE = 2
IN_CHANNELS = 5
OUT_CHANNELS = 7
SHAPES = [(i, i) for i in range(4, 8)]
MAX_DIFF = -1
TOLERANCE = 2e-4


def diff_arr(out1, out2):
    """compare two arrays.  Return the max difference."""
    if isinstance(out1, (list, tuple)):
        assert isinstance(out2, (list, tuple))
        return max(diff_arr(o1, o2) for o1, o2 in zip(out1, out2))
    assert all(s1 == s2 for s1, s2 in zip(out1.shape, out2.shape)), \
        (out1.shape, out2.shape)
    return absolute(out1 - out2).max()


def t_actv_to_k_helper(actv):
    # if isinstance(actv, (tuple, list)):
    #     return tuple(t_actv_to_k_helper(a) for a in actv)
    if len(actv.shape) == 4:
        tp = 0, 2, 3, 1
    elif len(actv.shape) == 3:
        tp = 0, 2, 1
    elif len(actv.shape) == 2:
        tp = 0, 1
    return actv.detach().numpy().transpose(*tp)


def t_actv_to_k(actv):
    return [t_actv_to_k_helper(a) for a in actv] if isinstance(actv, (tuple, list)) \
        else t_actv_to_k_helper(actv)


def validate_layer(layer, torch_inp, **akwds):
    """Given synet layer, test on some torch input activations and
return max error between two output activations

    """
    tout = layer(torch_inp[:])
    with askeras(imgsz=torch_inp[0].shape[-2:], **akwds):
        kout = layer(t_actv_to_k(torch_inp))
    if isinstance(kout, (list, tuple)):
        kout = [k.numpy() for k in kout]
    else:
        kout = kout.numpy()
    if isinstance(tout, dict):
        assert len(tout) == len(kout)
        return max(diff_arr(t_actv_to_k(tout[key]), kout[key])
                   for key in tout)
    elif isinstance(tout, list):
        assert len(tout) == len(kout)
        return max(diff_arr(t_actv_to_k(t), k)
                   for t, k in zip(tout, kout))
    return diff_arr(t_actv_to_k(tout), kout)


def init(module):
    for param in module.parameters():
        uniform_(param, -1)


def validate(layer, batch_size=BATCH_SIZE,
             in_channels=IN_CHANNELS, shapes=SHAPES, **akwds):
    """Run validate_layer on a set of random input shapes.  Prints the max
difference between all configurations.

    """
    init(layer)
    max_diff = max(validate_layer(layer,
                                  [rand(batch_size, in_channels, *s)*2-1
                                   for s in shape]
                                  if len(shape) and isinstance(shape[0], tuple)
                                  else rand(batch_size, in_channels, *shape)*2-1,
                                  **akwds)
                   for shape in shapes)
    print("max_diff:", max_diff)
    assert max_diff < TOLERANCE


def test_conv2d():
    from synet.base import Conv2d
    print("testing Conv2d")
    in_channels = 12
    out_channels = 24
    for bias in True, False:
        for kernel, stride in ((1, 1), (2, 1), (3, 1), (3, 2), (4, 1),
                               (4, 2), (4, 3)):
            for padding in True, False:
                for groups in 1, 2, 3:
                    validate(Conv2d(in_channels, out_channels, kernel,
                                    stride, bias, padding),
                             in_channels=in_channels)


def test_convtranspose():
    from synet.base import ConvTranspose2d
    validate(ConvTranspose2d(IN_CHANNELS, OUT_CHANNELS, 2, 2, 0, bias=True))


def test_relu():
    from synet.base import ReLU
    validate(ReLU(.6))


def test_upsample():
    from synet.base import Upsample
    for scale_factor in 1, 2, 3:
        for mode in Upsample.allowed_modes:
            validate(Upsample(scale_factor, mode))


def test_globavgpool():
    from synet.base import GlobalAvgPool
    validate(GlobalAvgPool())


def test_dropout():
    from synet.base import Dropout
    for p in 0.0, 0.5, 1.0:
        for inplace in True, False:
            layer = Dropout(p, inplace=inplace)
            layer.eval()
            validate(layer)


def test_linear():
    from synet.base import Linear
    for bias in True, False:
        validate(Linear(IN_CHANNELS, OUT_CHANNELS, bias), shapes=[()])


def test_batchnorm():
    from synet.base import BatchNorm
    validate(BatchNorm(IN_CHANNELS), train=True)


def test_ultralytics_detect():
    from synet.backends.ultralytics import Detect
    for sm_split in ((True, None), (2, True)):
        layer = Detect(80, (IN_CHANNELS, IN_CHANNELS), *sm_split)
        layer.eval()
        layer.export = True
        layer.format = "tflite"
        layer.stride[0], layer.stride[1] = 1, 2
        validate(layer,
                 shapes=[(( 4,  6), (2, 3)),
                         (( 5,  7), (3, 4)),
                         (( 6,  8), (3, 4))],
                 xywh=True)

def test_ultralytics_pose():
    from synet.backends.ultralytics import Pose
    for sm_split in ((True, None), (2, True)):
        for kpt_shape in ([17, 2], [17,3]):
            layer = Pose(80, kpt_shape, (IN_CHANNELS, IN_CHANNELS), *sm_split)
            layer.eval()
            layer.export = True
            layer.format = "tflite"
            layer.stride[0], layer.stride[1] = 1, 2
            validate(layer,
                     shapes=[(( 4,  6), (2, 3)),
                             (( 5,  7), (3, 4)),
                             (( 6,  8), (3, 4))],
                     xywh=True)


def test_ultralytics_segment():
    from synet.backends.ultralytics import Segment
    layer = Segment(nc=80, nm=32, npr=256, ch=(IN_CHANNELS, IN_CHANNELS))
    layer.eval()
    layer.export = True
    layer.format = "tflite"
    layer.stride[0], layer.stride[1] = 1, 2
    validate(layer,
             shapes=[(( 4,  4), (2, 2)),
                     (( 8,  8), (4, 4)),
                     ((12, 12), (6, 6))],
             xywh=True)

def test_ultralytics_classify():
    from synet.backends.ultralytics import Classify
    layer = Classify(None, c1=IN_CHANNELS, c2=OUT_CHANNELS)
    layer.eval()
    layer.export = True
    layer.format = 'tflite'
    validate(layer)

