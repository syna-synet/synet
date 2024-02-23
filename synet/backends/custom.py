from .base import askeras

from object_detection.models.tf import TFDetect as PC_TFDetect
from tensorflow.math import ceil
import tensorflow as tf
class TFDetect(PC_TFDetect):
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):
        super().__init__(nc, anchors, ch, imgsz, w)
        for i in range(self.nl):
            ny, nx = (ceil(self.imgsz[0] / self.stride[i]),
                      ceil(self.imgsz[1] / self.stride[i]))
            self.grid[i] = self._make_grid(nx, ny)

    # copy call method, but replace // with ceil div
    def call(self, inputs):
        if askeras.kwds.get('deploy'):
            return self.deploy(inputs)
        z = []  # inference output
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            ny, nx = (ceil(self.imgsz[0] / self.stride[i]),
                      ceil(self.imgsz[1] / self.stride[i]))
            x[i] = tf.transpose(tf.reshape(x[i], [-1, ny * nx, self.na, self.no]), [0, 2, 1, 3])

            if not self.training:  # inference
                y = tf.sigmoid(x[i])
                xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                y = tf.concat([xy, wh, y[..., 4:]], -1)
                # y = tf.concat([xy, wh, y[..., 4:]], 3)
                z.append(tf.reshape(y, [-1, self.na * ny * nx, self.no]))

        return x if self.training else (tf.concat(z, 1), x)

    def deploy(self, inputs):
        assert inputs[0].shape[0] == 1, 'requires batch_size == 1'
        box1, box2, cls = [], [], []
        for mi, xi, gi, ai, si in zip(self.m, inputs, self.grid, self.anchor_grid, self.stride):
            x = tf.reshape(tf.sigmoid(mi(xi)), (1, -1, self.na, self.no))
            xy = (x[..., 0:2] * 2 + (tf.transpose(gi, (0, 2, 1, 3)) - .5)) * si
            wh = (x[..., 2:4] * 2) ** 2 * tf.transpose(ai, (0, 2, 1, 3))
            box1.append(tf.reshape(xy - wh/2, (1, -1, 2)))
            box2.append(tf.reshape(xy + wh/2, (1, -1, 2)))
            cls.append(tf.reshape(x[..., 4:5]*x[..., 5:], (1, -1, x.shape[-1]-5)))
        return (tf.concat(box1, 1, name='box1'),
                tf.concat(box2, 1, name='box2'),
                tf.concat(cls,  1, name='cls'))


from object_detection.models.yolo import Detect as PC_PTDetect
class Detect(PC_PTDetect):
    def __init__(self, *args, **kwds):
        if len(args) == 4:
            args = args[:3]
        # construct normally
        super().__init__(*args, **kwds)
        # save args/kwargs for later construction of TF model
        self.args = args
        self.kwds = kwds
    def forward(self, x, theta=None):
        if askeras.use_keras:
            assert theta is None
            return self.as_keras(x)
        return super().forward(x, theta=theta)
    def as_keras(self, x):
        return TFDetect(*self.args, imgsz=askeras.kwds["imgsz"],
                        w=self, **self.kwds
                        )(x)

from object_detection.models import yolo
from importlib import import_module
def patch_custom(chip):
    # patch custom.models.yolo
    module = import_module(f'..{chip}', __name__)
    setattr(yolo, chip, module)
    yolo.Concat = module.Cat
    yolo.Detect = module.Detect = Detect
