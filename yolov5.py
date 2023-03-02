from yolov5.models.tf import TFDetect as Yolo_TFDetect
import tensorflow as tf
from tensorflow.math import ceil
class TFDetect(Yolo_TFDetect):

    # use orig __init__, but make nx, ny calculated via ceil div
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):
        super().__init__(nc, anchors, ch, imgsz, w)
        for i in range(self.nl):
            ny, nx = (ceil(self.imgsz[0] / self.stride[i]),
                      ceil(self.imgsz[1] / self.stride[i]))
            self.grid[i] = self._make_grid(nx, ny)

    # copy call method, but replace // with ceil div
    def call(self, inputs):
        z = []  # inference output
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            ny, nx = (ceil(self.imgsz[0] / self.stride[i]),
                      ceil(self.imgsz[1] / self.stride[i]))
            x[i] = tf.reshape(x[i], [-1, ny * nx, self.na, self.no])

            if not self.training:  # inference
                y = x[i]
                grid = tf.transpose(self.grid[i], [0, 2, 1, 3]) - 0.5
                anchor_grid = tf.transpose(self.anchor_grid[i], [0, 2, 1, 3])*4
                xy = (tf.sigmoid(y[..., 0:2]) * 2 + grid) * self.stride[i]# xy
                wh = tf.sigmoid(y[..., 2:4]) ** 2 * anchor_grid
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]],
                                  dtype=tf.float32)
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]],
                                  dtype=tf.float32)
                y = tf.concat([xy, wh, tf.sigmoid(y[..., 4:5 + self.nc]),
                               y[..., 5 + self.nc:]], -1)
                z.append(tf.reshape(y, [-1, self.na * ny * nx, self.no]))

        return tf.transpose(x, [0, 2, 1, 3]) \
            if self.training else (tf.concat(z, 1),)


from .models import askeras
from yolov5.models.yolo import Detect as Yolo_PTDetect
class Detect(Yolo_PTDetect):

    def __init__(self, *args, **kwds):
        # to account for args hack.
        if len(args) == 4:
            args = args[:3]
        # construct normally
        super().__init__(*args, **kwds)
        # save args/kwargs for later construction of TF model
        self.args = args
        self.kwds = kwds

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        return super().forward(x)

    def as_keras(self, x):
        returnTFDetect(*self.args, imgsz=askeras.kwds["imgsz"],
                       w=self, **self.kwds
                       )(x)
