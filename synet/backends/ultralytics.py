from importlib import import_module
from os.path import dirname, join, isfile, basename
from shutil import copy
from sys import argv

from cv2 import imread, imwrite
from numpy import (newaxis, int8, float32, concatenate as cat,
                   max as npmax, argmax)
from torch import tensor
from torch.nn import ModuleList
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.nn import tasks
from ultralytics.nn.modules.block import DFL as Torch_DFL
from ultralytics.nn.modules.head import (Pose as Torch_Pose,
                                         Detect as Torch_Detect)
from ultralytics.utils.ops import non_max_suppression

from . import Backend as BaseBackend
from ..base import askeras, Conv2d, ReLU
from ..layers import Sequential
from ..zoo import in_zoo, get_config


class DFL(Torch_DFL):
    def __init__(self, c1=16, sm_split=None):
        super().__init__(c1)
        weight = self.conv.weight
        self.conv = Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.conv.weight.data[:] = weight.data
        self.sm_split = sm_split

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        return super().forward(x)

    def as_keras(self, x):
        # b, ay, ax, c = x.shape
        from tensorflow.keras.layers import Reshape, Softmax
        if hasattr(self, "sm_split") and self.sm_split is not None:
            from tensorflow.keras.layers import Concatenate
            assert not (x.shape[0]*x.shape[1]*x.shape[2]*4) % self.sm_split
            x = Reshape((self.sm_split, -1, self.c1))(x)
            # tensorflow really wants to be indented like this.  I relent...
            return Reshape((-1, 4))(
                self.conv(
                    Concatenate(1)([
                        Softmax(-1)(x[:, i:i+1])
                        for i in range(x.shape[1])
                    ])
                )
            )

        return Reshape((-1, 4)
                       )(self.conv(Softmax(-1)(Reshape((-1, 4, self.c1))(x))))


class Detect(Torch_Detect):
    def __init__(self, nc=80, ch=(), sm_split=None, junk=None):
        super().__init__(nc, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.cv2 = ModuleList(Sequential([Conv2d(x, c2, 3, bias=True),
                                          ReLU(6),
                                          Conv2d(c2, c2, 3, bias=True),
                                          ReLU(6),
                                          Conv2d(c2, 4 * self.reg_max, 1,
                                                 bias=True)])
                              for x in ch)
        self.cv3 = ModuleList(Sequential([Conv2d(x, c3, 3, bias=True),
                                          ReLU(6),
                                          Conv2d(c3, c3, 3, bias=True),
                                          ReLU(6),
                                          Conv2d(c3, self.nc, 1, bias=True)])
                              for x in ch)
        self.dfl = DFL(sm_split=sm_split)
        self.type = "detect"

    def forward(self, x):
        if askeras.use_keras:
            return Detect.as_keras(self, x)
        return super().forward(x)

    def as_keras(self, x):
        from tensorflow.keras.layers import Reshape
        from tensorflow import meshgrid, range, stack, reshape, concat
        from tensorflow.math import ceil
        from tensorflow.keras.layers import Concatenate
        from tensorflow.keras.activations import sigmoid
        ltrb = Concatenate(-2)([self.dfl(cv2(xi)) * s.item()
                                for cv2, xi, s in
                                zip(self.cv2, x, self.stride)])
        H, W = askeras.kwds['imgsz']
        anchors = concat([stack((reshape((sx+.5)*s, (-1,)),
                                 reshape((sy+.5)*s, (-1,))),
                                -1)
                          for s, (sy, sx) in ((s.item(),
                                               meshgrid(range(ceil(H/s)),
                                                        range(ceil(W/s)),
                                                        indexing="ij"))
                                              for s in self.stride)],
                         -2)
        box1 = anchors - ltrb[..., :2]
        box2 = anchors + ltrb[..., 2:]
        if askeras.kwds.get("xywh"):
            box1, box2 = (box1 + box2) / 2, box2 - box1

        out = [box1, box2, sigmoid(Concatenate(-2)([Reshape((-1, self.nc)
                                                            )(cv3(xi))
                                                    for cv3, xi in
                                                    zip(self.cv3, x)]))]
        if self.type == "detect":
            return Concatenate(-1)(out)
        return out


class Pose(Torch_Pose, Detect):
    def __init__(self, nc, kpt_shape, ch, sm_split=None, junk=None):
        super().__init__(nc, kpt_shape, ch)
        Detect.__init__(self, nc, ch, sm_split)
        self.detect = Detect.forward
        self.type = "pose"
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = ModuleList(Sequential((Conv2d(x, c4, 3),
                                          ReLU(6),
                                          Conv2d(c4, c4, 3),
                                          ReLU(6),
                                          Conv2d(c4, self.nk, 1)))
                              for x in ch)

    def forward(self, *args, **kwds):
        if askeras.use_keras:
            return self.as_keras(*args, **kwds)
        return super().forward(*args, **kwds)

    def s(self, stride):
        if self.kpt_shape[1] == 3:
            from tensorflow import constant
            return constant([stride, stride, 1]*self.kpt_shape[0])
        return stride

    def as_keras(self, x):
        from tensorflow.keras.layers import Reshape
        from tensorflow import (meshgrid, range as trange, stack,
                                reshape, concat)
        from tensorflow.math import ceil
        from tensorflow.keras.layers import Concatenate
        from tensorflow.keras.activations import sigmoid
        presence_chans = [i*3+2 for i in range(17)]
        pres, coord = zip(*((Reshape((-1, self.kpt_shape[0], 1))(presence(xi)),
                             Reshape((-1, self.kpt_shape[0], 2))(coord(xi)*s*2))
                            for presence, coord, xi, s in
                            ((*cv[-1].split_channels(presence_chans),
                              cv[:-1](xi), s.item())
                             for cv, xi, s in zip(self.cv4, x, self.stride))))
        x = self.detect(self, x)
        H, W = askeras.kwds['imgsz']
        anchors = concat([stack((reshape(sx * s, (-1, 1)),
                                 reshape(sy * s, (-1, 1))),
                                -1)
                          for s, (sy, sx) in ((s.item(),
                                               meshgrid(trange(ceil(H/s)),
                                                        trange(ceil(W/s)),
                                                        indexing="ij"))
                                              for s in self.stride)],
                         -3)
        coord = Concatenate(-3)(coord) + anchors
        pres = sigmoid(Concatenate(-3)(pres))
        if askeras.kwds.get('test', False):
            return Concatenate(-1)(
                (*x, Reshape((-1, self.nk))(Concatenate(-1)((coord, pres))))
            )
        return *x, coord, pres


class Backend(BaseBackend):

    models = {}

    def get_model(self, model_path, full=False):
        if not isfile(model_path):
            model_path = join(dirname(__file__), "..", "zoo", "ultralytics",
                              model_path)
        if model_path in self.models:
            model = self.models[model_path]
        else:
            model = self.models[model_path] = YOLO(model_path)
        if full:
            return model
        return model.model

    def get_shape(self, model):
        if isinstance(model, str):
            model = self.get_model(model)
        return model.yaml["image_shape"]

    def patch(self):
        module = import_module("...layers", __name__)
        for name in dir(module):
            if name[0] != "_":
                setattr(tasks, name, getattr(module, name))
        tasks.Concat = module.Cat
        tasks.Detect = Detect
        tasks.Pose = Pose

    def val_post(self, weights, tflite, val_post, conf_thresh=.25,
                 iou_thresh=.7):
        """Default conf_thresh and iou_thresh taken from
        ultralytics/cfg/default.yaml."""

        print("processing", val_post)
        model = self.get_model(weights, full=True)
        num_kpts = model.model.model[-1].kpt_shape[0]

        print("tflite post processing output")
        tf_final = self.tf_post(tflite, val_post, conf_thresh, iou_thresh)
        print(tf_final)
        imwrite("tf_val.png",
                Results(orig_img=imread(val_post),
                        path=val_post,
                        names=model.names,
                        boxes=tf_final[:, :6],
                        keypoints=tf_final[:, 6:].reshape(-1, num_kpts, 3)
                        ).plot())

        print("pytorch post processing output")
        pt_final = self.pt_post(weights, val_post, conf_thresh, iou_thresh)
        print(pt_final)
        imwrite("pt_val.png",
                Results(orig_img=imread(val_post),
                        path=val_post,
                        names=model.names,
                        boxes=pt_final[:, :6],
                        keypoints=tf_final[:, 6:].reshape(-1, num_kpts, 3)
                        ).plot())

    def tf_post(self, tflite, val_post, conf_thresh, iou_thresh):

        # initialize tflite interpreter.
        from tensorflow import lite
        interpreter = lite.Interpreter(**{"model_path"
                                          if isinstance(tflite, str) else
                                          "model_content"
                                          : tflite})
        interpreter.allocate_tensors()
        in_scale, in_zero = interpreter.get_input_details()[0]['quantization']
        out_scale_zero_index = [(*detail['quantization'], detail['index'])
                                for detail in interpreter.get_output_details()]

        # make image RGB (not BGR) channel order, BCHW dimensions, and in the
        # range [0, 1].
        # cv2's imread reads in BGR channel order, with dimensions in Height,
        # Width, Channel order.  Also, imread keeps images as integers in
        # [0,255].  Normalize to floats in [0, 1].  Also, model expects a
        # batch dimension, so add a dimension at the beginning
        im = imread(val_post)[newaxis, ..., ::-1] / 255

        # run tflite on image
        assert interpreter.get_input_details()[0]['index'] == 0
        assert interpreter.get_input_details()[0]['dtype'] is int8
        interpreter.set_tensor(0, (im / in_scale + in_zero).astype(int8))
        interpreter.invoke()
        tout = [(interpreter.get_tensor(index)[0].astype(float32) - zero) * scale
                for scale, zero, index in out_scale_zero_index]
        b1, kpresence, kcoord, b2, bclass = tout

        # the number of keypoints and classes can be found from output shapes
        _, num_kpts, _ = kcoord.shape
        _, num_classes = bclass.shape

        # find the box class confidence and number.
        conf = npmax(bclass, axis=1, keepdims=True)
        class_num = argmax(bclass, axis=1, keepdims=True)

        # Combine results AFTER dequantization (so values in 0-1 and
        # 0-255 can be combined).
        preds = cat((b1, b2, conf, class_num, cat((kcoord, kpresence), -1
                                                  ).reshape(-1, num_kpts*3)),
                    axis=-1)

        # perform confidence thresholding, and convert to tensor for nms.
        preds = tensor(preds[preds[:, 4] > conf_thresh])

        # Perform NMS
        # https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html
        return preds[nms(preds[:, :4], preds[:, 4], iou_thresh)]

    def pt_post(self, weights, val_post, conf_thresh, iou_thresh):

        # obtain the pytorch model.
        model = self.get_model(weights, full=True)

        # Run the predictor.
        model(val_post)

        # The predictors saves the last batch to self.batch
        _, im0s, _, _ = model.predictor.batch

        # Rteurn the result of non-maximum supression -
        return non_max_suppression(
            # - on the input processed as in engine/predictor.py, -
            model.predictor.inference(model.predictor.preprocess(im0s)),
            # - with the parameters as used in models/yolo/pose/predict.py.
            model.predictor.args.conf,
            model.predictor.args.iou,
            agnostic=model.predictor.args.agnostic_nms,
            max_det=model.predictor.args.max_det,
            classes=model.predictor.args.classes,
            nc=len(model.predictor.model.names)
        # ignoring batch dimension (just one image)
        )[0]


def main():

    backend = Backend()

    # add synet ml modules to ultralytics
    backend.patch()

    # copy model from zoo if necessary
    for val in argv[1:]:
        if val.startswith("model="):
            model = val.split("=")[1]
            if in_zoo(model, "ultralytics"):
                src, model = get_config(model, "ultralytics"), basename(model)
                copy(src, model)
                argv.remove(val)
                argv.append("model="+model)
            break
    else:
        raise ValueError("no model specified")

    # add imgsz if not explicitly given
    for val in argv[1:]:
        if val.startswith("imgsz="):
            break
    else:
        argv.append(f"imgsz={max(backend.get_shape(model))}")

    # launch ultralytics
    from ultralytics.yolo.cfg import entrypoint
    entrypoint()
