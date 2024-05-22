
from importlib import import_module
from sys import argv

from cv2 import imread, imwrite, resize
from numpy import array
from torch import tensor
from torch.nn import ModuleList
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, check_cls_dataset
from ultralytics.engine import validator, predictor
from ultralytics.engine.results import Results
from ultralytics.models.yolo import model as yolo_model
from ultralytics.nn import tasks
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.block import DFL as Torch_DFL, Proto as Torch_Proto
from ultralytics.nn.modules.head import (Pose as Torch_Pose,
                                         Detect as Torch_Detect,
                                         Segment as Torch_Segment,
                                         Classify as Torch_Classify)
from ultralytics.utils import dist
from ultralytics.utils.ops import non_max_suppression, process_mask
from ultralytics.utils.checks import check_imgsz

from . import Backend as BaseBackend
from ..base import (askeras, Conv2d, ReLU, Upsample, GlobalAvgPool,
                    Dropout, Linear)
from ..layers import Sequential, CoBNRLU
from ..tflite_utils import tf_run, concat_reshape


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


class Proto(Torch_Proto):
    def __init__(self, c1, c_=256, c2=32):
        """arguments understood as in_channels, number of protos, and
        number of masks"""
        super().__init__(c1, c_, c2)
        self.cv1 = CoBNRLU(c1, c_, 3)
        self.upsample = Upsample(scale_factor=2, mode='bilinear')
        self.cv2 = CoBNRLU(c_, c_, 3)
        self.cv3 = CoBNRLU(c_, c2, 1, name='proto')


def generate_anchors(H, W, stride, offset):
    from tensorflow import meshgrid, range, stack, reshape, concat
    from tensorflow.math import ceil
    return concat([stack((reshape((sx + offset) * s, (-1,)),
                          reshape((sy + offset) * s, (-1,))),
                         -1)
                   for s, (sy, sx) in ((s.item(),
                                        meshgrid(range(ceil(H/s)),
                                                 range(ceil(W/s)),
                                                 indexing="ij"))
                                       for s in stride)],
                  -2)


class Detect(Torch_Detect):
    def __init__(self, nc=80, ch=(), sm_split=None, junk=None):
        super().__init__(nc, ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        self.cv2 = ModuleList(Sequential(Conv2d(x, c2, 3, bias=True),
                                         ReLU(6),
                                         Conv2d(c2, c2, 3, bias=True),
                                         ReLU(6),
                                         Conv2d(c2, 4 * self.reg_max, 1,
                                                bias=True))
                              for x in ch)
        self.cv3 = ModuleList(Sequential(Conv2d(x, x, 3, bias=True),
                                         ReLU(6),
                                         Conv2d(x, x, 3, bias=True),
                                         ReLU(6),
                                         Conv2d(x, self.nc, 1, bias=True))
                              for x in ch)
        if junk is None:
            sm_split = None
        self.dfl = DFL(sm_split=sm_split)

    def forward(self, x):
        if askeras.use_keras:
            return Detect.as_keras(self, x)
        return super().forward(x)

    def as_keras(self, x):
        from tensorflow.keras.layers import Reshape
        from tensorflow import stack
        from tensorflow.keras.layers import (Concatenate, Subtract,
                                             Add, Activation)
        from tensorflow.keras.activations import sigmoid
        ltrb = Concatenate(-2)([self.dfl(cv2(xi)) * s.item()
                                for cv2, xi, s in
                                zip(self.cv2, x, self.stride)])
        H, W = askeras.kwds['imgsz']
        anchors = generate_anchors(H, W, self.stride, .5)             # Nx2
        anchors = stack([anchors for batch in range(x[0].shape[0])])  # BxNx2
        box1 = Subtract(name="box1")((anchors, ltrb[:, :, :2]))
        box2 = Add(name="box2")((anchors, ltrb[:, :, 2:]))
        if askeras.kwds.get("xywh"):
            box1, box2 = (box1 + box2) / 2, box2 - box1

        cls = Activation(sigmoid, name='cls')(
            Concatenate(-2)([
                Reshape((-1, self.nc))(cv3(xi))
                for cv3, xi in zip(self.cv3, x)
            ])
        )
        out = [box1, box2, cls]
        if askeras.kwds.get("quant_export"):
            return out
        # everything after here needs to be implemented by post-processing
        out[:2] = (box/array((W, H)) for box in out[:2])
        return Concatenate(-1)(out)


class Pose(Torch_Pose, Detect):
    def __init__(self, nc, kpt_shape, ch, sm_split=None, junk=None):
        super().__init__(nc, kpt_shape, ch)
        Detect.__init__(self, nc, ch, sm_split, junk=junk)
        self.detect = Detect.forward
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = ModuleList(Sequential(Conv2d(x, c4, 3),
                                         ReLU(6),
                                         Conv2d(c4, c4, 3),
                                         ReLU(6),
                                         Conv2d(c4, self.nk, 1))
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

        from tensorflow.keras.layers import Reshape, Concatenate, Add
        from tensorflow import stack, reshape
        from tensorflow.keras.activations import sigmoid

        if self.kpt_shape[1] == 3:
            presence_chans = [i*3+2 for i in range(17)]
            pres, kpts = zip(*((Reshape((-1, self.kpt_shape[0], 1)
                                        )(presence(xi)),
                                Reshape((-1, self.kpt_shape[0], 2)
                                        )(keypoint(xi)*s*2))
                               for presence, keypoint, xi, s in
                               ((*cv[-1].split_channels(presence_chans),
                                 cv[:-1](xi), s.item())
                                for cv, xi, s in
                                zip(self.cv4, x, self.stride))))
            pres = Concatenate(-3, name="pres")([sigmoid(p) for p in pres])
        else:
            kpts = [Reshape((-1, self.kpt_shape[0], 2))(cv(xi)*s*2)
                    for cv, xi, s in
                    zip(self.cv4, x, self.stride)]

        H, W = askeras.kwds['imgsz']
        anchors = generate_anchors(H, W, self.stride, offset=0)       # Nx2
        anchors = reshape(anchors, (-1, 1, 2))                        # Nx1x2
        anchors = stack([anchors for batch in range(x[0].shape[0])])  # BxNx1x2
        kpts = Add(name='kpts')((Concatenate(-3)(kpts), anchors))

        x = self.detect(self, x)

        if askeras.kwds.get("quant_export"):
            if self.kpt_shape[1] == 3:
                return *x, kpts, pres
            return *x, kpts

        # everything after here needs to be implemented by post-processing
        if self.kpt_shape[1] == 3:
            kpts = Concatenate(-1)((kpts, pres))

        return Concatenate(-1)((x, Reshape((-1, self.nk))(kpts)))


class Segment(Torch_Segment, Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=(), sm_split=None, junk=None):
        super().__init__(nc, nm, npr, ch)
        Detect.__init__(self, nc, ch, sm_split, junk=junk)
        self.detect = Detect.forward
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = ModuleList(Sequential(CoBNRLU(x, c4, 3),
                                         CoBNRLU(c4, c4, 3),
                                         Conv2d(c4, self.nm, 1))
                              for x in ch)

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        return super().forward(x)

    def as_keras(self, x):
        from tensorflow.keras.layers import Reshape, Concatenate
        p = self.proto(x[0])
        mc = Concatenate(-2, name='seg')([Reshape((-1, self.nm))(cv4(xi))
                                          for cv4, xi in zip(self.cv4, x)])
        x = self.detect(self, x)
        if askeras.kwds.get("quant_export"):
            return *x, mc, p
        # everything after here needs to be implemented by post-processing
        return Concatenate(-1)((x, mc)), p


class Classify(Torch_Classify):
    def __init__(self, junk, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__(c1, c2, k=k, s=s, p=p, g=g)
        c_ = 1280
        assert p is None
        self.conv = CoBNRLU(c1, c_, k, s, groups=g)
        self.pool = GlobalAvgPool()
        self.drop = Dropout(p=0.0, inplace=True)
        self.linear = Linear(c_, c2)

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        return super().forward(x)

    def as_keras(self, x):
        from keras.layers import Concatenate, Flatten, Softmax
        if isinstance(x, list):
            x = Concatenate(-1)(x)
        x = self.linear(self.drop(Flatten()(self.pool(self.conv(x)))))
        return x if self.training else Softmax()(x)


class Backend(BaseBackend):

    models = {}
    name = "ultralytics"

    def get_model(self, model_path, full=False):

        model_path = self.maybe_grab_from_zoo(model_path)

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

    def patch(self, model_path=None):
        module = import_module("...layers", __name__)
        for name in dir(module):
            if name[0] != "_":
                setattr(tasks, name, getattr(module, name))
        tasks.Concat = module.Cat
        tasks.Pose = Pose
        tasks.Detect = Detect
        tasks.Segment = Segment
        tasks.Classify = Classify
        orig_ddp_file = dist.generate_ddp_file

        def generate_ddp_file(trainer):
            fname = orig_ddp_file(trainer)
            fstr = open(fname).read()
            open(fname, 'w').write(f"""\
from synet.backends import get_backend
get_backend('ultralytics').patch()
{fstr}""")
            return fname
        dist.generate_ddp_file = generate_ddp_file
        if model_path is not None and model_path.endswith('tflite'):
            print('SyNet: model provided is tflite.  Modifying validators'
                  ' to anticipate tflite output')
            task_map = yolo_model.YOLO(model_path).task_map
            for task in task_map:
                for mode in 'predictor', 'validator':
                    class Wrap(task_map[task][mode]):
                        def postprocess(self, preds, *args, **kwds):
                            # concate_reshape currently expect ndarry
                            # with batch size of 1, so remove and
                            # re-add batch and tensorship.
                            preds = concat_reshape([p[0].numpy()
                                                    for p in preds],
                                                   self.args.task,
                                                   classes_to_index=False,
                                                   xywh=True)
                            if isinstance(preds, tuple):
                                preds = (tensor(preds[0][None])
                                         .permute(0, 2, 1),
                                         tensor(preds[1][None]))
                            else:
                                preds = tensor(preds[None]).permute(0, 2, 1)
                            return super().postprocess(preds, *args, **kwds)
                    if task != 'classify':
                        task_map[task][mode] = Wrap
            yolo_model.YOLO.task_map = task_map

            def tflite_check_imgsz(*args, **kwds):
                kwds['stride'] = 1
                return check_imgsz(*args, **kwds)

            class TfliteAutoBackend(AutoBackend):
                def __init__(self, *args, **kwds):
                    super().__init__(*args, **kwds)
                    self.output_details.sort(key=lambda x: x['name'])
                    if len(self.output_details) == 1:  # classify
                        num_classes = self.output_details[0]['shape'][-1]
                    else:
                        num_classes = self.output_details[2]['shape'][2]
                    self.kpt_shape = (self.output_details[-1]['shape'][-2], 3)
                    self.names = {k: self.names[k] for k in range(num_classes)}

            validator.check_imgsz = tflite_check_imgsz
            predictor.check_imgsz = tflite_check_imgsz
            validator.AutoBackend = TfliteAutoBackend
            predictor.AutoBackend = TfliteAutoBackend

    def val_post(self, weights, tflite, val_post, conf_thresh=.25,
                 iou_thresh=.7, image_shape=None):
        """Default conf_thresh and iou_thresh (.25 and .75 resp.)
        taken from ultralytics/cfg/default.yaml.

        """

        # load model and image
        print("processing", val_post, "with", weights)
        model = self.get_model(weights, full=True)
        img = imread(val_post)
        if image_shape is not None:
            img = resize(img, (image_shape[::-1]))

        # run th tf model, save plot, and print the values
        print("tflite post processing")
        tf_final = tf_run(tflite, img, conf_thresh, iou_thresh, "ultralytics",
                          task=model.task)
        self.gen_visualization(tf_final, img, val_post, model, "tf_val.png",
                               print_arrs=True)

        # run the pt model, save plot, and print the values
        print("pytorch post processing output")
        pt_final = self.pt_run(model, val_post, conf_thresh, iou_thresh)
        self.gen_visualization(pt_final, img, val_post, model, "pt_val.png",
                               print_arrs=True)

    def gen_visualization(self, model_output, img, img_path, model, out_file,
                          print_arrs=True):

        # interpret model output format
        if isinstance(model_output, tuple):
            pred, proto = model_output
        else:
            pred = model_output

        # optionally, print arrays
        if print_arrs:
            if model.task == 'segment':
                print("proto, every 10^2 pixel, one channel:")
                print("proto shape,", proto.shape)
                print(proto[::10, ::10, 0])
            print("preds after NMS:")
            print(pred)

        # add task-specific options
        kwds = dict()
        if model.task != "classify":
            kwds['boxes'] = pred[:, :6]
        if model.task == "pose":
            kpt_shape = model.model.model[-1].kpt_shape
            kwds['keypoints'] = pred[:, 6:].reshape(-1, *kpt_shape)
        if model.task == "segment":
            kwds['masks'] = process_mask(proto, pred[:, 6:], pred[:, :4],
                                         img.shape[:2], upsample=True)

        # return image
        imwrite(out_file, Results(orig_img=img, path=img_path,
                                  names=model.names, **kwds
                                  ).plot())

    def pt_run(self, model, val_post, conf_thresh, iou_thresh):

        # Run the predictor.
        model(val_post)

        # The predictors saves the last batch to self.batch
        _, im0s, _, _ = model.predictor.batch

        # obtain and interpret output of model when run on the image
        pred = model.predictor.inference(model.predictor.preprocess(im0s))
        if model.task == 'segment':
            pred, proto = pred
            # treating as in
            # ultralytics/ultralytics/models/yolo/segment/predict.py:
            # second output is len 3 if pt, but only 1 if exported
            if len(proto) == 3:
                proto = proto[-1]
            proto = proto[0]

        # Rteurn the result of non-maximum supression -
        preds = non_max_suppression(
            # - on the input processed as in engine/predictor.py, -
            pred,
            # - with the parameters as used in models/yolo/pose/predict.py -
            model.predictor.args.conf,
            model.predictor.args.iou,
            agnostic=model.predictor.args.agnostic_nms,
            max_det=model.predictor.args.max_det,
            classes=model.predictor.args.classes,
            nc=len(model.predictor.model.names)
        )[0]  # - ignoring batch dimension (just one image)
        if model.task == 'segment':
            return preds, proto
        return preds

    def get_data(self, data):
        try:
            return check_det_dataset(data)
        except Exception as e:
            try:
                return check_cls_dataset(data)
            except Exception as e2:
                print("unable to load data as classification or detection dataset")
                print(e2)
                raise e


def main():

    backend = Backend()

    # copy model from zoo if necessary
    for ind, val in enumerate(argv):
        if val.startswith("model="):
            model = backend.maybe_grab_from_zoo(val.split("=")[1])
            argv[ind] = "model="+model

            # add synet ml modules to ultralytics
            backend.patch(model_path=model)

            # add imgsz if not explicitly given
            for val in argv:
                if val.startswith("imgsz="):
                    break
            else:
                argv.append(f"imgsz={max(backend.get_shape(model))}")

            break


    # launch ultralytics
    try:
        from ultralytics.cfg import entrypoint
    except:
        from ultralytics.yolo.cfg import entrypoint
    entrypoint()
