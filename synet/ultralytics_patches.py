from importlib import import_module

from torch.nn import ModuleList
from ultralytics import YOLO
from ultralytics.nn import tasks
from ultralytics.nn.modules.block import DFL as Torch_DFL
from ultralytics.nn.modules.head import Pose as Torch_Pose, Detect as Torch_Detect

from .base import askeras, Conv2d, ReLU
from .layers import Sequential


class DFL(Torch_DFL):
    def __init__(self, c1=16, sm_split=None):
        super().__init__(c1)
        weight = self.conv.weight
        self.conv = Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.conv.weight.data[:] = weight.data
        if isinstance(sm_split, int):
            self.sm_split = sm_split

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        return super().forward(x)

    def as_keras(self, x):
        # b, ay, ax, c = x.shape
        from tensorflow.keras.layers import Reshape, Softmax
        if hasattr(self, 'sm_split'):
            from tensorflow.keras.layers import Concatenate
            assert not (x.shape[0]*x.shape[1]*x.shape[2]) % self.sm_split
            x = Reshape((self.sm_split, -1, 4, self.c1))(x)
            # tensorflow really wants to be indented like this.  I relent...
            return Reshape((-1, 4))(
                self.conv(
                    Concatenate(1)([
                        Softmax(-1)(x[:, i])
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

        return Concatenate(-1)([box1, box2,
                                sigmoid(Concatenate(-2)([Reshape((-1, self.nc)
                                                                 )(cv3(xi))
                                                         for cv3, xi in
                                                         zip(self.cv3, x)]))])


class Pose(Torch_Pose, Detect):
    def __init__(self, nc, kpt_shape, ch, sm_split=None, junk=None):
        super().__init__(nc, kpt_shape, ch)
        Detect.__init__(self, nc, ch, sm_split)
        self.detect = Detect.forward
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
        from tensorflow import meshgrid, range, stack, reshape, concat
        from tensorflow.math import ceil
        from tensorflow.keras.layers import Concatenate
        from tensorflow.keras.activations import sigmoid
        kpt = Concatenate(-3)([Reshape((-1, *self.kpt_shape))(cv(xi) *
                                                              self.s(s.item()))
                               for cv, xi, s in zip(self.cv4, x, self.stride)])
        x = self.detect(self, x)
        H, W = askeras.kwds['imgsz']
        anchors = concat([stack((reshape(sx * s, (-1, 1)),
                                 reshape(sy * s, (-1, 1))),
                                -1)
                          for s, (sy, sx) in ((s.item(),
                                               meshgrid(range(ceil(H/s)),
                                                        range(ceil(W/s)),
                                                        indexing="ij"))
                                              for s in self.stride)],
                         -3)
        kpt = Reshape((-1, self.nk))(Concatenate(-1)([kpt[..., :2] * 2
                                                      + anchors,
                                                      sigmoid(kpt[..., 2:])]))
        return Concatenate(-1)([x, kpt])


def get_ultralytics_model(model_path, low_thld=0, raw=False, **kwds):
    if model_path.endswith(".yml") or model_path.endswith(".yaml"):
        assert raw
        return YOLO(model_path).model
    model = YOLO(model_path)
    if raw:
        return model.model
    return model


def patch_ultralytics(chip=None):

    # enable the  chip if given
    if chip is not None:
        module = import_module(f"..{chip}", __name__)
        for name in dir(module):
            if name[0] != "_":
                setattr(tasks, name, getattr(module, name))
        tasks.Concat = module.Cat

    tasks.Detect = Detect
    tasks.Pose = Pose

    import synet
    synet.get_model_backend = get_ultralytics_model
