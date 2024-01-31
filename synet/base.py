"""base.py is the "export" layer of synet.  As such, it includes the
logic of how to run as a keras model.  This is handled by cheking the
'askeras' context manager, and running in "keras mode" if that context
is enabled.  As a rule of thumb to differentiate between base.py,
layers.py:

- base.py should only import from torch, keras, and tensorflow.
- layers.py should only import from base.py.
"""

from typing import Tuple, Union, Optional

from torch import cat as torch_cat, minimum, tensor, no_grad
from torch.nn import (Module as Torch_Module,
                      Conv2d as Torch_Conv2d,
                      BatchNorm2d as Torch_Batchnorm,
                      ModuleList as Torch_Modulelist,
                      ReLU as Torch_ReLU,
                      ConvTranspose2d as Torch_ConvTranspose2d,
                      Upsample as Torch_Upsample,
                      AdaptiveAvgPool2d as Torch_AdaptiveAvgPool,
                      Dropout as Torch_Dropout,
                      Linear as Torch_Linear)
from torch.nn.functional import pad


class AsKeras:
    """AsKeras is a context manager used to export from pytorch to
keras.  See test.py and quantize.py for examples.

    """

    def __init__(self):
        self.use_keras = False
        self.kwds = dict(train=False)

    def __call__(self, **kwds):
        self.kwds.update(kwds)
        return self

    def __enter__(self):
        self.use_keras = True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__init__()


askeras = AsKeras()


class Module(Torch_Module):
    def forward(self, x):
        if askeras.use_keras and hasattr(self, 'as_keras'):
            return self.as_keras(x)
        return self.module(x)


class Conv2d(Module):
    """Convolution operator which ensures padding is done equivalently
    between PyTorch and TensorFlow.

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 bias: bool = False,
                 padding: Optional[bool] = True,
                 groups: Optional[int] = 1):
        """
        Implementation of torch Conv2D with option fot supporting keras
            inference
        :param in_channels: Number of channels in the input
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the kernel
        :param stride:
        :param bias:
        :param groups: using for pointwise/depthwise
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = "same" if padding else 'valid'
        self.groups = groups
        self.conv = Torch_Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 bias=bias,
                                 stride=stride,
                                 groups=self.groups)
        self.use_bias = bias

    def forward(self, x):

        # temporary code for backwards compatibility
        if not isinstance(self.padding, str):
            self.padding = "same" if self.padding else 'valid'

        if askeras.use_keras:
            return self.as_keras(x)

        if self.padding == "valid":
            return self.conv(x)

        # make padding like in tensorflow, which right aligns convolutionn.
        H, W = (s if isinstance(s, int) else s.item() for s in x.shape[-2:])
        # radius of the kernel and carry.  Border size + carry.  All in y
        ry, rcy = divmod(self.kernel_size[0]-1, 2)
        by, bcy = divmod((H-1) % self.stride - rcy, 2)
        # radius of the kernel and carry.  Border size + carry.  All in x
        rx, rcx = divmod(self.kernel_size[1]-1, 2)
        bx, bcx = divmod((W-1) % self.stride - rcx, 2)
        # apply pad
        return self.conv(
            pad(x, (rx - bx - bcx, rx - bx, ry - by - bcy, ry - by)))

    def as_keras(self, x):
        if askeras.kwds.get('demosaic'):
            from .demosaic import Demosaic, reshape_conv
            demosaic = Demosaic(*askeras.kwds['demosaic'].split('-'))
            del askeras.kwds['demosaic']
            return reshape_conv(self)(demosaic(x))
        from keras.layers import Conv2D as Keras_Conv2d
        assert x.shape[-1] == self.in_channels, (x.shape, self.in_channels)
        conv = Keras_Conv2d(filters=self.out_channels,
                            kernel_size=self.kernel_size,
                            strides=self.stride,
                            padding=self.padding,
                            use_bias=self.use_bias,
                            groups=self.groups)
        conv.build(x.shape[1:])
        if isinstance(self.conv, Torch_Conv2d):
            tconv = self.conv
        else:
            # for NNI compatibility
            tconv = self.conv.module
        weight = tconv.weight.detach().numpy().transpose(2, 3, 1, 0)
        conv.set_weights([weight, tconv.bias.detach().numpy()]
                         if self.use_bias else
                         [weight])
        return conv(x)

    def requires_grad_(self, val):
        self.conv = self.conv.requires_grad_(val)
        return self

    def __getattr__(self, name):
        if name in ("bias", "weight"):
            return getattr(self.conv, name)
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in ("bias", "weight"):
            return setattr(self.conv, name, value)
        return super().__setattr__(name, value)

    def split_channels(self, chans):

        with no_grad():
            split = Conv2d(self.in_channels, len(chans),
                           self.kernel_size, self.stride, self.use_bias)
            split.weight[:] = self.weight[chans]

            rest_chans = [i for i in range(self.out_channels) if i not in chans]
            rest = Conv2d(self.in_channels, self.out_channels - len(chans),
                          self.kernel_size, self.stride, self.use_bias)
            rest.weight[:] = self.weight[rest_chans]

            if self.use_bias:
                split.bias[:] = self.bias[chans]
                rest.bias[:] = self.bias[rest_chans]

        return split, rest


# don't try to move this assignment into class def.  It won't work.
# This is for compatibility with NNI so it does not treat this like a
# pytorch conv2d, and instead finds the nested conv2d.
Conv2d.__name__ = "Synet_Conv2d"


class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=False):
        print("WARNING: synet ConvTranspose2d mostly untested")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = "valid" if padding == 0 else "same"
        self.use_bias = bias
        self.module = Torch_ConvTranspose2d(in_channels, out_channels,
                                          kernel_size, stride,
                                          padding, bias=bias)

    def as_keras(self, x):
        from keras.layers import Conv2DTranspose as Keras_ConvTrans
        conv = Keras_ConvTrans(self.out_channels, self.kernel_size, self.stride,
                               self.padding, use_bias=self.use_bias)
        conv.build(x.shape)
        if isinstance(self.module, Torch_ConvTranspose2d):
            tconv = self.module
        else:
            # for NNI compatibility
            tconv = self.module.module
        weight = tconv.weight.detach().numpy().transpose(2, 3, 1, 0)
        conv.set_weights([weight, tconv.bias.detach().numpy()]
                         if self.use_bias else
                         [weight])
        return conv(x)


class Grayscale(Module):
    """Training frameworks often fix input channels to 3.  This
grayscale layer can be added to the beginning of a model to convert to
grayscale.  This layer is ignored when converting to tflite.  The end
result is that the pytorch model can take any number of input
channels, but the tensorflow (tflite) model expects exactly one input
channel.

    """

    def forward(self, x):
        if askeras.use_keras:
            return x
        return x.mean(1, keepdims=True)


class Cat(Module):
    """Concatenate along feature dimension."""

    def __init__(self, *args):
        super().__init__()

    def forward(self, xs):
        if askeras.use_keras:
            return self.as_keras(xs)
        return torch_cat(xs, dim=1)

    def as_keras(self, xs):
        assert all(len(x.shape) == 4 for x in xs)
        from keras.layers import Concatenate as Keras_Concatenate
        return Keras_Concatenate(-1)(xs)


class ReLU(Module):
    def __init__(self, max_val=None, name=None):
        super().__init__()
        self.max_val = None if max_val is None else tensor(max_val,
                                                           dtype=float)
        self.name = name
        self.relu = Torch_ReLU()

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        if self.max_val is None:
            return self.relu(x)
        return minimum(self.relu(x), self.max_val)

    def as_keras(self, x):
        from keras.layers import ReLU as Keras_ReLU
        return Keras_ReLU(self.max_val, name=self.name)(x)


class BatchNorm(Module):
    def __init__(self, features, epsilon=1e-3, momentum=0.999):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.module = Torch_Batchnorm(features, epsilon, momentum)

    def as_keras(self, x):
        from keras.layers import BatchNormalization as Keras_Batchnorm
        batchnorm = Keras_Batchnorm(momentum=self.momentum,
                                    epsilon=self.epsilon)
        batchnorm.build(x.shape)
        if isinstance(self.module, Torch_Batchnorm):
            bn = self.module
        else:
            bn = self.module.module
        weights = bn.weight.detach().numpy()
        bias = bn.bias.detach().numpy()
        running_mean = bn.running_mean.detach().numpy()
        running_var = bn.running_var.detach().numpy()
        batchnorm.set_weights([weights, bias, running_mean, running_var])
        return batchnorm(x, training=askeras.kwds["train"])


class Upsample(Module):
    allowed_modes = "bilinear", "nearest"

    def __init__(self, scale_factor, mode="nearest"):
        assert mode in self.allowed_modes
        if not isinstance(scale_factor, int):
            for sf in scale_factor:
                assert isinstance(sf, int)
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.module = Torch_Upsample(scale_factor=scale_factor, mode=mode)

    def as_keras(self, x):
        from keras.layers import UpSampling2D
        return UpSampling2D(size=self.scale_factor,
                            interpolation=self.mode,
                            )(x)


class Sequential(Module):
    def __init__(self, *sequence):
        super().__init__()
        self.ml = Torch_Modulelist(sequence)

    def forward(self, x):
        for layer in self.ml:
            x = layer(x)
        return x

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.ml[i]
        return Sequential(*self.ml[i])


class GlobalAvgPool(Module):
    def __init__(self):
        super().__init__()
        self.module = Torch_AdaptiveAvgPool(1)
    def as_keras(self, x):
        from keras.layers import GlobalAveragePooling2D
        return GlobalAveragePooling2D(keepdims=True)(x)


class Dropout(Module):
    def __init__(self, p=0, inplace=False):
        super().__init__()
        self.p = p
        self.module = Torch_Dropout(p, inplace=inplace)
    def as_keras(self, x):
        from keras.layers import Dropout
        return Dropout(self.p)(x, training=askeras.kwds["train"])


class Linear(Module):
    def __init__(self, in_c, out_c, bias=True):
        super().__init__()
        self.use_bias = bias
        self.module = Torch_Linear(in_c, out_c, bias)
    def as_keras(self, x):
        from keras.layers import Dense
        out_c, in_c = self.module.weight.shape
        params = [self.module.weight.detach().numpy().transpose(1, 0 )]
        if self.use_bias:
            params.append(self.module.bias.detach().numpy())
        dense = Dense(out_c, use_bias=self.use_bias)
        dense.build(x.shape[1:])
        dense.set_weights(params)
        return dense(x)
