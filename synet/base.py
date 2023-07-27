"""base.py is the "export" layer of synet.  As such, it includes the
logic of how to run as a keras model.  This is handled by cheking the
'askeras' context manager, and running in "keras mode" if that context
is enabled.  As a rule of thumb to differentiate between base.py,
layers.py, and [chip].py:

- base.py should only import from torch, keras, and tensorflow.
- layers.py should only import from base.py.
- [chip].py should only import from base.py and layers.py.
"""
import keras
from typing import Tuple, Union, Optional
from torch.nn import Module
from torch.nn import Conv2d as Torch_Conv2d
from torch.nn.functional import pad
from keras.layers import Conv2D as Keras_Conv2d


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


class Conv2d(Module):
    """Convolution operator which ensures padding is done equivalently
between PyTorch and TensorFlow.  Currently, only supports kernel_size
and stride combose specified in first assert in __init__.  If you add
more supported configurations, be sure to add those configurations to
test.py.

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
        assert (kernel_size, stride) in [(1, 1), (3, 1), (3, 2)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.conv = Torch_Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 bias=bias,
                                 stride=stride,
                                 groups=self.groups)
        self.use_bias = bias

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)

        # make padding like in tensorflow, which right aligns convolutions.
        if self.kernel_size == 3 and self.padding:
            if self.stride == 1:
                x = pad(x, (1,1,1,1))
            elif self.stride == 2:
                x = pad(x, (x.shape[-1]%2, 1, x.shape[-2]%2, 1))

        return self.conv(x)

    def as_keras(self, x):
        assert x.shape[-1] == self.in_channels, (x.shape, self.in_channels)
        padding_param = 'same' if self.padding else 'valid'
        conv = Keras_Conv2d(filters=self.out_channels,
                            kernel_size=self.kernel_size,
                            strides=self.stride,
                            padding=padding_param,
                            use_bias=self.use_bias,
                            groups=self.groups)
        conv.build(x.shape[1:])
        weight = self.conv.weight.detach().numpy().transpose(2, 3, 1, 0)
        conv.set_weights([weight, self.conv.bias.detach().numpy()]
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


class DepthwiseConv2d(Conv2d):
    def as_keras(self, x):
        assert x.shape[-1] == self.in_channels, (x.shape, self.in_channels)
        padding_param = 'same' if self.padding else 'valid'

        depthwise = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=padding_param,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.Constant(
                self.conv.weight.permute(2, 3, 1, 0).numpy())
        )

        return depthwise(x)


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


from torch import cat as torch_cat
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


from torch.nn import ReLU as Torch_ReLU
from torch import minimum, tensor
class ReLU(Module):
    def __init__(self, max_val=None):
        super().__init__()
        self.max_val = None if max_val is None else tensor(max_val,dtype=float)
        self.relu = Torch_ReLU()

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        if self.max_val is None:
            return self.relu(x)
        return minimum(self.relu(x), self.max_val)

    def as_keras(self, x):
        from keras.layers import ReLU as Keras_ReLU
        return Keras_ReLU(self.max_val)(x)


from torch.nn import BatchNorm2d as Torch_Batchnorm
class BatchNorm(Module):
    def __init__(self, features, epsilon=1e-3, momentum=0.999):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.batchnorm = Torch_Batchnorm(features, epsilon, momentum)

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        return self.batchnorm(x)

    def as_keras(self, x):
        from keras.layers import BatchNormalization as Keras_Batchnorm
        batchnorm = Keras_Batchnorm(momentum=self.momentum,
                                    epsilon=self.epsilon)
        batchnorm.build(x.shape)
        weights      = self.batchnorm.weight.detach().numpy()
        bias         = self.batchnorm.bias.detach().numpy()
        running_mean = self.batchnorm.running_mean.detach().numpy()
        running_var  = self.batchnorm.running_var.detach().numpy()
        batchnorm.set_weights([weights, bias, running_mean, running_var])
        return batchnorm(x, training=askeras.kwds["train"])


from torch.nn import ModuleList as Torch_Modulelist
class Sequential(Module):
    def __init__(self, sequence):
        super().__init__()
        self.ml = Torch_Modulelist(sequence)

    def forward(self, x):
        for layer in self.ml:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.ml[i]


