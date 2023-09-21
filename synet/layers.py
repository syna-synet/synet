"""layers.py is the high level model building layer of synet.  It
defines useful composite layers which are compatible with multiple
chips.  Because it is built with layers from base.py, exports come
"free".  As a rule of thumb to differentiate between base.py,
layers.py:

- base.py should only import from torch, keras, and tensorflow.
- layers.py should only import from base.py.
"""
from typing import Union, Tuple, Optional

from torch import Tensor

from .base import (ReLU, BatchNorm, Conv2d, Module, Cat, Grayscale,
                   Sequential)


class DepthwiseConv2d(Conv2d):
    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 bias: bool = False,
                 padding: Optional[bool] = True):
        super().__init__(channels, channels, kernel_size, stride,
                         bias, padding, groups=channels)


class Conv2dInvertedResidual(Module):
    """
    Block of conv2D -> activation -> linear pointwise with residual concat.
    Inspired by Inverted Residual blocks which are the main building block
    of MobileNet.  It is stable and gives low peek memory before and after.
    Additionally, the computations are extremely efficient on our chips
    """

    def __init__(self, in_channels, expansion_factor,
                 out_channels=None, stride=1):
        """This inverted residual takes in_channels to
        in_channels*expansion_factor with a 3x3 convolution.  Then
        after a batchnorm and ReLU, the activations are taken back
        down to in_channels (or out_channels, if specified).  If
        out_channels is not specified (or equals in_channels), and the
        stride is 1, then the input will be added to the output before
        returning."""
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        hidden = int(in_channels * expansion_factor)
        self.layers = Sequential(Conv2d(in_channels,
                                        out_channels=hidden,
                                        kernel_size=3, stride=stride),
                                 BatchNorm(hidden),
                                 ReLU(6),
                                 Conv2d(in_channels=hidden,
                                        out_channels=out_channels,
                                        kernel_size=1),
                                 BatchNorm(out_channels))
        self.stride = stride
        self.cheq = in_channels == out_channels
        assert self.stride in (1, 2)

    def forward(self, x):
        y = self.layers(x)
        if self.stride == 1 and self.cheq:
            return x + y
        if self.stride == 2 or (self.stride == 1 and not self.cheq):
            return y


# for backwards compatibility
InvertedResidual = Conv2dInvertedResidual


class Head(Module):
    def __init__(self, in_channels, out_channels, num=4):
        """Creates a sequence of convolutions with ReLU(6)'s.
in_channels features are converted to out_channels in the first
convolution.  All other convolutions have out_channels going in and
out of that layer.  num (default 4) convolutions are used in total.

        """
        super().__init__()
        self.relu = ReLU(6)
        out_channels = [in_channels] * (num - 1) + [out_channels]
        self.model = Sequential(*(Sequential(Conv2d(in_channels,
                                                    out_channels,
                                                    3, bias=True),
                                             self.relu)
                                  for out_channels in out_channels)
        )

    def forward(self, x):
        return self.model(x)


class DepthwiseSeparableConv2D(Module):
    """
    Depthwise Separable convolution using pytorch
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: Optional[bool] = True,
                 bias: bool = False):
        """

        :param in_channels: Number of channels in the input
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param conv_2d_params: Additional arguments passed to nn.Conv2d
        """
        super().__init__()
        self.depthwise = DepthwiseConv2d(channels=in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         bias=False,
                                         padding=padding)
        self.pointwise = Conv2d(in_channels, out_channels,
                                kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseInvertedResidual(Conv2dInvertedResidual):
    """Classic Inverted Residual blocks are the main building block of
        MobileNet.  It is stable and gives low peek memory before and after.
        Additionally, the computations are extremely efficient on our chips

    """

    def __init__(self, in_channels, expansion_factor,
                 out_channels=None, stride=1):
        """
        This inverted residual takes in_channels to
         in_channels*expansion_factor with a 3x3 convolution.
         Then after a batchnorm and ReLU, the activations are taken back down
          to in_channels (or out_channels, if specified).  If out_channels is
          not specified (or equals in_channels), and the stride is 1, then the
          input will be added to the output before returning.
        """
        super().__init__(in_channels, expansion_factor, out_channels, stride)
        if out_channels is None:
            out_channels = in_channels
        hidden = int(in_channels * expansion_factor)
        self.layers = Sequential(Conv2d(in_channels=in_channels,
                                        out_channels=hidden,
                                        kernel_size=1,
                                        padding=False),
                                 BatchNorm(hidden),
                                 ReLU(6),
                                 DepthwiseConv2d(channels=hidden,
                                                 kernel_size=3,
                                                 bias=False),
                                 ReLU(6),
                                 BatchNorm(hidden),
                                 Conv2d(in_channels=hidden,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        padding=False),
                                 BatchNorm(out_channels))


class DepthwiseSeparableInvertedResidual(Conv2dInvertedResidual):
    """
    Kind of efficient (in aspect of work memory) InvertedResidual block.
    Same as "Conv2dInvertedResidual" but with depthwise separable instead of
    Conv2D
    """

    def __init__(self, in_channels, expansion_factor,
                 out_channels=None, stride=1):
        super().__init__(in_channels, expansion_factor, out_channels, stride)
        if out_channels is None:
            out_channels = in_channels
        hidden = int(in_channels * expansion_factor)
        self.layers = Sequential(DepthwiseSeparableConv2D(in_channels,
                                                          out_channels=hidden,
                                                          kernel_size=3,
                                                          stride=stride),
                                 BatchNorm(hidden),
                                 ReLU(6),
                                 Conv2d(in_channels=hidden,
                                        out_channels=out_channels,
                                        kernel_size=1),
                                 BatchNorm(out_channels))


class CoBNRLU(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, bias=False, groups=1, max_val=6):
        super().__init__()
        self.module = Sequential(Conv2d(in_channels, out_channels,
                                        kernel_size, stride, bias, groups),
                                 BatchNorm(out_channels),
                                 ReLU(max_val))

    def forward(self, x):
        return self.module(x)
