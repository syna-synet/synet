"""layers.py is the high level model building layer of synet.  It defines useful composite layers which are compatible with multiple chips.  Because it is built with layers from base.py, exports come "free".  As a rule of thumb to differentiate between base.py, layers.py, and [chip].py:

- base.py should only import from torch, keras, and tensorflow.
- layers.py should only import from base.py.
- [chip].py should only import from base.py and layers.py."""
from typing import Union, Tuple, Optional

from torch import Tensor

from .base import (ReLU, BatchNorm, Conv2d, Module, Sequential,
                   DepthwiseConv2d)


class Conv2dInvertedResidual(Module):
    """Inverted Resudual blocks are the main building block of
MobileNet.  It is stable and gives low peek memory before and after.
Additionally, the computations are extremely efficient on our chips

    """
    def __init__(self, in_channels, expansion_factor,
                 out_channels=None, stride=1):
        """This inverted residual takes in_channels to in_channels*expansion_factor with a 3x3 convolution.  Then after a batchnorm and ReLU, the activations are taken back down to in_channels (or out_channels, if specified).  If out_channels is not specified (or equals in_channels), and the stride is 1, then the input will be added to the output before returning."""
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        hidden = int(in_channels * expansion_factor)
        self.layers = Sequential([
            Conv2d(in_channels,
                   out_channels = hidden,
                   kernel_size  = 3,
                   stride       = stride),
            BatchNorm(hidden),
            ReLU(6),
            Conv2d(in_channels  = hidden,
                   out_channels = out_channels,
                   kernel_size  = 1),
            BatchNorm(out_channels)
        ])
        self.stride = stride
        self.cheq = in_channels == out_channels # and isinstance(expansion_factor, int)
        assert self.stride in (1, 2)

    def forward(self, x):
        y = self.layers(x)
        if self.stride == 1 and self.cheq:
            return x + y
        if self.stride == 2 or (self.stride == 1 and not self.cheq):
            return y


class Head(Module):
    def __init__(self, in_channels, out_channels, num=4):
        """Creates a sequence of convolutions with ReLU(6)'s.
in_channels features are converted to out_channels in the first
convolution.  All other convolutions have out_channels going in and
out of that layer.  num (default 4) convolutions are used in total.

        """
        super().__init__()
        self.relu = ReLU(6)
        self.model = Sequential([
            Sequential([Conv2d(out_channels if i else in_channels,
                               out_channels,
                               3,
                               bias=True),
                        self.relu])
            for i in range(num)
        ])

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
        self.depthwise = DepthwiseConv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            groups=in_channels, bias=False,
            padding=padding)
        self.pointwise = Conv2d(in_channels, out_channels,
                                kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseInvertedResidual(Conv2dInvertedResidual):
    """Inverted Resudual blocks are the main building block of
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
        pointwise_expand = Conv2d(
            in_channels=in_channels, out_channels=hidden, kernel_size=1,
            padding=False)
        depthwise = DepthwiseConv2d(
            hidden, hidden, kernel_size=3, groups=hidden, bias=False)
        pointwise_squeeze = Conv2d(
            in_channels=hidden, out_channels=out_channels, kernel_size=1,
            padding=False)

        self.layers = Sequential([
            pointwise_expand,
            BatchNorm(hidden),
            ReLU(6),
            depthwise,
            ReLU(6),
            BatchNorm(hidden),
            pointwise_squeeze,
            BatchNorm(out_channels)
        ])


class DepthwiseSeparableInvertedResidual(Conv2dInvertedResidual):
    """
    Kind of efficient (in aspect of work memory) InvertedResidual block.
    """

    def __init__(self, in_channels, expansion_factor,
                 out_channels=None, stride=1):
        super().__init__(in_channels, expansion_factor, out_channels, stride)
        if out_channels is None:
            out_channels = in_channels
        hidden = int(in_channels * expansion_factor)
        self.layers = Sequential([
            DepthwiseSeparableConv2D(in_channels,
                                     out_channels=hidden,
                                     kernel_size=3,
                                     stride=stride),
            BatchNorm(hidden),
            ReLU(6),
            Conv2d(in_channels=hidden,
                   out_channels=out_channels,
                   kernel_size=1),
            BatchNorm(out_channels)
        ])
