"""layers.py is the high level model building layer of synet.  It
defines useful composite layers which are compatible with multiple
chips.  Because it is built with layers from base.py, exports come
"free".  As a rule of thumb to differentiate between base.py,
layers.py:

- base.py should only import from torch, keras, and tensorflow.
- layers.py should only import from base.py.

If you sublcass from something in base.py OTHER than Module, you
should add a test case for it in tests/test_keras.py.

"""
from typing import Union, Tuple, Optional
import torch
from torch.nn import ModuleList

from .base import (ReLU, BatchNorm, Conv2d, Module, Cat, Grayscale,
                   Sequential, RNN, GRU, LSTM, Transpose, Reshape, Flip, Add,
                   Shape, ModuleList, ChannelSlice)


# because this module only reinterprets Conv2d parameters, the test
# case is omitted.
class DepthwiseConv2d(Conv2d):
    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 bias: bool = False,
                 padding: Optional[bool] = True):
        super().__init__(channels, channels, kernel_size, stride,
                         bias, padding, groups=channels)


class InvertedResidual(Module):
    """
    Block of conv2D -> activation -> linear pointwise with residual concat.
    Inspired by Inverted Residual blocks which are the main building block
    of MobileNet.  It is stable and gives low peek memory before and after.
    Additionally, the computations are extremely efficient on our chips
    """

    def __init__(self, in_channels, expansion_factor,
                 out_channels=None, stride=1, kernel_size=3,
                 skip=True):
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
                                        kernel_size=kernel_size,
                                        stride=stride),
                                 BatchNorm(hidden),
                                 ReLU(6),
                                 Conv2d(in_channels=hidden,
                                        out_channels=out_channels,
                                        kernel_size=1),
                                 BatchNorm(out_channels))
        self.stride = stride
        self.cheq = in_channels == out_channels and skip
        assert self.stride in (1, 2)

    def forward(self, x):
        y = self.layers(x)
        if self.stride == 1 and self.cheq:
            return x + y
        if self.stride == 2 or (self.stride == 1 and not self.cheq):
            return y


# for backwards compatibility
Conv2dInvertedResidual = InvertedResidual


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
                                  for out_channels in out_channels))

    def forward(self, x):
        return self.model(x)


class CoBNRLU(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, bias=False, padding=True, groups=1,
                 max_val=6, name=None):
        super().__init__()
        self.module = Sequential(Conv2d(in_channels, out_channels,
                                        kernel_size, stride, bias, padding,
                                        groups),
                                 BatchNorm(out_channels),
                                 ReLU(max_val, name=name))

    def forward(self, x):
        return self.module(x)


class GenericSRNN(Module):
    """
    Implements GenericSRNN (Generic Separable RNN), which processes an input tensor sequentially
    along its X-axis and Y-axis using two RNNs.
    This approach first applies an RNN along the X-axis of the input, then feeds the resulting tensor
    into another RNN along the Y-axis.

    Args:
        - hidden_size_x (int): The number of features in the hidden state of the X-axis RNN.
        - hidden_size_y (int): The number of features in the hidden state of the Y-axis RNN,
          which also determines the output size.
        - num_layers (int, optional): Number of recurrent layers for each RNN, defaulting to 1.

    Returns:
        - output (tensor): The output tensor from the Y-axis RNN.
        - hn_x (tensor): The final hidden state from the X-axis RNN.
        - hn_y (tensor): The final hidden state from the Y-axis RNN.
    """

    def __init__(self, hidden_size_x: int, hidden_size_y: int) -> None:
        super(GenericSRNN, self).__init__()
        self.output_size_x = hidden_size_x
        self.output_size_y = hidden_size_y

        self.transpose = Transpose()
        self.reshape = Reshape()
        self.get_shape = Shape()

    def forward(self, x, rnn_x: Module, rnn_y: Module):
        """
        Performs the forward pass for the HierarchicalRNN module.

        Args:
            - x (tensor): Input tensor with shape [batch_size, channels, height, width]

        Returns:
            - output (tensor): The final output tensor from the Y-axis RNN.
        """
        batch_size, channels, H, W = self.get_shape(x)

        # Rearrange the tensor to (batch_size*H, W, channels) for
        # RNN processing over height This step prepares the data by aligning
        # it along the width, treating each row separately.
        # the keep_channel_last is true in case using channel last, but the
        # assumption of the transpose dims is that the input is channels first,
        # so is sign to keep the channels in the last dim.
        x_w = self.transpose(x, (0, 2, 3, 1),
                             keep_channel_last=True)  # Rearranges to (batch_size, H, W, channels)
        x_w = self.reshape(x_w, (batch_size * H, W, channels))

        output_x, _ = rnn_x(x_w)

        # Prepare the output from the X-axis RNN for Y-axis processing by
        # rearranging it to (batch_size*W, H, output_size_x),
        # enabling RNN application over width.
        output_x_reshape = self.reshape(output_x,
                                        (batch_size, H, W, self.output_size_x))
        output_x_permute = self.transpose(output_x_reshape, (0, 2, 1, 3))
        output_x_permute = self.reshape(output_x_permute,
                                        (batch_size * W, H, self.output_size_x))

        output, _ = rnn_y(output_x_permute)

        # Reshape and rearrange the final output to
        # (batch_size, channels, height, width),
        # restoring the original input dimensions with the transformed data.
        output = self.reshape(output, (batch_size, W, H, self.output_size_y))
        output = self.transpose(output, (0, 3, 2, 1), keep_channel_last=True)

        return output


class SRNN(GenericSRNN):
    """
    Implements the Separable Recurrent Neural Network (SRNN).
    This model extends a standard RNN by introducing separability in processing.
    """

    def __init__(self, input_size: int, hidden_size_x: int, hidden_size_y: int,
                 base: str = 'RNN', num_layers: int = 1, bias: bool = True,
                 batch_first: bool = True, dropout: float = 0.0,
                 bidirectional: bool = False) -> None:
        """
        Initializes the SRNN model with the given parameters.

        Parameters:
        - input_size: The number of expected features in the input `x`
        - hidden_size_x: The number of features in the hidden state `x`
        - hidden_size_y: The number of features in the hidden state `y`
        - base: The type of RNN to use (e.g., 'RNN', 'LSTM', 'GRU')
        - num_layers: Number of recurrent layers. E.g., setting `num_layers=2`
        would mean stacking two RNNs together
        - bias: If `False`, then the layer does not use bias weights `b_ih` and
        `b_hh`. Default: `True`
        - batch_first: If `True`, then the input and output tensors are provided
        as (batch, seq, feature). Default: `True`
        - dropout: If non-zero, introduces a `Dropout` layer on the outputs of
        each RNN layer except the last layer,
                   with dropout probability equal to `dropout`. Default: 0
       - bidirectional: If `True`, becomes a torch implementation of
       bidirectional RNN (two RNN blocks, one for the forward pass and one
       for the backward). Default: `False`

        Creates two `RNN` instances for processing in `x` and `y`
        dimensions, respectively.

        From our experiments, we found that the best results were
        obtained with the following parameters:
        base='RNN', num_layers=1, bias=True, batch_first=True, dropout=0
        """
        super(SRNN, self).__init__(hidden_size_x, hidden_size_y)

        # Dictionary mapping base types to their respective PyTorch class
        RNN_bases = {'RNN': RNN,
                     'GRU': GRU,
                     'LSTM': LSTM}

        self.rnn_x = RNN_bases[base](input_size=input_size,
                                     hidden_size=hidden_size_x,
                                     num_layers=num_layers,
                                     bias=bias,
                                     batch_first=batch_first,
                                     dropout=dropout,
                                     bidirectional=bidirectional)

        self.rnn_y = RNN_bases[base](input_size=hidden_size_x,
                                     hidden_size=hidden_size_y,
                                     num_layers=num_layers,
                                     bias=bias,
                                     batch_first=batch_first,
                                     dropout=dropout,
                                     bidirectional=bidirectional)

        # Output sizes of the model in the `x` and `y` dimensions.
        self.output_size_x = hidden_size_x
        self.output_size_y = hidden_size_y

    def forward(self, x):
        """
        Defines the forward pass of the SRNN.

        Parameters:
        - x: The input tensor to the RNN

        Returns:
        - The output of the SRNN after processing the input tensor
        `x` through both the `x` and `y` RNNs.
        """
        output = super().forward(x, self.rnn_x, self.rnn_y)
        return output


class SWSBiRNN(GenericSRNN):
    """
    Implements the Weights Shared Bi-directional Separable Recurrent Neural Network (WSBiSRNN).
    This model extends a standard RNN by introducing bi-directionality and separability in processing,
    with weight sharing to reduce the number of parameters.
    """

    def __init__(self, input_size: int, hidden_size_x: int, hidden_size_y: int,
                 base: str = 'RNN', num_layers: int = 1, bias: bool = True,
                 batch_first: bool = True, dropout: float = 0.0) -> None:
        """
        Initializes the WSBiSRNN model with the given parameters.

        Parameters:
        - input_size: The number of expected features in the input `x`
        - hidden_size_x: The number of features in the hidden state `x`
        - hidden_size_y: The number of features in the hidden state `y`
        - base: The type of RNN to use (e.g., 'RNN', 'LSTM', 'GRU')
        - num_layers: Number of recurrent layers. E.g., setting `num_layers=2`
        would mean stacking two RNNs together
        - bias: If `False`, then the layer does not use bias weights `b_ih` and
        `b_hh`. Default: `True`
        - batch_first: If `True`, then the input and output tensors are provided
        as (batch, seq, feature). Default: `True`
        - dropout: If non-zero, introduces a `Dropout` layer on the outputs of
        each RNN layer except the last layer,
                   with dropout probability equal to `dropout`. Default: 0

        Creates two `WSBiRNN` instances for processing in `x` and `y`
        dimensions, respectively.

        From our experiments, we found that the best results were
        obtained with the following parameters:
        base='RNN', num_layers=1, bias=True, batch_first=True, dropout=0
        """
        super(SWSBiRNN, self).__init__(hidden_size_x, hidden_size_y)

        self.rnn_x = WSBiRNN(input_size=input_size,
                             hidden_size=hidden_size_x,
                             num_layers=num_layers,
                             base=base,
                             bias=bias,
                             batch_first=batch_first,
                             dropout=dropout)

        self.rnn_y = WSBiRNN(input_size=hidden_size_x,
                             hidden_size=hidden_size_y,
                             num_layers=num_layers,
                             base=base,
                             bias=bias,
                             batch_first=batch_first,
                             dropout=dropout)

        # Output sizes of the model in the `x` and `y` dimensions.
        self.output_size_x = hidden_size_x
        self.output_size_y = hidden_size_y

    def forward(self, x):
        """
        Defines the forward pass of the WSBiSRNN.

        Parameters:
        - x: The input tensor to the RNN

        Returns:
        - The output of the WSBiSRNN after processing the input tensor `x` through both the `x` and `y` RNNs.
        """
        output = super().forward(x, self.rnn_x, self.rnn_y)
        return output


class WSBiRNN(Module):
    """
    WSBiRNN (Weight-Shared Bidirectional) RNN is a custom implementation of a bidirectional
    RNN that processes input sequences in both forward and reverse directions
    and combines the outputs. This class manually implements
    bidirectional functionality using a specified base RNN (e.g., vanilla RNN, GRU, LSTM)
    and combines the forward and reverse outputs.

    Attributes:
        rnn (Module): The RNN module used for processing sequences in the forward direction.
        hidden_size (int): The size of the hidden layer in the RNN.
        flip (Flip): An instance of the Flip class for reversing the sequence order.
        add (Add): An instance of the Add class for combining forward and reverse outputs.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 base: str = 'RNN', bias: bool = True, batch_first: bool = True,
                 dropout: float = 0.0) -> None:
        """
        Initializes the BiDirectionalRNN module with the specified parameters.

        Parameters:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h`.
            num_layers (int, optional): Number of recurrent layers. Default: 1.
            base (str, optional): Type of RNN ('RNN', 'GRU', 'LSTM'). Default: 'RNN'.
            bias (bool, optional): If False, then the layer does not use bias weights. Default: True.
            batch_first (bool, optional): If True, then the input and output tensors are provided
                                          as (batch, seq, feature). Default: True.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of
                                       each RNN layer except the last layer. Default: 0.
        """
        super(WSBiRNN, self).__init__()

        # Dictionary mapping base types to their respective PyTorch class
        RNN_bases = {'RNN': RNN,
                     'GRU': GRU,
                     'LSTM': LSTM}

        # Initialize the forward RNN module
        self.rnn = RNN_bases[base](input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bias=bias,
                                   batch_first=batch_first,
                                   dropout=dropout,
                                   bidirectional=False)

        # Initialize utilities for flipping sequences and combining outputs
        self.flip = Flip()
        self.add = Add()

    def forward(self, x):
        """
        Defines the forward pass for the bidirectional RNN.

        Parameters:
            x (Tensor): The input sequence tensor.

        Returns:
            Tensor: The combined output of the forward and reverse processed sequences.
            _: Placeholder for compatibility with the expected RNN output format.
        """
        # Reverse the sequence for processing in the reverse direction
        x_reverse = self.flip(x, [1])

        # Process sequences in forward and reverse directions
        out_forward, _ = self.rnn(x)
        out_reverse, _ = self.rnn(x_reverse)

        # Flip the output from the reverse direction to align with forward
        # direction
        out_reverse_flip = self.flip(out_reverse, [1])

        # Combine the outputs from the forward and reverse directions
        return self.add(out_reverse_flip, out_forward), _


class S2f(Module):
    # Synaptics C2f.  Constructed from tflite inspection
    def __init__(self, in_channels, n, out_channels=None, nslice=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if nslice is not None:
            c = nslice
        else:
            c = in_channels // 2
        self.slice = ChannelSlice(slice(c))
        self.ir = ModuleList([InvertedResidual(c, 1) for _ in range(n)])
        self.cat = Cat()
        self.decode = CoBNRLU(in_channels + n*c, out_channels, bias=True)

    def forward(self, x):
        out = [x]
        y = self.slice(x)
        for ir in self.ir:
            out.append(y := ir(y))
        return self.decode(self.cat(out))
