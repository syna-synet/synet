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
                   Sequential, RNN, Transpose, Reshape, Flip, Add, Shape)


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
                 out_channels=None, stride=1, kernel_size=3):
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
        self.cheq = in_channels == out_channels
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
                                  for out_channels in out_channels)
                                )

    def forward(self, x):
        return self.model(x)


class RNNHead(Module):
    """
    A modular neural network component that stacks multiple instances of HierarchicalRNN
    modules to process sequences through multiple layers of abstraction. This structure
    is designed to enhance the model's ability to capture complex temporal patterns by
    sequentially applying hierarchical processing steps.
    """

    def __init__(self, input_size, hidden_size_x, hidden_size_y, num_layers=1,
                 base='RNN', bidirectional=False, num_times=4, bias=True,
                 batch_first=True, dropout=0):
        """
        Initializes the RNNHead module with specified parameters, dynamically creating
        a series of HierarchicalRNN modules based on the provided configuration.

        Parameters:
            input_size (int): The number of expected features in the input `x`.
            hidden_size_x (int): The size of the hidden layer `x` within each HierarchicalRNN.
            hidden_size_y (int): The size of the hidden layer `y` within each HierarchicalRNN,
                                 also the output size of each HierarchicalRNN module.
            num_layers (int, optional): Number of recurrent layers in each HierarchicalRNN. Default: 1.
            base (str, optional): Type of RNN (e.g., 'RNN', 'GRU', 'LSTM') used in HierarchicalRNN. Default: 'RNN'.
            bidirectional (bool, optional): If True, creates bidirectional HierarchicalRNNs. Default: False.
            num_times (int, optional): The number of HierarchicalRNN modules to stack. Default: 4.
            bias (bool, optional): If False, then the layers will not use bias weights. Default: True.
            batch_first (bool, optional): If True, then the input and output tensors are provided
                                          as (batch, seq, feature). Default: True.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of
                                       each RNN layer except the last layer, in each HierarchicalRNN. Default: 0.
        """
        super(RNNHead, self).__init__()

        self.stacked_layers = ModuleList()

        # Dynamically create and stack HierarchicalRNN modules
        for _ in range(num_times):
            self.stacked_layers.append(
                HierarchicalRNN(input_size=input_size,
                                hidden_size_x=hidden_size_x,
                                hidden_size_y=hidden_size_y,
                                num_layers=num_layers,
                                base=base,
                                bias=bias,
                                batch_first=batch_first,
                                dropout=dropout,
                                bidirectional=bidirectional)
            )
            # Update input size for the next HierarchicalRNN module
            input_size = hidden_size_y

    def forward(self, x):
        """
        Defines the forward pass through the stacked HierarchicalRNN modules.

        Parameters:
            x (Tensor): The input sequence tensor to the RNNHead.

        Returns:
            Tensor: The output tensor after processing by all stacked HierarchicalRNN modules.
        """
        # Sequentially pass the input through all stacked HierarchicalRNN modules
        for layer in self.stacked_layers:
            x = layer(x)
        return x


class CoBNRLU(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, bias=False, padding=True, groups=1, max_val=6):
        super().__init__()
        self.module = Sequential(Conv2d(in_channels, out_channels,
                                        kernel_size, stride, bias, padding,
                                        groups),
                                 BatchNorm(out_channels),
                                 ReLU(max_val))

    def forward(self, x):
        return self.module(x)


class HierarchicalRNN(Module):
    """
    HierarchicalRNN processes a given tensor first along its X-axis using an RNN,
    and then feeds the output of this RNN along the Y-axis to another RNN.

    Args:
    - hidden_size_x (int): The number of features in the hidden state of the
    X-axis RNN.
    - hidden_size_y (int): The number of features in the hidden state of the
    Y-axis RNN.
    - num_layers (int, optional): Number of recurrent layers for each RNN.
    Default: 1.

    Returns:
    - output (tensor): Output from the Y-axis RNN after processing the output of
    the X-axis RNN.
    - hn_x (tensor): Hidden state for the RNN processing along the X-axis after
    the last timestep.
    - hn_y (tensor): Hidden state for the RNN processing along the Y-axis after
    the last timestep.
    """

    def __init__(self, input_size, hidden_size_x, hidden_size_y, num_layers=1,
                 base='RNN', bidirectional=False, bias=True, batch_first=True,
                 dropout=0):
        super(HierarchicalRNN, self).__init__()

        self.bidirectional = 2 if bidirectional else 1

        RNN_base = BiDirectionalRNN if bidirectional else RNN

        self.rnn_x = RNN_base(input_size=input_size,
                              hidden_size=hidden_size_x,
                              num_layers=num_layers,
                              base=base,
                              bias=bias,
                              batch_first=batch_first,
                              dropout=dropout)

        self.rnn_y = RNN_base(input_size=hidden_size_x,
                              hidden_size=hidden_size_y,
                              num_layers=num_layers,
                              base=base,
                              bias=bias,
                              batch_first=batch_first,
                              dropout=dropout)

        self.output_size_x = hidden_size_x
        self.output_size_y = hidden_size_y

        self.transpose = Transpose()
        self.reshape = Reshape()
        self.get_shape = Shape()

    def forward(self, x):
        """
        Forward pass for the HierarchicalRNN module.

        Args:
        - x (tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
        - output (tensor): Output from the Y-axis RNN.
        """
        N, C, H, W = self.get_shape(x)
        # RNN over height (h)

        # Rearrange the tensor to the shape (N*H, W, C)
        x_w = self.transpose(x, (0, 2, 3, 1),
                             keras_keep_channel=True)  # (N, H, W, C )
        x_w = self.reshape(x_w, (N * H, W, C))

        output_x, h = self.rnn_x(x_w)

        # RNN over width (w)

        # Rearrange the output from the first RNN to the shape (N*W, H, C)
        output_x_reshape = self.reshape(output_x, (N, H, W, self.output_size_x))
        output_x_permute = self.transpose(output_x_reshape, (0, 2, 1, 3))
        output_x_permute = self.reshape(output_x_permute,
                                        (N * W, H, self.output_size_x))

        output, h = self.rnn_y(output_x_permute)

        # Reshape to (batch, channels, height, width)
        output = self.reshape(output, (N, W, H, self.output_size_y))
        output = self.transpose(output, (0, 3, 2, 1), keras_keep_channel=True)

        return output


class BiDirectionalRNN(Module):
    """
    A custom implementation of a bidirectional RNN that processes input sequences in both
    forward and reverse directions and combines the outputs. This class manually implements
    bidirectional functionality using a specified base RNN (e.g., vanilla RNN, GRU, LSTM)
    and combines the forward and reverse outputs.

    Attributes:
        rnn (Module): The RNN module used for processing sequences in the forward direction.
        hidden_size (int): The size of the hidden layer in the RNN.
        flip (Flip): An instance of the Flip class for reversing the sequence order.
        add (Add): An instance of the Add class for combining forward and reverse outputs.
    """

    def __init__(self, input_size, hidden_size, num_layers=1,
                 base='RNN', bias=True, batch_first=True,
                 dropout=0):
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
        super(BiDirectionalRNN, self).__init__()

        # Initialize the forward RNN module
        self.rnn = RNN(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       base=base,
                       bias=bias,
                       batch_first=batch_first,
                       dropout=dropout,
                       bidirectional=False)
        self.hidden_size = hidden_size

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

        # Flip the output from the reverse direction to align with forward direction
        out_reverse_flip = self.flip(out_reverse, [1])

        # Combine the outputs from the forward and reverse directions
        return self.add(out_reverse_flip, out_forward), _