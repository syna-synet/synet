"""base.py is the "export" layer of synet.  As such, it includes the
logic of how to run as a keras model.  This is handled by cheking the
'askeras' context manager, and running in "keras mode" if that context
is enabled.  As a rule of thumb to differentiate between base.py,
layers.py:

- base.py should only import from torch, keras, and tensorflow.
- layers.py should only import from base.py.
"""

from typing import Tuple, Union, Optional, List
from torch import cat as torch_cat, minimum, tensor, no_grad
from torch.nn import (Module as Torch_Module,
                      Conv2d as Torch_Conv2d,
                      BatchNorm2d as Torch_Batchnorm,
                      ModuleList,
                      ReLU as Torch_ReLU,
                      ConvTranspose2d as Torch_ConvTranspose2d,
                      Upsample as Torch_Upsample,
                      AdaptiveAvgPool2d as Torch_AdaptiveAvgPool,
                      Dropout as Torch_Dropout,
                      Linear as Torch_Linear)
from torch.nn.functional import pad
import torch.nn as nn
import torch


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
        if not hasattr(self, 'padding'):
            self.padding = 'same'
        if not hasattr(self, 'groups'):
            self.groups = 1
        if not isinstance(self.padding, str):
            self.padding = "same" if self.padding else 'valid'

        if askeras.use_keras:
            return self.as_keras(x)

        if self.padding == "valid":
            return self.conv(x)

        # make padding like in tensorflow, which right aligns convolutionn.
        H, W = (s if isinstance(s, int) else s.item() for s in x.shape[-2:])
        # radius of the kernel and carry.  Border size + carry.  All in y
        ry, rcy = divmod(self.kernel_size[0] - 1, 2)
        by, bcy = divmod((H - 1) % self.stride - rcy, 2)
        # radius of the kernel and carry.  Border size + carry.  All in x
        rx, rcx = divmod(self.kernel_size[1] - 1, 2)
        bx, bcx = divmod((W - 1) % self.stride - rcx, 2)
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

            rest_chans = [i for i in range(self.out_channels)
                          if i not in chans]
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
        # temporary code for backwards compatibility
        if not hasattr(self, 'name'):
            self.name = None
        from keras.layers import ReLU as Keras_ReLU
        return Keras_ReLU(self.max_val, name=self.name)(x)


class BatchNorm(Module):
    def __init__(self, features, epsilon=1e-3, momentum=0.999):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.module = Torch_Batchnorm(features, epsilon, momentum)

    def forward(self, x):
        # temporary code for backwards compatibility
        if hasattr(self, 'batchnorm'):
            self.module = self.batchnorm
        return super().forward(x)

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

    def forward(self, x):
        # temporary code for backwards compatibility
        if not hasattr(self, 'module'):
            self.module = self.upsample
        return super().forward(x)

    def as_keras(self, x):
        from keras.layers import UpSampling2D
        return UpSampling2D(size=self.scale_factor,
                            interpolation=self.mode,
                            )(x)


class Sequential(Module):
    def __init__(self, *sequence):
        super().__init__()
        self.ml = ModuleList(sequence)

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
        params = [self.module.weight.detach().numpy().transpose(1, 0)]
        if self.use_bias:
            params.append(self.module.bias.detach().numpy())
        dense = Dense(out_c, use_bias=self.use_bias)
        dense.build(x.shape[1:])
        dense.set_weights(params)
        return dense(x)


class Transpose(Module):
    """
    A class designed to transpose tensors according to specified dimension permutations, compatible
    with both PyTorch and TensorFlow (Keras). It allows for flexible tensor manipulation, enabling
    dimension reordering to accommodate the requirements of different neural network architectures
    or operations.

    The class supports optional channel retention during transposition in TensorFlow to ensure
    compatibility with Keras' channel ordering conventions.
    """

    def forward(self, x, perm: Union[Tuple[int], List[int]],
                keep_channel_last: bool = False):
        """
        Transposes the input tensor according to the specified dimension permutation. If integrated
        with Keras, it converts PyTorch tensors to TensorFlow tensors before transposing, with an
        option to retain channel ordering as per Keras convention.

        Parameters:
            x (Tensor): The input tensor to be transposed.
            perm (tuple or list): The permutation of dimensions to apply to the tensor.
            keep_channel_last (bool, optional): Specifies whether to adjust the permutation to
                                                 retain Keras' channel ordering convention. Default
                                                 is False.

        Returns:
            Tensor: The transposed tensor.
        """
        if askeras.use_keras:
            return self.as_keras(x, perm, keep_channel_last)
        # Use PyTorch's permute method for the operation
        return x.permute(*perm)

    def as_keras(self, x, perm: Union[Tuple[int], List[int]],
                 keep_channel_last: bool):
        """
        Handles tensor transposition in a TensorFlow/Keras environment, converting PyTorch tensors
        to TensorFlow tensors if necessary, and applying the specified permutation. Supports an
        option for channel retention according to Keras conventions.

        Parameters:
            x (Tensor): The input tensor, possibly a PyTorch tensor.
            perm (tuple or list): The permutation of dimensions to apply.
            keep_channel_last (bool): If True, adjusts the permutation to retain Keras' channel
                                       ordering convention.

        Returns:
            Tensor: The transposed tensor in TensorFlow format.
        """
        import tensorflow as tf

        # Adjust for TensorFlow's default channel ordering if necessary
        tf_format = [0, 3, 1, 2]

        # Map PyTorch indices to TensorFlow indices if channel retention is enabled
        mapped_indices = [tf_format[index] for index in
                          perm] if keep_channel_last else perm

        # Convert PyTorch tensors to TensorFlow tensors if necessary
        x_tf = tf.convert_to_tensor(x.detach().numpy(),
                                    dtype=tf.float32) if isinstance(x,
                                                                    torch.Tensor) else x

        # Apply the transposition with TensorFlow's transpose method
        x_tf_transposed = tf.transpose(x_tf, perm=mapped_indices)

        return x_tf_transposed


class Reshape(Module):
    """
    A class designed to reshape tensors to a specified shape, compatible with both PyTorch and
    TensorFlow (Keras). This class facilitates tensor manipulation across different deep learning
    frameworks, enabling the adjustment of tensor dimensions to meet the requirements of different
    neural network layers or operations.

    It supports dynamic reshaping capabilities, automatically handling the conversion between
    PyTorch and TensorFlow tensors and applying the appropriate reshaping operation based on the
    runtime context.
    """

    def forward(self, x, shape: Union[Tuple[int], List[int]]):
        """
        Reshapes the input tensor to the specified shape. If integrated with Keras, it converts
        PyTorch tensors to TensorFlow tensors before reshaping.

        Parameters:
            x (Tensor): The input tensor to be reshaped.
            shape (tuple or list): The new shape for the tensor. The specified shape can include
                                   a `-1` to automatically infer the dimension that ensures the
                                   total size remains constant.

        Returns:
            Tensor: The reshaped tensor.
        """
        if askeras.use_keras:
            return self.as_keras(x, shape)
        # Use PyTorch's reshape method for the operation
        return x.reshape(*shape)

    def as_keras(self, x, shape: Union[Tuple[int], List[int]]):
        """
        Converts PyTorch tensors to TensorFlow tensors, if necessary, and performs the reshape
        operation using TensorFlow's reshape function. This method ensures compatibility and
        functionality within a TensorFlow/Keras environment.

        Parameters:
            x (Tensor): The input tensor, possibly a PyTorch tensor.
            shape (tuple or list): The new shape for the tensor, including the possibility
                                   of using `-1` to infer a dimension automatically.

        Returns:
            Tensor: The reshaped tensor in TensorFlow format.
        """
        import tensorflow as tf
        # Convert PyTorch tensors to TensorFlow tensors if necessary
        x_tf = tf.convert_to_tensor(x.detach().numpy(),
                                    dtype=tf.float32) if isinstance(x,
                                                                    torch.Tensor) else x

        # Use TensorFlow's reshape function to adjust the tensor's dimensions
        x_tf_reshaped = tf.reshape(x_tf, shape)

        # TensorFlow's reshape might introduce an additional dimension if shape is fully defined,
        # use tf.squeeze to adjust dimensions if necessary
        return x_tf_reshaped


class Flip(Module):
    """
    A class to flip tensors along specified dimensions, supporting both PyTorch and TensorFlow
    (Keras). This class enables consistent tensor manipulation across different deep learning
    frameworks, facilitating operations like data augmentation or image processing where flipping
    is required.

    The class automatically detects the runtime environment to apply the appropriate flipping
    operation, handling tensor conversions between PyTorch and TensorFlow as needed.
    """

    def forward(self, x, dims: Union[List[int], Tuple[int]]):
        """
        Flips the input tensor along specified dimensions. If integrated with Keras, it
        converts PyTorch tensors to TensorFlow tensors before flipping.

        Parameters:
            x (Tensor): The input tensor to be flipped.
            dims (list or tuple): The dimensions along which to flip the tensor.

        Returns:
            Tensor: The flipped tensor.
        """
        # Check if Keras usage is flagged and handle accordingly
        if askeras.use_keras:
            return self.as_keras(x, dims)
        # Use PyTorch's flip function for the operation
        return torch.flip(x, dims)

    def as_keras(self, x, dims: Union[List[int], Tuple[int]]):
        """
        Converts PyTorch tensors to TensorFlow tensors, if necessary, and performs the flip
        operation using TensorFlow's reverse function. This method ensures compatibility and
        functionality within a TensorFlow/Keras environment.

        Parameters:
            x (Tensor): The input tensor, possibly a PyTorch tensor.
            dims (list or tuple): The dimensions along which to flip the tensor.

        Returns:
            Tensor: The flipped tensor in TensorFlow format.
        """
        import tensorflow as tf
        # Convert PyTorch tensors to TensorFlow tensors if necessary
        x_tf = tf.convert_to_tensor(x.detach().numpy(),
                                    dtype=tf.float32) if isinstance(x,
                                                                    torch.Tensor) else x

        # Use TensorFlow's reverse function for flipping along specified dimensions
        return tf.reverse(x_tf, axis=dims)


class Add(Module):
    """
    A class designed to perform element-wise addition on tensors, compatible with both
    PyTorch and TensorFlow (Keras). This enables seamless operation across different deep
    learning frameworks, supporting the addition of tensors regardless of their originating
    framework.

    The class automatically handles framework-specific tensor conversions and uses the
    appropriate addition operation based on the runtime context, determined by whether
    TensorFlow/Keras or PyTorch is being used.
    """

    def forward(self, x, y):
        """
        Performs element-wise addition of two tensors. If integrated with Keras, converts
        PyTorch tensors to TensorFlow tensors before addition.

        Parameters:
            x (Tensor): The first input tensor.
            y (Tensor): The second input tensor to be added to the first.

        Returns:
            Tensor: The result of element-wise addition of `x` and `y`.
        """
        if askeras.use_keras:
            return self.as_keras(x, y)
        # Use PyTorch's add function for element-wise addition
        return torch.add(x, y)

    def as_keras(self, x, y):
        """
        Converts PyTorch tensors to TensorFlow tensors, if necessary, and performs
        element-wise addition using TensorFlow's add function. This method ensures
        compatibility and functionality within a TensorFlow/Keras environment.

        Parameters:
            x (Tensor): The first input tensor, possibly a PyTorch tensor.
            y (Tensor): The second input tensor, possibly a PyTorch tensor.

        Returns:
            Tensor: The result of element-wise addition of `x` and `y` in TensorFlow format.
        """
        import tensorflow as tf
        # Convert PyTorch tensors to TensorFlow tensors if necessary
        x_tf = tf.convert_to_tensor(x.detach().numpy(),
                                    dtype=tf.float32) if isinstance(x,
                                                                    torch.Tensor) else x
        y_tf = tf.convert_to_tensor(y.detach().numpy(),
                                    dtype=tf.float32) if isinstance(y,
                                                                    torch.Tensor) else y

        # Use TensorFlow's add function for element-wise addition
        return tf.add(x_tf, y_tf)


class Shape(Module):
    """
    A utility class for obtaining the shape of a tensor in a format compatible
    with either PyTorch or Keras. This class facilitates the transformation of
    tensor shapes, particularly useful for adapting model input or output
    dimensions across different deep learning frameworks.

    The class provides a method to directly return the shape of a tensor for
    PyTorch use cases and an additional method for transforming the shape to a
    Keras-compatible format, focusing on the common difference in dimension
    ordering between the two frameworks.
    """

    def forward(self, x):
        """
        Returns the shape of the tensor. If integrated with Keras, it transforms the tensor shape
        to be compatible with Keras dimension ordering.

        Parameters:
            x (Tensor): The input tensor whose shape is to be obtained or transformed.

        Returns:
            Tuple: The shape of the tensor, directly returned for PyTorch or transformed for Keras.
        """
        if askeras.use_keras:
            return self.as_keras(x)
        # Directly return the shape for PyTorch tensors
        return x.shape

    def as_keras(self, x):
        """
        Transforms the tensor shape to be compatible with Keras' expected dimension ordering.
        This method is designed to switch between CHW and HWC formats based on the tensor's
        dimensionality, handling common cases for 2D, 3D, and 4D tensors.

        Parameters:
            x (Tensor): The input tensor whose shape is to be transformed for Keras.

        Returns:
            Tuple: The transformed shape of the tensor, suitable for Keras models.
        """
        # Handle different tensor dimensionality with appropriate
        # transformations
        if len(x.shape) == 4:  # Assuming NCHW format, convert to NHWC
            N, W, H, C = x.shape
            x_shape = (N, C, H, W)
        elif len(x.shape) == 3:  # Assuming CHW format, convert to HWC
            H, W, C = x.shape
            x_shape = (C, H, W)
        else:  # Assuming 2D tensor, no channel dimension involved
            H, W = x.shape
            x_shape = (H, W)

        return x_shape


class GenericRNN(nn.Module):
    """
    A base class for customizable RNN models supporting RNN, GRU, and LSTM networks.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = False, bias: bool = True,
                 batch_first: bool = True, dropout: float = 0) -> None:
        super(GenericRNN, self).__init__()
        self.bidirectional = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.input_size = input_size
        self.batch_first = batch_first

    def init_rnn(self, input_size: int, hidden_size: int, num_layers: int,
                 bidirectional: bool, bias: bool, batch_first: bool,
                 dropout: float) -> None:

        raise NotImplementedError("Must be implemented by subclass.")

    def forward(self, x, h0=None, c0=None):

        raise NotImplementedError("Must be implemented by subclass.")

    def as_keras(self, x):
        raise NotImplementedError("Must be implemented by subclass.")

    def generic_as_keras(self, x, RNNBase):
        """
        Converts the model architecture and weights to a Keras-compatible format and applies
        the model to the provided input.

        This method enables the use of PyTorch-trained models within the Keras framework by
        converting the input tensor to a TensorFlow tensor, recreating the model architecture
        in Keras, and setting the weights accordingly.

        Parameters:
            x (Tensor): The input tensor for the model, which can be a PyTorch tensor.
            RNNBase: The base class for the RNN model to be used in the Keras model.

        Returns:
            Tuple[Tensor, None]: A tuple containing the output of the Keras model applied to the
                                 converted input tensor and None (since Keras models do not
                                 necessarily return the final hidden state as PyTorch models do).

        Raises:
            ImportError: If the required TensorFlow or Keras modules are not available.
        """

        # Import necessary modules from Keras and TensorFlow
        from keras.layers import Bidirectional
        from keras.models import Sequential as KerasSequential
        import tensorflow as tf

        # Convert PyTorch tensor to TensorFlow tensor if necessary
        if isinstance(x, torch.Tensor):
            x_tf = tf.convert_to_tensor(x.detach().numpy(), dtype=tf.float32)
        else:
            x_tf = x

        # Create a Keras Sequential model for stacking layers
        model = KerasSequential()

        # Add RNN layers to the Keras model
        for i in range(self.num_layers):
            # Determine if input shape needs to be specified (only for the first layer)
            if i == 0:
                layer = RNNBase(units=self.hidden_size, return_sequences=True,
                                input_shape=list(x.shape[1:]),
                                use_bias=self.bias,
                                dropout=self.dropout if i < self.num_layers - 1 else 0)
            else:
                layer = RNNBase(units=self.hidden_size, return_sequences=True,
                                use_bias=self.bias,
                                dropout=self.dropout if i < self.num_layers - 1 else 0)

            # Wrap the layer with Bidirectional if needed
            if self.bidirectional == 2:
                layer = Bidirectional(layer)

            model.add(layer)

        # Apply previously extracted PyTorch weights to the Keras model
        self.set_keras_weights(model)

        # Process the input through the Keras model
        output = model(x_tf)

        # Return the output and None for compatibility with PyTorch output format
        return output, None

    def extract_pytorch_rnn_weights(self):
        """
        Extracts weights from a PyTorch model's RNN layers and prepares them for
        transfer to a Keras model.

        This function iterates through the named parameters of a PyTorch model,
        detaching them from the GPU (if applicable),
        moving them to CPU memory, and converting them to NumPy arrays.
        It organizes these weights in a dictionary,
        using the parameter names as keys, which facilitates their later use in
        setting weights for a Keras model.

        Returns:
        A dictionary containing the weights of the PyTorch model, with parameter
        names as keys and their corresponding NumPy array representations as values.
        """

        weights = {}  # Initialize a dictionary to store weights

        # Iterate through the model's named parameters
        for name, param in self.named_parameters():
            # Process the parameter name to extract the relevant part
            # and use it as the key in the weights dictionary
            key = name.split('.')[
                -1]  # Extract the last part of the parameter name

            # Detach the parameter from the computation graph, move it to CPU,
            # and convert to NumPy array
            weights[key] = param.detach().cpu().numpy()

        return weights  # Return the dictionary of weights

    def set_keras_weights(self, keras_model):
        raise NotImplementedError("Must be implemented by subclass.")

    def generic_set_keras_weights(self, keras_model, RNNBase: str):
        """
        Sets the weights of a Keras model based on the weights from a PyTorch model.

        This function is designed to transfer weights from PyTorch RNN layers (SimpleRNN, GRU, LSTM)
        to their Keras counterparts, including handling for bidirectional layers. It ensures that the
        weights are correctly transposed and combined to match Keras's expectations.

        Parameters:
        - keras_model: The Keras model to update the weights for.
        """

        # Import necessary modules
        from keras.layers import Bidirectional
        import numpy as np

        # Extract weights from PyTorch model
        pytorch_weights = self.extract_pytorch_rnn_weights()

        # Iterate over each layer in the Keras model
        for layer in keras_model.layers:
            # Check if layer is bidirectional and set layers to update
            # accordingly
            if isinstance(layer, Bidirectional):
                layers_to_update = [layer.layer, layer.backward_layer]
            else:
                layers_to_update = [layer]

            # Update weights for each RNN layer in layers_to_update
            for rnn_layer in layers_to_update:

                num_gates = {'SimpleRNN': 1, 'GRU': 3, 'LSTM': 4}.get(
                    RNNBase, 0)

                # Initialize lists for input-hidden, hidden-hidden weights,
                # and biases
                ih_weights, hh_weights, biases = [], [], []

                # Process weights and biases for each gate
                for i in range(num_gates):
                    gate_suffix = f'_l{i}'
                    for prefix in ('weight_ih', 'weight_hh'):
                        key = f'{prefix}{gate_suffix}'
                        if key in pytorch_weights:
                            weights = \
                                pytorch_weights[key].T  # Transpose to match Keras shape

                            (ih_weights if prefix == 'weight_ih' else hh_weights) \
                                .append(weights)

                    bias_keys = (
                        f'bias_ih{gate_suffix}', f'bias_hh{gate_suffix}')
                    if all(key in pytorch_weights for key in bias_keys):
                        # Sum biases from input-hidden and hidden-hidden
                        biases.append(
                            sum(pytorch_weights[key] for key in bias_keys))

                # Combine weights and biases into a format suitable for Keras
                keras_weights = [np.vstack(ih_weights),
                                 np.vstack(hh_weights), np.hstack(biases)]

                # Set the weights for the Keras layer
                if not isinstance(layer, Bidirectional):
                    rnn_layer.set_weights(keras_weights)
                else:
                    rnn_layer.cell.set_weights(keras_weights)


class RNN(GenericRNN):
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.rnn = nn.RNN(input_size=kwargs['input_size'],
                          hidden_size=kwargs['hidden_size'],
                          num_layers=kwargs['num_layers'],
                          bias=kwargs['bias'],
                          batch_first=kwargs['batch_first'],
                          dropout=kwargs['dropout'],
                          bidirectional=kwargs['bidirectional'])

    def forward(self, x, h0=None):
        if askeras.use_keras:
            return self.as_keras(x)

        out, h = self.rnn(x, h0)
        return out, h

    def as_keras(self, x):
        """
        Converts the model architecture and weights to a Keras-compatible format and applies
        the model to the provided input.

        This method enables the use of PyTorch-trained models within the Keras framework by
        converting the input tensor to a TensorFlow tensor, recreating the model architecture
        in Keras, and setting the weights accordingly.

        Parameters:
            x (Tensor): The input tensor for the model, which can be a PyTorch tensor.

        Returns:
            Tuple[Tensor, None]: A tuple containing the output of the Keras model applied to the
                                 converted input tensor and None (since Keras models do not
                                 necessarily return the final hidden state as PyTorch models do).

        Raises:
            ImportError: If the required TensorFlow or Keras modules are not available.
        """

        # Import necessary modules from Keras and TensorFlow
        from keras.layers import SimpleRNN

        output, _ = super().generic_as_keras(x, SimpleRNN)

        return output, None

    def set_keras_weights(self, keras_model):
        """
        Sets the weights of a Keras model based on the weights from a PyTorch model.

        This function is designed to transfer weights from PyTorch RNN layers
        to their Keras counterparts, including handling for bidirectional layers. It ensures that the
        weights are correctly transposed and combined to match Keras's expectations.

        Parameters:
        - keras_model: The Keras model to update the weights for.
        """

        self.generic_set_keras_weights(keras_model, 'SimpleRNN')


class GRU(GenericRNN):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__(*args, **kwargs)
        self.rnn = nn.GRU(input_size=kwargs['input_size'],
                          hidden_size=kwargs['hidden_size'],
                          num_layers=kwargs['num_layers'],
                          bias=kwargs['bias'],
                          batch_first=kwargs['batch_first'],
                          dropout=kwargs['dropout'],
                          bidirectional=kwargs['bidirectional'])

    def forward(self, x, h0=None):
        if askeras.use_keras:
            return self.as_keras(x)

        out, h = self.rnn(x, h0)
        return out, h

    def as_keras(self, x):
        """
        Converts the model architecture and weights to a Keras-compatible format and applies
        the model to the provided input.

        This method enables the use of PyTorch-trained models within the Keras framework by
        converting the input tensor to a TensorFlow tensor, recreating the model architecture
        in Keras, and setting the weights accordingly.

        Parameters:
            x (Tensor): The input tensor for the model, which can be a PyTorch tensor.

        Returns:
            Tuple[Tensor, None]: A tuple containing the output of the Keras model applied to the
                                 converted input tensor and None (since Keras models do not
                                 necessarily return the final hidden state as PyTorch models do).

        Raises:
            ImportError: If the required TensorFlow or Keras modules are not available.
        """

        # Import necessary modules from Keras and TensorFlow
        from keras.layers import GRU

        output, _ = super().generic_as_keras(x, GRU)

        return output, None

    def set_keras_weights(self, keras_model):
        """
        Sets the weights of a Keras model based on the weights from a PyTorch model.

        This function is designed to transfer weights from PyTorch RNN layers
        to their Keras counterparts, including handling for bidirectional layers. It ensures that the
        weights are correctly transposed and combined to match Keras's expectations.

        Parameters:
        - keras_model: The Keras model to update the weights for.
        """

        self.generic_set_keras_weights(keras_model, 'GRU')


class LSTM(GenericRNN):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        self.rnn = nn.GRU(input_size=kwargs['input_size'],
                          hidden_size=kwargs['hidden_size'],
                          num_layers=kwargs['num_layers'],
                          bias=kwargs['bias'],
                          batch_first=kwargs['batch_first'],
                          dropout=kwargs['dropout'],
                          bidirectional=kwargs['bidirectional'])

    def forward(self, x, h0=None, c0=None):
        if askeras.use_keras:
            return self.as_keras(x)

        out, h = self.rnn(x, (h0, c0))

        return out, h

    def as_keras(self, x):
        """
        Converts the model architecture and weights to a Keras-compatible format and applies
        the model to the provided input.

        This method enables the use of PyTorch-trained models within the Keras framework by
        converting the input tensor to a TensorFlow tensor, recreating the model architecture
        in Keras, and setting the weights accordingly.

        Parameters:
            x (Tensor): The input tensor for the model, which can be a PyTorch tensor.

        Returns:
            Tuple[Tensor, None]: A tuple containing the output of the Keras model applied to the
                                 converted input tensor and None (since Keras models do not
                                 necessarily return the final hidden state as PyTorch models do).

        Raises:
            ImportError: If the required TensorFlow or Keras modules are not available.
        """

        # Import necessary modules from Keras and TensorFlow
        from keras.layers import LSTM

        output, _ = super().generic_as_keras(x, LSTM)

        return output, None

    def set_keras_weights(self, keras_model):
        """
        Sets the weights of a Keras model based on the weights from a PyTorch model.

        This function is designed to transfer weights from PyTorch RNN layers
        to their Keras counterparts, including handling for bidirectional layers. It ensures that the
        weights are correctly transposed and combined to match Keras's expectations.

        Parameters:
        - keras_model: The Keras model to update the weights for.
        """

        self.generic_set_keras_weights(keras_model, 'LSTM')


class ChannelSlice(Module):
    def __init__(self, slice):
        super().__init__()
        self.slice = slice

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        return x[:, self.slice]

    def as_keras(self, x):
        return x[(len(x.shape)-1)*(slice(None),)+(self.slice,)]
