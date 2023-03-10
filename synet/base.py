from torch.nn import Module
from torch.nn.init import uniform_


class AsKeras:
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


t_conv_wght_to_k = lambda wght: wght.detach().numpy().transpose(2, 3, 1, 0)
from torch.nn import Conv2d as Torch_Conv2d
from torch.nn.functional import pad
from keras.layers import Conv2D as Keras_Conv2d
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=False):
        super().__init__()
        assert (kernel_size, stride) in [(1, 1), (3, 1), (3, 2)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = Torch_Conv2d(in_channels  = in_channels,
                                 out_channels = out_channels,
                                 kernel_size  = kernel_size,
                                 bias         = bias,
                                 stride       = stride)
        self.use_bias = bias
        if self.use_bias:
            self.bias = self.conv.bias

    def forward(self, x):
        # make padding like in tensorflow, which right aligns convolutions.
        if askeras.use_keras:
            return self.as_keras(x)
        if self.kernel_size == 3:
            if self.stride == 1:
                x = pad(x, (1,1,1,1))
            elif self.stride == 2:
                x = pad(x, (x.shape[-1]%2, 1, x.shape[-2]%2, 1))
        return self.conv(x)

    def as_keras(self, x):
        assert x.shape[-1] == self.in_channels
        conv = Keras_Conv2d(filters     = self.out_channels,
                            kernel_size = self.kernel_size,
                            strides     = self.stride,
                            padding     = "same",
                            use_bias    = self.use_bias)
        conv.build(x.shape[1:])
        weight = self.conv.weight.detach().numpy().transpose(2, 3, 1, 0)
        conv.set_weights([weight, self.conv.bias.detach().numpy()]
                         if self.use_bias else
                         [weight])
        return conv(x)


from torch import mean
from tensorflow import expand_dims, reduce_mean
from keras.layers import Lambda
class Grayscale(Module):
    def forward(self, x):
        if askeras.use_keras:
            return x
        return x.mean(1, keepdims=True)



from keras.layers import Concatenate as Keras_Concatenate
from torch import cat as torch_cat
class Cat(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim=dim
    def forward(self, xs):
        if askeras.use_keras:
            return self.as_keras(xs)
        return torch_cat(xs, dim=self.dim)
    def as_keras(self, xs):
        assert all(len(x.shape) == 4 for x in xs)
        dim = {0:0, 1:-1}[self.dim]
        return Keras_Concatenate(dim)(xs)


from keras.layers import ReLU as Keras_ReLU
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
        return Keras_ReLU(self.max_val)(x)


from torch.nn import BatchNorm2d as Torch_Batchnorm
from keras.layers import BatchNormalization as Keras_Batchnorm
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
        batchnorm = Keras_Batchnorm(momentum=self.momentum,
                                    epsilon=self.epsilon)
        batchnorm.build(x.shape)
        weights      = self.batchnorm.weight.detach().numpy()
        bias         = self.batchnorm.bias.detach().numpy()
        running_mean = self.batchnorm.running_mean.detach().numpy()
        running_var  = self.batchnorm.running_var.detach().numpy()
        batchnorm.set_weights([weights, bias, running_mean, running_var])
        return batchnorm(x, training=askeras.kwds["train"])
