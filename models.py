from torch.nn import Module
from torch.nn.init import uniform_


t_actv_to_k = lambda actv: actv.detach().numpy().transpose(0, 2, 3, 1)
t_conv_wght_to_k = lambda wght: wght.detach().numpy().transpose(2, 3, 1, 0)


from torch.nn import Conv2d as Torch_Conv2d
from torch.nn.functional import pad
from keras.layers import Conv2D as Keras_Conv2d
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        assert (kernel_size, stride) in [(1, 1), (3, 1), (3, 2)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = Torch_Conv2d(in_channels  = in_channels,
                                 out_channels = out_channels,
                                 kernel_size  = kernel_size,
                                 bias         = False,
                                 stride       = stride)

    def forward(self, x):
        # make padding like in tensorflow, which right aligns convolutions.
        if self.kernel_size == 3:
            if self.stride == 1:
                x = pad(x, (1,1,1,1))
            elif self.stride == 2:
                x = pad(x, (x.shape[-1]%2, 1, x.shape[-2]%2, 1))
        return self.conv(x)

    def rand_init(self):
        uniform_(self.conv.weight)

    def as_keras(self, x):
        assert x.shape[-1] == self.in_channels
        conv = Keras_Conv2d(filters     = self.out_channels,
                            kernel_size = self.kernel_size,
                            strides     = self.stride,
                            padding     = "same",
                            use_bias    = False)
        conv.build(x.shape[1:])
        conv.set_weights([t_conv_wght_to_k(self.conv.weight)])
        return conv(x)


from keras.layers import ReLU as Keras_ReLU
from torch.nn import ReLU as Torch_ReLU
from torch import minimum, tensor
class ReLU(Module):
    def __init__(self, max_val=None):
        super().__init__()
        self.max_val = None if max_val is None else tensor(max_val,dtype=float)
        self.relu = Torch_ReLU()

    def forward(self, x):
        if self.max_val is None:
            return self.relu(x)
        return minimum(self.relu(x), self.max_val)

    def rand_init(self):
        pass

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
        return self.batchnorm(x)

    def rand_init(self):
        uniform_(self.batchnorm.weight)
        uniform_(self.batchnorm.bias)
        uniform_(self.batchnorm.running_mean)
        uniform_(self.batchnorm.running_var)

    def as_keras(self, x):
        batchnorm = Keras_Batchnorm(momentum=self.momentum,
                                    epsilon=self.epsilon)
        batchnorm.build(x.shape)
        weights      = self.batchnorm.weight.detach().numpy()
        bias         = self.batchnorm.bias.detach().numpy()
        running_mean = self.batchnorm.running_mean.detach().numpy()
        running_var  = self.batchnorm.running_var.detach().numpy()
        batchnorm.set_weights([weights, bias, running_mean, running_var])
        return batchnorm(x, training=True)


from torch.nn import ModuleList as Torch_Modulelist
class Sequential(Module):
    def __init__(self, sequence):
        super().__init__()
        self.ml = Torch_Modulelist(sequence)

    def forward(self, x):
        for layer in self.ml:
            x = layer(x)
        return x

    def rand_init(self):
        for layer in self.ml:
            layer.rand_init()

    def as_keras(self, x):
        for layer in self.ml:
            x = layer.as_keras(x)
        return x

class InvertedResidule(Module):
    def __init__(self, in_channels, expansion_factor,
                 out_channels=None, stride=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.layers = Sequential([
            Conv2d(in_channels,
                   out_channels = in_channels * expansion_factor,
                   kernel_size  = 3,
                   stride       = stride),
            BatchNorm(in_channels * expansion_factor),
            ReLU(.6),
            Conv2d(in_channels  = in_channels * expansion_factor,
                   out_channels = out_channels,
                   kernel_size  = 1),
            BatchNorm(out_channels)
        ])
        self.stride = stride

    def forward(self, x):
        y = self.layers(x)
        if self.stride == 1:
            return x + y
        if self.stride == 2:
            return y

    def rand_init(self):
        self.layers.rand_init()

    def as_keras(self, x):
        y = self.layers.as_keras(x)
        if self.stride == 1:
            return x + y
        if self.stride == 2:
            return y


class Head(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = Relu(.6)
        self.model = Sequential([
            Sequential([Conv2d(out_channels if i else in_channels,
                               out_channels,
                               3),
                        self.relu])
            for i in range(4)
        ])

    def forward(self, x):
        return self.module(x)

    def rand_init(self):
        self.model.rand_init()

    def as_keras(self, x):
        return self.model.as_keras(x)
    
from torch.nn import ModuleDict
class Backbone(Module):
    def __init__(self, levels=range(1,6)):
        super().__init__()
        self.levels = levels
        self.relu = ReLU(.6)
        self.layers = ModuleDict(dict(
            c1=Sequential([Conv2d(in_channels=1, out_channels=4,
                                  kernel_size=3, stride=2),
                           self.relu]),
            c2=Sequential([InvertedResidule(in_channels=4,
                                            expansion_factor=4,
                                            out_channels=8,
                                            stride=2),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=2)]),
            c3=Sequential([InvertedResidule(in_channels=8,
                                            expansion_factor=6,
                                            stride=2),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=6),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=6)]),
            c4=Sequential([InvertedResidule(in_channels=8,
                                            expansion_factor=6,
                                            stride=2),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=6),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=6),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=6),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=6),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=6),
                           InvertedResidule(in_channels=8,
                                            expansion_factor=6)]),
            c5=Sequential([InvertedResidule(in_channels=8,
                                               expansion_factor=6,
                                               stride=2),
                              InvertedResidule(in_channels=8,
                                               expansion_factor=6),
                              InvertedResidule(in_channels=8,
                                               expansion_factor=6),
                              Conv2d(8, 48, 3),
                              self.relu,
                              Conv2d(48, 16, 1)])
        ))

    def forward(self, x):
        output = {}
        for i in range(1, max(self.levels) + 1):
            x = self.layers[f"c{i}"](x)
            if i in self.levels:
                output[f"c{i}"] = x
        if len(self.levels) == 1:
            return output[f"c{self.levels[0]}"]
        return output

    def rand_init(self):
        for v in self.layers.values():
            v.rand_init()

    def as_keras(self, x):
        output = {}
        for i in range(1, max(self.levels) + 1):
            x = self.layers[f"c{i}"].as_keras(x)
            if i in self.levels:
                output[f"c{i}"] = x
        if len(self.levels) == 1:
            return output[f"c{self.levels[0]}"]
        return output
