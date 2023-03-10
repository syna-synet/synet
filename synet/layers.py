from torch.nn import Module

from base import ReLU, BatchNorm, Conv2d, askeras


from torch.nn import ModuleList as Torch_Modulelist
class Sequential(Module):
    def __init__(self, sequence):
        super().__init__()
        self.ml = Torch_Modulelist(sequence)

    def forward(self, x):
        for layer in self.ml:
            x = layer(x)
        return x


from base import BatchNorm
class InvertedResidual(Module):
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
            ReLU(6),
            Conv2d(in_channels  = in_channels * expansion_factor,
                   out_channels = out_channels,
                   kernel_size  = 1),
            BatchNorm(out_channels)
        ])
        self.stride = stride
        self.cheq = in_channels == out_channels
        assert self.stride in (1, 2)

    def forward(self, x):
        y = self.layers(x)
        if self.stride == 1 and self.cheq:
            return x + y
        if self.stride == 2 or (self.stride == 1 and not self.cheq):
            return y


class Head(Module):
    def __init__(self, in_channels, out_channels, num=4):
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
