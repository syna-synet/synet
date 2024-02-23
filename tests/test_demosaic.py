from torch import rand
from torch.nn.init import uniform_

from synet.base import Conv2d
from synet.demosaic import reshape_conv


def test_reshape_conv():
    conv = Conv2d(3, 13, 3, 2)
    for param in conv.parameters():
        uniform_(param, -1)
    reshaped_conv = reshape_conv(conv)
    inp = rand(3, 480, 640)
    reshaped_inp = inp.reshape(3, 240, 2, 320, 2
                               ).permute(2, 4, 0, 1, 3
                                         ).reshape(12, 240, 320)
    assert (reshaped_conv(reshaped_inp) - conv(inp)).abs().max() < 1e-5
