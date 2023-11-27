from torch import rand

from synet.base import Conv2d


def test_reshape_conv():
    from synet.demosaic import reshape_conv
    conv = Conv2d(3, 13, 3, 2)
    reshaped_conv = reshape_conv(conv)
    inp = rand(3, 480, 640)
    reshaped_inp = inp.reshape(3, 240, 2, 320, 2).permute(2, 4, 0, 1, 3).reshape(12, 240, 320)
    assert (reshaped_conv(reshaped_inp) - conv(inp)).abs().max() < 1e-4
