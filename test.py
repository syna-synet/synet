#!/usr/bin/env python
from numpy import absolute
def test_layer(layer, torch_inp):
    out1 = t_actv_to_k(layer(torch_inp))
    out2 = layer.as_keras(t_actv_to_k(torch_inp))
    assert all(s1 == s2 for s1, s2 in zip(out1.shape, out2.shape))
    max_diff = absolute(out1 - out2).max()
    print("max diff:", max_diff)
    return max_diff


from torch import rand
def test_sizes(layer, batch_size, in_channels, shapes):
    return max(test_layer(layer, rand(batch_size, in_channels, *shape) * 2 - 1)
               for shape in shapes)


from models import Conv2d, ReLU, BatchNorm, InvertedResidule, Backbone, t_actv_to_k
def run_tests():
    print("running tests")
    batch_size = 2
    in_channels = 5
    out_channels = 7
    expansion_factor = 11
    shapes = [(i, i) for i in range(5,10)]

    print("testing Conv2d")
    for kernel, stride in ((1, 1), (3, 1), (3, 2)):
        print("kernel, stride", kernel, stride)
        tconv = Conv2d(in_channels, out_channels, kernel, stride)
        tconv.rand_init()
        print("max diff:", test_sizes(tconv, batch_size, in_channels, shapes))

    print("testing ReLU")
    relu = ReLU(.6)
    relu.rand_init()
    print("max diff:", test_sizes(relu, batch_size, in_channels, shapes))

    print("testing BatchNorm")
    batchnorm = BatchNorm(in_channels)
    batchnorm.rand_init()
    print("max diff:", test_sizes(batchnorm, batch_size, in_channels, shapes))

    for stride in 1, 2:
        print(f"testing InvRes (stride={stride})")
        invres = InvertedResidule(in_channels, expansion_factor, stride=stride)
        invres.rand_init()
        print("max diff:", test_sizes(invres, batch_size, in_channels, shapes))

    print("testing backbone")
    backbone = Backbone(levels=[5])
    backbone.rand_init()
    print("max diff", test_sizes(backbone, batch_size, in_channels=1,
                                 shapes=shapes))


if __name__ == "__main__":
    run_tests()
