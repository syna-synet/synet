"""asymetric.py diverges from base.py and layers.py in that its core
assumption is switched.  In base.py/layers.py, the output of a module
in keras vs torch is identical, while asymetric modules act as the
identity function in keras.  To get non-identity behavior in keras
mode, you should call module.clf().  'clf' should be read as 'channels
last forward'; such methods take in and return a channels-last numpy
array.

The main use case for these modules is for uniform preprocessing to
bridge the gap between 'standard' training scenarios and actual
execution environments.  So far, the main examples implemented are
conversions to grayscale, bayer, and camera augmented images.  This
way, you can train your model on a standard RGB pipeline.  The
resulting tflite model will not have these extra layers, and is ready
to operate on the raw input at deployment.

The cfl methods are mainly used for python demos where the sensor
still needs to be simulated, but not included in the model.

"""

from os.path import join, dirname
from cv2 import GaussianBlur as cv2GaussianBlur
from numpy import array, interp, ndarray
from numpy.random import normal
from torch import empty, tensor
from torchvision.transforms import GaussianBlur

from .demosaic import Demosaic, UnfoldedDemosaic, Mosaic
from .base import askeras, Module


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


class Camera(Module):
    def __init__(self,
                 color_cal=join(dirname(__file__), 'camcal.csv'),
                 bayer_pattern='gbrg',
                 from_bayer=False,
                 to_bayer=False,
                 blur_sigma=0.4,
                 noise_sigma=10):
        super().__init__()
        self.mosaic = Mosaic(bayer_pattern)
        self.demosaic = UnfoldedDemosaic('malvar', bayer_pattern)
        self.blur_sigma = blur_sigma
        self.noise_sigma = noise_sigma
        self.from_bayer = from_bayer
        self.to_bayer = to_bayer
        if isinstance(color_cal, str):
            with open(color_cal) as f:
                color_cal = [[float(val) for val in line.split(',')]
                             for line in f.read().split()[1:]]
        # yp = [rp, gp, bp], rgb sample points
        self.xp, *self.yp = array(color_cal).T
        self.blur = GaussianBlur(3, blur_sigma)

    def interp(self, x, xp, yp):
        if isinstance(x, ndarray):
            return interp(x, xp, yp)
        return tensor(interp(x.cpu(), xp, yp)).to(x.device)

    def map_to_linear(self, image):
        for yoff, xoff, chan in zip(self.mosaic.rows,
                                    self.mosaic.cols,
                                    self.mosaic.bayer_pattern):
            # the gamma correction (from experiments) is channel dependent
            image[..., yoff::2, xoff::2] = self.interp(image[...,
                                                             yoff::2,
                                                             xoff::2],
                                                       self.xp,
                                                       self.yp[chan])
        return image

    def forward(self, im, normalized=True):
        if askeras.use_keras:
            return im
        if normalized:
            im *= 255
        if not self.from_bayer:
            im = self.mosaic(im)
        this_noise_sigma, = empty(1).normal_(self.noise_sigma, 2)
        im = self.map_to_linear(self.blur(im))
        im += empty(im.shape, device=im.device).normal_(0.0, this_noise_sigma)
        if not self.to_bayer:
            im = self.demosaic(im)
        if normalized:
            im /= 255
        return im.clip(0, 1 if normalized else 255)

    def clf(self, im):
        # augmentation should always be done on bayer image.
        if not self.from_bayer:
            im = self.mosaic.clf(im)
        # let the noise level vary
        this_noise_sigma = normal(self.noise_sigma, 2)
        # if you blur too much, the image becomes grayscale
        im = cv2GaussianBlur(im, [3, 3], self.blur_sigma)
        im = self.map_to_linear(im)
        # GaussianBlur likes to remove singleton channel dimension
        im = im[..., None] + normal(0.0, this_noise_sigma, im.shape + (1,))
        # depending on scenario, you may not want to return an RGB image.
        if not self.to_bayer:
            im = self.demosaic.clf(im)
        return im.clip(0, 255)
