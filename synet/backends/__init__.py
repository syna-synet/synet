from importlib import import_module


class Backend:
    def get_model(self, pth):
        """Load model from config, or pretrained save."""
        raise NotImplemented("Please subclass and implement")
    def get_shape(self, model):
        """Get shape of model."""
        raise NotImplemented("Please subclass and implement")
    def get_chip(self, model):
        """Get chip of model."""
        raise NotImplemented("Please subclass and implement")


def get_backend(name):
    return import_module(f".{name}", __name__).Backend()
