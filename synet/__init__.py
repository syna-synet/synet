from .backends import get_backend


__all__ = "backends", "base", "katana", "sabre", "quantize", "test", \
    "metrics", "tflite_utils"


def get_model(model_path, backend, *args, **kwds):
    """Method to get the model.  For now, only the katananet model is
supported in ultralytics format."""

    print("loading", model_path)

    backend = get_backend(backend)
    return backend.get_model(model_path, *args, **kwds)
