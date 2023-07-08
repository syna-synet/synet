from .base import askeras

from os.path import dirname, join, exists
from .zoo import find_model_path

from importlib import import_module
def get_model(model_path, raw=True, backend=None, chip=None, **kwds):
    """Method to get the model.  For now, only the katananet model is
supported in legacy and yolov5 format."""

    print("loading", model_path)

    if exists(join(dirname(model_path), "anchors.json")):
        from .legacy import get_katananet_model
        print("LEGACY KATANANET LOADING")
        return get_katananet_model(model_path, **kwds)

    if chip is not None:
        assert backend is not None
        getattr(import_module(f"{backend}_patches"), f"patch_{backend}"
                )(chip)

    if backend is not None:
        return getattr(import_module(f"{backend}_patches"),
                       f"get_{backend}_model"
                       )(find_model_path(model_path), raw=raw, **kwds)

    return get_model_backend(find_model_path(model_path), raw=raw, **kwds)
