from .base import askeras

from os.path import dirname, join, exists
from .zoo import find_model_path
def get_model(model_path, raw=True, **kwds):
    """Method to get the model.  For now, only the katananet model is
supported in legacy and yolov5 format."""
    print("loading", model_path)
    if exists(join(dirname(model_path), "anchors.json")):
        from .legacy import get_katananet_model
        print("LEGACY KATANANET LOADING")
        return get_katananet_model(model_path, **kwds)
    else:
        from .yolov5_patches import get_yolov5_model, patch_yolov5
        patch_yolov5('katana')
        return get_yolov5_model(find_model_path(model_path), raw=raw,
                                **kwds)
