from os.path import dirname, join, exists
def get_model(model_path, input_shape, low_thld, **kwds):
    """Method to get the model.  For now, only the katananet model is
    supported."""
    print("loading", model_path)
    if exists(join(dirname(model_path), "anchors.json")):
        from .legacy import get_katananet_model
        print("LEGACY KATANANET LOADING")
        return get_katananet_model(model_path, input_shape, low_thld, **kwds)
    else:
        print("NEW LOADING")
        from .yolov5 import get_yolov5_model, yolov5fix
        yolov5fix()
        return get_yolov5_model(model_path, input_shape, low_thld, **kwds)
