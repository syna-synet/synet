from importlib import import_module
from os.path import join, dirname
from sys import argv

mode = argv.pop(1)
chip = argv.pop(1)
assert mode in ("train", "quantize")
if mode == 'train':
    framework = argv.pop(1)
    if framework == "yolov5":
        from yolov5_patches import patch_yolov5
        from yolov5.train import run
        patch_yolov5(chip)
        run()
    else:
        from sys import path
        path.insert(0, join(dirname(__file__), "custom"))
        from train import run
        from custom_patches import patch_custom
        patch_custom(chip)
        run()
else:
    import_module(mode).main(chip)
