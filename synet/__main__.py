from importlib import import_module
from os.path import join, dirname, isfile
from sys import argv
from argparse import ArgumentParser
from .zoo import find_model_path


parser = ArgumentParser()
parser.add_argument("mode")
parser.add_argument("chip")
parser.add_argument("--cfg")
args = parser.parse_known_args()[0]


assert args.mode in ("train", "quantize")
if args.mode == 'train':
    from .yolov5_patches import patch_yolov5
    from yolov5.train import run
    patch_yolov5(args.chip)
    run(cfg=find_model_path(args.chip, args.cfg))
else:
    import_module(f".{args.mode}", "synet").main(args.chip)
