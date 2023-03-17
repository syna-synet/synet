# parse arguments
from argparse import ArgumentParser
from importlib import import_module
from sys import argv
parser = ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--cfg")
parser.add_argument("--weights")
parser.add_argument("--imgsz")
args = parser.parse_known_args()[0]
argv.remove(args.mode)
src = 'synet' if args.mode in ('quantize', 'test', 'data_subset') else 'yolov5'
module = import_module(f'{src}.{args.mode}')


# extract model defaults
from yaml import safe_load
from torch import load, device
from .zoo import find_model_path
from .yolov5_patches import patch_yolov5
if args.cfg:
    args.cfg = find_model_path(args.cfg)
    yaml = safe_load(open(args.cfg))
    patch_yolov5(yaml['chip'])
elif args.weights:
    yaml = load(args.weights, map_location=device("cpu"))['model'].yaml
    patch_yolov5(yaml['chip'])

# override opt
opt = module.parse_opt()
if args.cfg: opt.cfg = args.cfg
if hasattr(opt, 'image_shape'): opt.image_shape = yaml['image_shape']
if hasattr(opt, 'imgsz') and not args.imgsz:
    opt.imgsz = max(yaml['image_shape'])

# run function
module.run(**vars(opt))
