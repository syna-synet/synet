from argparse import ArgumentParser
from importlib import import_module
from sys import argv
from .zoo import find_model_path


# parse arguments
parser = ArgumentParser()
parser.add_argument("mode")
parser.add_argument("chip")
parser.add_argument("--cfg")
args = parser.parse_known_args()[0]
argv.remove(args.mode)
argv.remove(args.chip)


# run synet functions
if args.mode in ("quantize", "test"):
    import_module(f"synet.{args.mode}").main(args.chip)
    exit()


# run yolov5 functions
from .yolov5_patches import patch_yolov5
patch_yolov5(args.chip)
module = import_module(f"yolov5.{args.mode}")
opt = module.parse_opt()
if args.cfg: opt.cfg = find_model_path(args.chip, args.cfg)
print("foo")
module.main(opt)
