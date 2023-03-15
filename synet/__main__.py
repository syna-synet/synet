from argparse import ArgumentParser
from importlib import import_module
import sys
from .zoo import find_model_path


# parse arguments
parser = ArgumentParser()
parser.add_argument("mode")
parser.add_argument("chip")
parser.add_argument("--cfg")
args = parser.parse_known_args()[0]
sys.argv = sys.argv[2:]


# run synet functions
if args.mode in ("quantize", "test"):
    import_module(f"synet.{args.mode}").main(args.chip)
    exit()


# run yolov5 functions
from .yolov5_patches import patch_yolov5
patch_yolov5(args.chip)
module = import_module(f"yolov5.{args.mode}")
kwds = vars(module.parse_opt())
if args.cfg: kwds['cfg'] = find_model_path(args.chip, args.cfg)
module.run(**kwds)
