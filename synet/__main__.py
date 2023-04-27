from argparse import ArgumentParser
from sys import argv
def parse_opt():
    parser = ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--imgsz")
    parser.add_argument("--image-shape", nargs=2, type=int)
    args = parser.parse_known_args()[0]
    # ensure argv can be parsed by downstream parser
    argv.remove(args.mode)
    return args


from yaml import safe_load
from torch import load, device
from .zoo import find_model_path
from .yolov5_patches import patch_yolov5
def interpret_model_enable_chip(args):
    if args.cfg:
        # update args.cfg if it is specified from the zoo
        args.cfg = find_model_path(args.cfg)
        # enable chip
        yaml = safe_load(open(args.cfg))
        patch_yolov5(yaml['chip'])
    elif args.weights and not args.weights.endswith(".tflite"):
        # enable chip
        yaml = load(args.weights, map_location=device("cpu"))['model'].yaml
        patch_yolov5(yaml['chip'])
    else:
        # not running model, no chip enabled
        patch_yolov5()
        return
    return yaml

def opt_override(module, args, yaml):
    # parse opt according to module
    opt = module.parse_opt()
    # apply possibly updated cfg
    if args.cfg: opt.cfg = args.cfg
    # obtain image_shape from model yaml
    if hasattr(opt, 'image_shape') and not args.image_shape:
        opt.image_shape = yaml['image_shape']
    # if imgsz is not specified, overwrite with model's imgsz
    if hasattr(opt, 'imgsz') and not args.imgsz \
       and not (args.weights is not None and args.weights.endswith(".tflite")):
        opt.imgsz = max(yaml['image_shape'])
    return opt

from importlib import import_module
def main():
    args = parse_opt()
    # only try to grab use mode from synet if it exists.  Otherwise, use yolov5
    src = 'synet' if args.mode in ('quantize', 'test', 'data_subset', 'metrics'
                                   ) else 'yolov5'
    # select the mode
    module = import_module(f'{src}.{args.mode}')
    yaml = interpret_model_enable_chip(args)
    opt = opt_override(module, args, yaml)

    # run function
    module.run(**vars(opt))

if __name__ == "__main__":
    import sys
    sys.exit(main())
