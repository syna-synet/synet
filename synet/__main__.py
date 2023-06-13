from argparse import ArgumentParser
from sys import argv
def parse_opt():
    parser = ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("--cfg", "--model")
    parser.add_argument("--weights")
    parser.add_argument("--imgsz")
    parser.add_argument("-u", action="store_true")
    parser.add_argument("--image-shape", nargs=2, type=int)
    args = parser.parse_known_args()[0]
    # ensure argv can be parsed by downstream parser
    argv.remove(args.mode)
    if args.u: argv.remove("-u")
    return args


from yaml import safe_load
from torch import load, device
from .zoo import find_model_path
def interpret_model_enable_chip(args):
    if args.u:
        from .ultralytics_patches import patch_ultralytics as patch
    else:
        from .yolov5_patches import patch_yolov5 as patch

    if args.cfg and args.cfg.endswith("ml"): # only .yaml or .yml
        # update args.cfg if it is specified from the zoo
        args.cfg = find_model_path(args.cfg)
        # enable chip
        yaml = safe_load(open(args.cfg))
        patch(yaml['chip'])
    elif args.cfg or args.weights and not args.weights.endswith(".tflite"):
        # enable chip
        yaml = load(args.weights or args.cfg,
                    map_location=device("cpu"))['model'].yaml
        patch(yaml['chip'])
    else:
        # not running model, no chip enabled
        patch()
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
    if args.mode in ('quantize', 'test', 'data_subset', 'metrics'):
        module = import_module(f'synet.{args.mode}')
    elif args.u: # ultralytics
        from ultralytics.yolo.cfg import entrypoint
        yaml = interpret_model_enable_chip(args)

        # run function
        return entrypoint()
    else: # yolov5
        module = import_module(f'yolov5.{args.mode}')

    yaml = interpret_model_enable_chip(args)
    opt = opt_override(module, args, yaml)

    # run function
    return module.run(**vars(opt))


if __name__ == "__main__":
    import sys
    sys.exit(main())
