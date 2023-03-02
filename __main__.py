from importlib import import_module
from sys import argv

mode, chip = argv[1:3]
argv = argv[2:]
assert mode in ("train", "quantize", "check")
if mode != "check": assert chip in ("katana")
import_module(mode).main(chip)
