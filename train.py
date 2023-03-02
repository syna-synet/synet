from yolov5_patches import patch_yolo
from yolov5.train import run

def main(chip):
    patch_yolo(chip)
    run()
