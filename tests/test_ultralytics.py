from os import chdir  # , listdir
# from os.path import join, dirname

from synet.backends import get_backend
from synet.quantize import main


backend = get_backend("ultralytics")


def test_quantize(tmp_path):
    chdir(tmp_path)
    for config in backend.get_configs():
        main(("--backend=ultralytics",
              "--model="+config,
              "--number=1"))
    main(("--backend=ultralytics",
          "--model="+config,
          "--number=1",
          "--image-shape", "321", "319"))
