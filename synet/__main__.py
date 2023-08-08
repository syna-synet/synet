from importlib import import_module
from sys import argv, exit

import synet


def main():
    if argv[1] in synet.__all__:
        return import_module(f"synet.{argv.pop(1)}").main()
    return import_module(f"synet.backends.{argv.pop(1)}").main()


if __name__ == "__main__":
    exit(main())
