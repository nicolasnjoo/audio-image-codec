import argparse
from . import __version__


def main() -> int:
    p = argparse.ArgumentParser(prog="codec")
    p.add_argument("--version", action="store_true")
    args = p.parse_args()

    if args.version:
        print(__version__)
        return 0

    print("codec: not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
