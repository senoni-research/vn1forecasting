# vn1forecasting/__main__.py
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="vn1forecasting CLI")
    parser.add_argument("--version", action="store_true", help="Show the version")
    # Filter out pytest arguments when running tests
    args = parser.parse_args(args=None if sys.argv[0].endswith("pytest") else sys.argv[1:])

    if args.version:
        print("vn1forecasting v0.1.0")
    else:
        print("Welcome to vn1forecasting!")


if __name__ == "__main__":
    main()
