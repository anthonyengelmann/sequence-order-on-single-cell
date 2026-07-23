import argparse
import glob
import os
from lyra_lite.analysis.multiseed import aggregate_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or max(glob.glob("outputs/multirun/*/*"), key=os.path.getmtime)
    aggregate_sweep(run_dir)


if __name__ == "__main__":
    main()