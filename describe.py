#!/usr/bin/env python3

import sys
from utils import load_csv


def describe(pathname: str):
    data = load_csv(pathname)
    print(data)


if __name__ == "__main__":
    dataset_pathname = sys.argv[1]
    describe(dataset_pathname)
