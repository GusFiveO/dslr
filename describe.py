#!/usr/bin/env python3

import polars as pl
import sys


def load_csv(pathname: str):
    try:
        content = pl.read_csv(pathname)
        return content
    except Exception as e:
        print(e)
        return None


def describe(pathname: str):
    data = load_csv(pathname)
    print(data)


if __name__ == "__main__":
    dataset_pathname = sys.argv[1]
    describe(dataset_pathname)
