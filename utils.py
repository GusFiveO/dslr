import polars as pl
import pandas as pd


def print_describe_usage():
    print("usage: ./describe.py <filepath>")


def print_histogram_usage():
    print("usage: ./histogram.py <filepath>")


def load_csv(pathname: str):
    try:
        content = pl.read_csv(pathname)
        return content
    except Exception as e:
        print(e)
        return None


def load_pandas_csv(pathname: str):
    try:
        content = pd.read_csv(pathname)
        return content
    except Exception as e:
        print(e)
        return None
