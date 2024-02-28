import polars as pl
import pandas as pd


def print_usage():
    print("usage: ./describe.py <filepath>")


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
