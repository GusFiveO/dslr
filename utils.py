import polars as pl


def load_csv(pathname: str):
    try:
        content = pl.read_csv(pathname)
        return content
    except Exception as e:
        print(e)
        return None
