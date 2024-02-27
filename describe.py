#!/usr/bin/env python3

import sys
from utils import load_csv
import polars as pl

label_column = {"statistic": ["count", "mean"]}


def apply_count(column: pl.Series):
    count = 0
    for row in column:
        count += 1
    return count


def apply_mean(column: pl.Series):
    count = apply_count(column)
    return column.sum() / count


def describe(df: pl.DataFrame):
    numeric_features = df.select(pl.selectors.numeric())
    describe_df = pl.DataFrame(label_column)

    for column in numeric_features:
        column = column.drop_nans()
        column = column.drop_nulls()

        count_value = apply_count(column)
        mean_value = apply_mean(column)

        temp_df = pl.DataFrame(label_column | {column.name: [count_value, mean_value]})
        describe_df = describe_df.with_columns(temp_df)

    print(describe_df)


if __name__ == "__main__":
    dataset_pathname = sys.argv[1]
    df = load_csv(dataset_pathname)
    describe(df)
    print(df.select(pl.selectors.numeric()).describe())
