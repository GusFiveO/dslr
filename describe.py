#!/usr/bin/env python3

import sys
from utils import load_csv, load_pandas_csv
import numpy as np
import polars as pl

label_column = {
    "statistic": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
}


def apply_count(column: pl.Series):
    count = 0
    for row in column:
        count += 1
    return count


def apply_min(column: pl.Series):
    min = None
    for row in column:
        if min is None or row < min:
            min = row
    return min


def apply_max(column: pl.Series):
    max = None
    for row in column:
        if max is None or row > max:
            max = row
    return max


def apply_mean(column: pl.Series):
    count = apply_count(column)
    return column.sum() / count


def apply_std(column: pl.Series):
    square_diff = 0
    count = 0
    mean = apply_mean(column)
    for row in column:
        square_diff += np.square(row - mean)
        count += 1
    return np.sqrt(square_diff / (count - 1))


def apply_first_quartile(column: pl.Series):
    sorted_column = column.sort()
    count = apply_count(column)
    return sorted_column[count // 4]


def apply_median(column: pl.Series):
    sorted_column = column.sort()
    count = apply_count(column)
    index = count // 2
    if count % 2 == 0:
        return (sorted_column[index - 1] + sorted_column[index]) / 2
    return sorted_column[index]


def apply_last_quartile(column: pl.Series):
    sorted_column = column.sort()
    count = apply_count(column)
    last_index = count - 1
    index = last_index - (count // 4)
    return sorted_column[index]


import math


def apply_percentile(column: pl.Series, percent: int):
    sorted_column = column.sort()
    count = apply_count(column)
    if count % 2:
        index = math.ceil(percent / 100 * count)
    index = math.floor(percent / 100 * count)
    return sorted_column[index]


def describe(df: pl.DataFrame):
    numeric_features = df.select(pl.selectors.numeric())
    describe_df = pl.DataFrame(label_column)

    for column in numeric_features:
        column = column.drop_nans()
        column = column.drop_nulls()

        count_value = apply_count(column)
        mean_value = apply_mean(column)
        std_value = apply_std(column)
        min_value = apply_min(column)
        first_quartile_value = apply_first_quartile(column)
        # first_quartile_value = apply_percentile(column, 25)
        median_value = apply_median(column)
        # median_value = apply_percentile(column, 50)
        last_quartile_value = apply_last_quartile(column)
        # last_quartile_value = apply_percentile(column, 75)
        max_value = apply_max(column)

        temp_df = pl.DataFrame(
            label_column
            | {
                column.name: [
                    count_value,
                    mean_value,
                    std_value,
                    min_value,
                    first_quartile_value,
                    median_value,
                    last_quartile_value,
                    max_value,
                ]
            }
        )
        describe_df = describe_df.with_columns(temp_df)

    print(describe_df)


if __name__ == "__main__":
    dataset_pathname = sys.argv[1]
    df = load_csv(dataset_pathname)
    df_pd = load_pandas_csv(dataset_pathname)
    describe(df)
    print(df.select(pl.selectors.numeric()).describe())
    # print(df_pd.select_dtypes(include=np.number).describe())
