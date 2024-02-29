#!/usr/bin/env python3

import sys
from ft_statistics import (
    apply_count,
    apply_first_quartile,
    apply_last_quartile,
    apply_max,
    apply_mean,
    apply_median,
    apply_min,
    apply_std,
)
from utils import load_csv, load_pandas_csv, print_describe_usage
import numpy as np
import polars as pl

label_column = {
    "statistic": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
}


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
        median_value = apply_median(column)
        last_quartile_value = apply_last_quartile(column)
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
    try:
        dataset_pathname = sys.argv[1]
    except Exception:
        print_describe_usage()
        exit()
    df = load_csv(dataset_pathname)
    if df is None:
        exit()
    df_pd = load_pandas_csv(dataset_pathname)
    describe(df)
    print(df_pd.select_dtypes(include=np.number).describe())
