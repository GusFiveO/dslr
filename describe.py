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
    apply_variance,
    apply_skewness,
    apply_kurtosis,
)
from utils import load_csv, load_pandas_csv, print_describe_usage
import numpy as np
import polars as pl

label_column = {
    "statistic": ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "bonus_variance", "bonus_skewness", "bonus_kurtosis"]
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
        bonus_variance = apply_variance(column)
        bonus_skewness = apply_skewness(column)
        bonus_kurtosis = apply_kurtosis(column)
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
                    bonus_variance,
                    bonus_skewness,
                    bonus_kurtosis,
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
    try:
        df = load_csv(dataset_pathname)
        if df is None:
            exit()
        df_pd = load_pandas_csv(dataset_pathname)
        describe(df)
        print(df_pd.select_dtypes(include=np.number).describe())
    except Exception as e:
        print(e)
        exit()
