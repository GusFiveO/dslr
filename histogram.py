#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import polars as pl
import numpy as np
import sys
from describe import apply_mean, apply_std, describe

from utils import load_csv, load_pandas_csv, print_histogram_usage

matplotlib.use("TkAgg")


def plot_score_distribution(df):
    ROW = 4
    COLUMN = 4
    fig, axs = plt.subplots(ROW, COLUMN, layout="constrained")
    slytherin_df = df[df["Hogwarts House"] == "Slytherin"]
    gryffindor_df = df[df["Hogwarts House"] == "Gryffindor"]
    ravenclaw_df = df[df["Hogwarts House"] == "Ravenclaw"]
    hufflepuff_df = df[df["Hogwarts House"] == "Hufflepuff"]
    course_name_list = [
        "Arithmancy",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying",
    ]

    for idx, course_name in enumerate(course_name_list):
        row = idx // ROW
        col = idx % ROW
        axs[col, row].set_title(
            f"{course_name[:10]}{'...' if len(course_name) > 10 else ''}"
        )
        axs[col, row].hist(gryffindor_df[course_name], alpha=0.5)
        axs[col, row].hist(slytherin_df[course_name], alpha=0.5)
        axs[col, row].hist(ravenclaw_df[course_name], alpha=0.5)
        axs[col, row].hist(hufflepuff_df[course_name], alpha=0.5)
    for i in range(13, 16):
        row = i // ROW
        col = i % ROW
        axs[col, row].axis("off")

    plt.show()


def z_score_normalize(column):
    mean = apply_mean(column)
    std = apply_std(column)
    normalized_column = (column - mean) / std
    return normalized_column


def plot_homogeneous_distribution_course(df: pl.DataFrame):

    course_name_list = [
        "Arithmancy",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying",
    ]
    course_columns = df[course_name_list]
    normalized_df = course_columns.apply(z_score_normalize)
    normalized_df["Hogwarts House"] = df["Hogwarts House"]

    print(normalized_df.groupby("Hogwarts House").apply(lambda x: x.apply(apply_mean)))
    normalized_df = normalized_df.groupby("Hogwarts House").apply(
        lambda x: x.apply(apply_mean)
    )

    print(normalized_df)
    min_std = None
    best = None
    dict_best = dict()
    for column in normalized_df:
        tmp_std = apply_std(normalized_df[column])
        dict_best[column] = tmp_std
        if min_std is None or tmp_std < min_std:
            min_std = tmp_std
            best = column
    print(best)
    print(dict_best)
    return None


if __name__ == "__main__":
    try:
        dataset_pathname = sys.argv[1]
    except Exception:
        print_histogram_usage()
        exit()
    df = load_pandas_csv(dataset_pathname)
    if df is None:
        exit()
    df = df.dropna()
    plot_score_distribution(df)
    plot_homogeneous_distribution_course(df)
