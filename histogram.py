#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import polars as pl
import numpy as np
import sys
from describe import apply_mean, apply_std, describe

from utils import load_csv, print_histogram_usage

matplotlib.use("TkAgg")


def z_score_normalize(column: pl.Series):
    normalized_array = (column - apply_mean(column)) / apply_std(column)
    return normalized_array


def plot_histogram(df: pl.DataFrame):
    ROW = 4
    COLUMN = 4
    fig, axs = plt.subplots(ROW, COLUMN, layout="constrained")
    slytherin_df = df.filter(pl.col("Hogwarts House") == "Slytherin")
    gryffindor_df = df.filter(pl.col("Hogwarts House") == "Gryffindor")
    ravenclaw_df = df.filter(pl.col("Hogwarts House") == "Ravenclaw")
    hufflepuff_df = df.filter(pl.col("Hogwarts House") == "Hufflepuff")
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
    return None


if __name__ == "__main__":
    try:
        dataset_pathname = sys.argv[1]
    except Exception:
        print_histogram_usage()
        exit()
    df = load_csv(dataset_pathname)
    if df is None:
        exit()
    plot_histogram(df)
