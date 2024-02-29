#!/usr/bin/env python3

import operator
import matplotlib.pyplot as plt
import matplotlib
from describe import apply_mean, apply_std

from utils import load_pandas_csv, z_score_normalize

matplotlib.use("TkAgg")


def plot_score_distribution(df, sorted_course_names):
    ROW = 4
    COLUMN = 4
    _, axs = plt.subplots(ROW, COLUMN, layout="constrained")
    slytherin_df = df[df["Hogwarts House"] == "Slytherin"]
    gryffindor_df = df[df["Hogwarts House"] == "Gryffindor"]
    ravenclaw_df = df[df["Hogwarts House"] == "Ravenclaw"]
    hufflepuff_df = df[df["Hogwarts House"] == "Hufflepuff"]

    for idx, course_name in enumerate(sorted_course_names):
        row = idx % ROW
        col = idx // ROW
        axs[col, row].set_title(
            f"{course_name[:10]}{'...' if len(course_name) > 10 else ''}"
        )
        axs[col, row].hist(gryffindor_df[course_name], alpha=0.5)
        axs[col, row].hist(slytherin_df[course_name], alpha=0.5)
        axs[col, row].hist(ravenclaw_df[course_name], alpha=0.5)
        axs[col, row].hist(hufflepuff_df[course_name], alpha=0.5)
    for i in range(13, 16):
        row = i % ROW
        col = i // ROW
        axs[col, row].axis("off")

    plt.show()


def plot_course_score_distribution(df, course_name):
    gryffindor_scores = df[df["Hogwarts House"] == "Gryffindor"][course_name]
    slytherin_scores = df[df["Hogwarts House"] == "Slytherin"][course_name]
    ravenclaw_scores = df[df["Hogwarts House"] == "Ravenclaw"][course_name]
    hufflepuff_scores = df[df["Hogwarts House"] == "Hufflepuff"][course_name]
    _, ax = plt.subplots()
    ax.hist(gryffindor_scores, alpha=0.5, label="Gryffindor")
    ax.hist(slytherin_scores, alpha=0.5, label="Slytherin")
    ax.hist(ravenclaw_scores, alpha=0.5, label="Ravenclaw")
    ax.hist(hufflepuff_scores, alpha=0.5, label="Hufflepuff")
    ax.set_title(course_name + " houses scores distribution")
    ax.set_ylabel("number of students")
    ax.set_xlabel("score")
    plt.legend(loc="upper right")
    plt.show()


def get_homogeneous_distribution_course(df):

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

    normalized_df = normalized_df.groupby("Hogwarts House").apply(
        lambda x: x.apply(apply_mean)
    )

    min_std = None
    most_homogenous = None
    course_score = dict()
    for column in normalized_df:
        tmp_std = apply_std(normalized_df[column])
        course_score[column] = tmp_std
        if min_std is None or tmp_std < min_std:
            min_std = tmp_std
            most_homogenous = column
    sorted_column_name = [
        t[0] for t in sorted(course_score.items(), key=lambda item: item[1])
    ]
    return (most_homogenous, sorted_column_name)


if __name__ == "__main__":
    df = load_pandas_csv("./datasets/dataset_train.csv")
    if df is None:
        exit()
    df = df.dropna()
    most_homogenous_course, sorted_course_names = get_homogeneous_distribution_course(
        df
    )
    plot_score_distribution(df, sorted_course_names)
    plot_course_score_distribution(df, most_homogenous_course)
