#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from utils import load_pandas_csv

matplotlib.use("TkAgg")

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


def customize_xlabels(label):
    text = label.get_text()

    label.set_text(f"{text[:10]}{'...' if len(text) > 10 else ''}")


def customize_ylabels(label):
    text = label.get_text()

    label.set_text(f"{text[:10]}{'...' if len(text) > 10 else ''}")
    label.set_horizontalalignment("right")


if __name__ == "__main__":
    df = load_pandas_csv("./datasets/dataset_train.csv")
    if df is None:
        exit()
    print(df[course_name_list])
    df = df[course_name_list]
    scatter_matrix = pd.plotting.scatter_matrix(df, alpha=0.5)
    for ax in scatter_matrix.flatten():
        customize_xlabels(ax.xaxis.label)
        customize_ylabels(ax.yaxis.label)

        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(45)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(top=0.90, bottom=0.25, left=0.2)
    plt.show()
