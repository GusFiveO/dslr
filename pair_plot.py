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


def customize_label(label, text):
    label.set_rotation(45)
    label.set_horizontalalignment("right")
    label.set_text(f"{text[:10]}{'...' if len(text) > 10 else ''}")


def customize_scatter_xlabels(label):
    text = label.get_text()

    label.set_text(f"{text[:10]}{'...' if len(text) > 10 else ''}")


def customize_scatter_ylabels(label):
    text = label.get_text()

    label.set_text(f"{text[:10]}{'...' if len(text) > 10 else ''}")
    label.set_horizontalalignment("right")


def pd_scatter_matrix(df):
    scatter_matrix = pd.plotting.scatter_matrix(df, alpha=0.5)
    for ax in scatter_matrix.flatten():
        customize_scatter_xlabels(ax.xaxis.label)
        customize_scatter_ylabels(ax.yaxis.label)

        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(45)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(top=0.90, bottom=0.25, left=0.2)
    plt.show()


def scatter_matrix(df: pd.DataFrame):
    nb_feature = len(df.axes[1])
    fig, axs = plt.subplots(nb_feature, nb_feature)
    for i in range(nb_feature**2):
        row = i // nb_feature
        col = i % nb_feature
        axs[row, col].set_yticks([])
        axs[row, col].set_xticks([])
        if col != 0:
            axs[row, col].set_yticks([])
        else:
            customize_label(axs[row, col].yaxis.label, course_name_list[row])
        if row != nb_feature - 1:
            axs[row, col].set_xticks([])
        else:
            customize_label(axs[row, col].xaxis.label, course_name_list[col])
        if row == col:
            axs[row, col].hist(df[course_name_list[row]])
        else:
            axs[row, col].scatter(
                df[course_name_list[row]], df[course_name_list[col]], alpha=0.5, s=1
            )

    plt.subplots_adjust(top=0.90, bottom=0.25, left=0.2, wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    df = load_pandas_csv("./datasets/dataset_train.csv")
    if df is None:
        exit()
    df = df[course_name_list]
    # pd_scatter_matrix(df)
    scatter_matrix(df)
