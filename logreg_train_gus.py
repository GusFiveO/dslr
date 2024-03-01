#!/usr/bin/env python3

import sys

import pandas as pd
import numpy as np
from utils import load_pandas_csv, print_logreg_usage, z_score_normalize

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


def sigmoide(X):
    return 1 / (1 + np.exp(-X))


def gradient(features, targets, weights):
    features_len = len(features)
    sig = sigmoide(np.dot(weights, features))
    return (1 / features_len) * np.dot(sig - targets, features.T)


def logreg(df: pd.DataFrame, epochs: int, learning_rate: float):
    nb_features = len(df.axes[1]) - 1
    weights = np.matrix(np.zeros(nb_features))
    features = df.drop("Hogwarts House", axis=1).to_numpy().T
    np.hstack((np.ones((features.shape[0], 1)), features))
    targets = np.matrix(df["Hogwarts House"].to_numpy())

    for _ in range(epochs):
        weights -= learning_rate * gradient(features, targets, weights)
    return weights


def logreg_one_vs_all(df: pd.DataFrame):
    houses_list = df["Hogwarts House"].unique()
    weights_dict = {}
    for house in houses_list:
        df_for_logreg = df.copy()

        house_column = df["Hogwarts House"]
        new_house_column = house_column.map(lambda row: 1 if row == house else 0)

        course_columns = df_for_logreg[course_name_list]
        course_columns = course_columns.fillna(course_columns.mean())
        normalized_df = course_columns.apply(z_score_normalize)
        normalized_df["Hogwarts House"] = new_house_column
        weights = logreg(normalized_df, 1000, 0.01)
        weights_dict[house] = weights
    print(weights_dict)
    return weights_dict


def test_weights(df: pd.DataFrame, weights_dict: dict):
    df = df.dropna()
    houses_list = df["Hogwarts House"].unique()
    df = df[df["First Name"] == "Carmen"]
    print(df)
    df = df[course_name_list]
    df = df.apply(z_score_normalize)
    for house in houses_list:
        weights = weights_dict[house]
        features = df.to_numpy().T
        print(features.shape)
        print(weights.shape)
        predict = sigmoide(np.dot(weights, features))
        print(house, predict)
    print(df)


if __name__ == "__main__":
    try:
        dataset_pathname = sys.argv[1]
    except Exception:
        print_logreg_usage()
        exit()
    df = load_pandas_csv(dataset_pathname)
    if df is None:
        exit()
    weights_dict = logreg_one_vs_all(df[course_name_list + ["Hogwarts House"]])
    print(df[df["First Name"] == "Carmen"])
    test_weights(df, weights_dict)
