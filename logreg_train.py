#!/usr/bin/env python3

import sys
from utils import load_pandas_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse
import random

matplotlib.use("TkAgg")


def print_usage():
    print("Usage: ./logreg_train.py dataset_pathname")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5  # Pour éviter log(0)
    cost = (-1 / m) * np.sum(
        y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
    )
    return cost


def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history


def gradient(features, targets, weights):

    features_len = len(features)

    sig = sigmoid(np.dot(features, weights))

    return (1 / features_len) * np.dot(features.T, sig - targets)


def stochastic_gradient_descent(features, targets, weights, learning_rate):
    cost_history = list()
    features_len = len(features)
    for i in range(features_len):
        tmp_features = features.T[:, [i]].T
        tmp_targets = targets[[i]]
        weights -= learning_rate * gradient(tmp_features, tmp_targets, weights)
        cost = compute_cost(features, targets, weights)
        # cost = compute_cost(tmp_features, tmp_targets, weights)
        cost_history.append(cost)

    return weights, cost_history


def mini_batch_gradient_descent(features, targets, weights, learning_rate, batch_size):
    cost_history = list()
    features_len = len(features)
    for i in range(0, features_len, batch_size):
        print(i)
        batch_rows = [row for row in range(i, i + batch_size) if row < features_len]
        tmp_features = features.T[:, batch_rows].T
        tmp_targets = targets[batch_rows]
        weights -= learning_rate * gradient(tmp_features, tmp_targets, weights)
        cost = compute_cost(features, targets, weights)
        # cost = compute_cost(tmp_features, tmp_targets, weights)
        cost_history.append(cost)
    return weights, cost_history


def transform_labels(y, class_label):
    return np.where(y == class_label, 1, 0)


def prepare_data(df, target_column):
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    X = X.drop(
        "Care of Magical Creatures", axis=1
    )  # Supprimer la colonne "Care of Magical Creatures" (contient des valeurs manquantes
    X = X.drop(
        "Arithmancy", axis=1
    )  # Supprimer la colonne "Arithmancy" (contient des valeurs manquantes)
    X = X.drop("Astronomy", axis=1)
    X = X.select_dtypes(include="number")  # Garder seulement les colonnes numériques
    X = X.fillna(X.mean())  # Remplacer les valeurs manquantes par la moyenne
    X = (X - X.mean()) / X.std()  # Normalisation
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Ajout d'une colonne de biais
    return X, y


def parsing_args():
    parser = argparse.ArgumentParser(
        description="Votre programme de régression logistique."
    )
    parser.add_argument(
        "fichier",
        type=str,
        help="Le chemin du fichier à analyser.\n\n Doit etre placer en premier",
    )
    parser.add_argument(
        "-gradient",
        choices=["batch", "stochastic", "mini_batch"],
        help="Choisir le type de descente de gradient à utiliser.",
    )
    parser.add_argument(
        "-show",
        nargs="+",
        help="Afficher l'historique des coûts pour les maisons spécifiées: Gryffindor, Hufflepuff, Ravenclaw, Slytherin.\n\nExemple:./logreg_train.py datasets/dataset_train.csv -show Gryffindor Hufflepuff Ravenclaw Slytherin",
    )

    args = parser.parse_args()

    if args.show:
        # Afficher l'historique des coûts pour les maisons spécifiées
        tab = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

        print(f"Affichage de l'historique des coûts pour {args.show}")
        for element in args.show:
            # si element n'est pas dans tab
            if element not in tab:
                print(f"La maison {element} n'existe pas")
                print(f"Les maisons existantes sont: {tab}")
                exit(1)

        # On enleve les doublons
        args.show = list(set(args.show))

    if args.gradient:
        # Choisir le type de gradient
        print(f"Utilisation du gradient {args.gradient}")

    if args.fichier:
        print(f"Le fichier à analyser est {args.fichier}")
        if args.fichier != "datasets/dataset_train.csv":
            print("Le fichier à analyser doit être datasets/dataset_train.csv")
            exit(1)

    if args.gradient is None:
        args.gradient = "batch"
    return args


def main():

    args = parsing_args()

    df = pd.read_csv(args.fichier)

    target_column = "Hogwarts House"
    X, y = prepare_data(df, target_column)

    classes = y.unique()
    weights = {}
    learning_rate = 0.01
    iterations = 1000
    for class_label in classes:
        y_transformed = transform_labels(y, class_label)
        initial_weights = np.zeros(X.shape[1])
        history_cost = list()
        if args.gradient == "batch":
            weights[class_label], history_cost = gradient_descent(
                X, y_transformed, initial_weights, learning_rate, iterations
            )
        elif args.gradient == "stochastic":
            weights[class_label], history_cost = stochastic_gradient_descent(
                X, y_transformed, initial_weights, learning_rate
            )
        elif args.gradient == "mini_batch":
            weights[class_label], history_cost = mini_batch_gradient_descent(
                X, y_transformed, initial_weights, learning_rate, 4
            )
        if args.show is not None and class_label in args.show:
            ## Print the cost history for Ravenclaw
            ## using graphique to see the cost history
            plt.plot(history_cost)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Cost History for {}".format(class_label))
            plt.show()

    ## Sauvegarde des poids dans un fichier
    np.save("weights.npy", weights)
    exit(1)


if __name__ == "__main__":
    ##    try:
    ##dataset_pathname = sys.argv[1]
    ##except Exception:
    ##print_usage()
    ##exit(1)
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)
    pass
