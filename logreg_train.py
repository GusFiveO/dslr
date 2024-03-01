#!/usr/bin/env python3

import sys
from utils import load_pandas_csv
import numpy as np
import pandas as pd

def print_usage():
    print("Usage: ./logreg_train.py dataset_pathname")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5  # Pour éviter log(0)
    cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        #cost = compute_cost(X, y, weights)
        #cost_history.append(cost)
    
    return weights, cost_history

def transform_labels(y, class_label):
    return np.where(y == class_label, 1, 0)


def transform_labels(y, class_label):
    return np.where(y == class_label, 1, 0)


def prepare_data(df, target_column):
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    X = X.drop("Care of Magical Creatures", axis=1)  # Supprimer la colonne "Care of Magical Creatures" (contient des valeurs manquantes
    X = X.drop("Arithmancy", axis=1)  # Supprimer la colonne "Arithmancy" (contient des valeurs manquantes)
    X = X.select_dtypes(include="number")  # Garder seulement les colonnes numériques
    X = X.fillna(X.mean())  # Remplacer les valeurs manquantes par la moyenne
    X = (X - X.mean()) / X.std()  # Normalisation
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Ajout d'une colonne de biais
    return X, y


def main():
    df = pd.read_csv(sys.argv[1])

    target_column = 'Hogwarts House'
    X, y = prepare_data(df, target_column)

    classes = y.unique()
    weights = {}
    learning_rate = 0.01
    iterations = 1000
    for class_label in classes:
        y_transformed = transform_labels(y, class_label)
        initial_weights = np.zeros(X.shape[1])
        weights[class_label], _ = gradient_descent(X, y_transformed, initial_weights, learning_rate, iterations)
    # Ici, vous pouvez sauvegarder les poids dans un fichier pour une utilisation future

    dfa = pd.read_csv(sys.argv[2])
    Xb, yb = prepare_data(dfa, target_column)
    predictions = []

    for x in Xb:  # Pour chaque observation dans l'ensemble de test
        probs = {class_label: sigmoid(np.dot(x, w)) for class_label, w in weights.items()}
        # Choisir la classe avec la probabilité la plus élevée
        predicted_class = max(probs, key=probs.get)
        predictions.append(predicted_class)

    print("Predictions:")
    print(predictions)

    results_df = pd.DataFrame(predictions, columns=['Hogwarts House'])
    results_df.index.name = 'Index'

    print("Results:")
    print(results_df)


    reference_df = pd.read_csv(sys.argv[3])

    nbr_false_elem = sum(reference_df["Hogwarts House"] != results_df["Hogwarts House"])
    # Calculer l'exactitude
    print(f"Exactitude: {nbr_false_elem}")

    print("Weights:")
    print(weights)



if __name__ == "__main__":
    try:
        dataset_pathname = sys.argv[1]
    except Exception:
        print_usage()
        exit(1)
    main()
    pass
