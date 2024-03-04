#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np

def print_usage():
    print("Usage: ./logreg_predict.py dataset_pathname weights.npy")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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
    predictions = []

    weights = np.load(sys.argv[2], allow_pickle=True).item()
    classes = y.unique()

    for x in X:
        probs = {class_label: sigmoid(np.dot(x, w)) for class_label, w in weights.items()}
        predicted_class = max(probs, key=probs.get)
        predictions.append(predicted_class)

    result = pd.DataFrame(predictions, columns=["Hogwarts House"])
    result.index.name = "Index"
    
    print("Predictions dataframe:")
    print(result)

    result.to_csv("houses.csv")


if __name__ == "__main__":
    try:
        dataset_pathname = sys.argv[1]
        weiths_pathname = sys.argv[2]
        if weiths_pathname != "weights.npy":
            print_usage()
            exit(1)
    except Exception:
        print_usage()
        exit(1)
    try:
        main()
    except Exception as e:
        print(f"An error occured. Please check your input files. : {e}")
        exit(1)
    pass
