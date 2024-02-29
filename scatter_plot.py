#!/usr/bin/env python3

import sys
from utils import load_pandas_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_usage():
    print("Usage: scatter_plot.py dataset_pathname")


def pearson_correlation(X, Y):
    """Calcule la corrélation de Pearson entre deux vecteurs."""
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    numerator = np.sum((X - mean_X) * (Y - mean_Y))
    denominator = np.sqrt(np.sum((X - mean_X)**2) * np.sum((Y - mean_Y)**2))
    return numerator / denominator if denominator != 0 else 0


def compute_correlation_matrix(data):
    """Calcule la matrice de corrélation pour un ensemble de données."""
    n = data.shape[1]  # Nombre de caractéristiques
    corr_matrix = np.zeros((n, n))  # Initialisation de la matrice de corrélation
    for i in range(n):
        
        for j in range(i + 1, n):
            # Calculer seulement pour j > i
            corr = pearson_correlation(data.iloc[:, i], data.iloc[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    np.fill_diagonal(corr_matrix, 0)
    return corr_matrix


def find_max_correlation(corr_matrix):
    """Trouve la paire de caractéristiques avec la corrélation maximale."""
    max_corr_value = 1  # Initialiser avec une valeur impossible pour une corrélation valide
    max_indices = (0, 0)  # Initialiser les indices de la paire maximale
    # Parcourir la matrice de corrélation
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):  # Eviter la diagonale et les duplicatas
            corr_distance = abs(abs(corr_matrix[i, j]) - 1)
            if corr_distance < max_corr_value :
                max_corr_value = abs(abs(corr_matrix[i, j]) -1)
                max_indices = (i, j)
    
    print(f"correlation value: {corr_matrix[max_indices[0], max_indices[1]]}")
    return max_indices, max_corr_value

def print_graph(data, a, b):
    var1 = data.columns[a]
    var2 = data.columns[b]
    plt.figure(figsize=(10, 6))
    plt.scatter(data[var1], data[var2], alpha=0.5)
    plt.title(f"Scatter Plot entre {var1} et {var2}")
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    try:
        dataset_pathname = sys.argv[1]
    except Exception:
        print_usage()
        exit(1)

    df = load_pandas_csv(dataset_pathname)
    df = df.select_dtypes(include="number")

    ## Data clean, ready to get exploited


    corr = compute_correlation_matrix(df)
    a, b = find_max_correlation(corr)
    print(f"La paire de caractéristiques avec la corrélation maximale est: {df.columns[a[0]]} et {df.columns[a[1]]}")
    print_graph(df, a[0], a[1])
