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
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
    
    return weights, cost_history

def transform_labels(y, class_label):
    return np.where(y == class_label, 1, 0)

def main():
    df = pd.read_csv('dataset_train.csv')
    df = df.select_dtypes(include="number")

    X = df.drop("Hogwarts House", axis=1)
    y = df.columns["Hogwarts House"]
    classes = y.unique()
    weights = {}


if __name__ == "__main__":
    try:
        dataset_pathname = sys.argv[1]
    except Exception:
        print_usage()
        exit(1)
    main()
    pass
