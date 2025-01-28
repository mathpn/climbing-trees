from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from src.cart import DecisionTreeClassifier, DecisionTreeRegressor, print_tree


def main():
    random.seed(42)
    np.random.seed(42)
    # X, y = load_wine(return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True)
    X = pd.DataFrame(np.random.random(size=(20, 4)))
    X["cat"] = np.array([["B"] * 9 + ["A"] * 6 + ["C"] * 5]).T
    y = np.array([0] * 10 + [1] * 10)

    max_depth = 4
    min_samples_leaf = 3

    tree = DecisionTreeClassifier(max_depth, min_samples_leaf)
    tree.fit(X, y, sample_weights=np.ones(len(y)) / 10)
    pred = tree.predict(X)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    print_tree(tree._root_node)
    print(f"classification tree -> F1: {score:.2f} accuracy: {acc:.2%}")

    X, y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(X)

    tree = DecisionTreeRegressor(max_depth, min_samples_leaf)
    tree.fit(X, y)
    pred = tree.predict(X)
    mse = mean_squared_error(y, pred)
    print_tree(tree._root_node)
    print(f"regression tree -> MSE: {mse:.2f}")


if __name__ == "__main__":
    main()
