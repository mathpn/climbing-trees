from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

from src.bagging import BaggingClassifier, BaggingRegressor
from src.random_forest import RandomForestClassifier, RandomForestRegressor
from src.cart import DecisionTreeClassifier, DecisionTreeRegressor, print_tree


def main():
    random.seed(42)
    np.random.seed(42)
    X, y = load_breast_cancer(return_X_y=True)
    X = pd.DataFrame(X)

    # X = pd.DataFrame(np.random.random(size=(20, 4)))
    # X["cat"] = np.array([["B"] * 9 + ["A"] * 6 + ["C"] * 5]).T
    # y = np.array([0] * 10 + [1] * 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    max_depth = 8
    min_samples_leaf = 1

    tree = DecisionTreeClassifier(max_depth, min_samples_leaf)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    score = f1_score(y_test, pred, average="macro")
    acc = accuracy_score(y_test, pred)
    print_tree(tree._root_node)
    print(f"classification tree -> F1: {score:.2f} accuracy: {acc:.2%}")
    print()

    tree = BaggingClassifier(
        lambda: DecisionTreeClassifier(),
        n_estimators=10,
    )
    tree.fit(X_train, y_train, sample_weights=np.ones(len(y)) / 10)
    pred = tree.predict(X_test)
    score = f1_score(y_test, pred, average="macro")
    acc = accuracy_score(y_test, pred)
    print(f"bagging tree -> F1: {score:.2f} accuracy: {acc:.2%}")
    print()

    tree = RandomForestClassifier(n_estimators=10)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    score = f1_score(y_test, pred, average="macro")
    acc = accuracy_score(y_test, pred)
    print(f"random forest -> F1: {score:.2f} accuracy: {acc:.2%}")
    print()

    max_depth = 8
    min_samples_leaf = 3

    X, y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    tree = DecisionTreeRegressor(max_depth, min_samples_leaf)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print_tree(tree._root_node)
    print(f"regression tree -> MSE: {mse:.2f}")
    print()

    tree = BaggingRegressor(
        lambda: DecisionTreeRegressor(),
        n_estimators=10,
    )
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print(f"bagging tree -> MSE: {mse:.2f}")
    print()

    tree = RandomForestRegressor(n_estimators=10)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print(f"random forest -> MSE: {mse:.2f}")
    print()


if __name__ == "__main__":
    main()
