from __future__ import annotations

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

from src.cart import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
)
from spoiler import *


def main():
    import random

    from sklearn.tree import DecisionTreeClassifier as DTC
    from sklearn.tree import DecisionTreeRegressor as DTR
    from sklearn.tree import plot_tree

    random.seed(42)
    np.random.seed(42)
    # X, y = load_wine(return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True)
    # X = np.random.random(size=(20, 4))
    # y = np.array([0] * 10 + [1] * 10)
    # print(X)
    # print(y)
    max_depth = 6
    min_samples_leaf = 1

    tree = DecisionTreeClassifier(max_depth, min_samples_leaf)
    tree.fit(X, y)
    pred_proba = tree.predict_proba(X)
    pred = tree.predict(X)
    print(y)
    print(pred)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    node_count = tree.node_count()
    tree.print_tree()
    print(f"tree -> F1: {score:.2f} accuracy: {acc:.2%} with {node_count} nodes")

    tree = DTC(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(X, y)
    pred_proba = tree.predict_proba(X)
    pred = tree.predict(X)
    print(y)
    print(pred)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    print(f"tree -> F1: {score:.2f} accuracy: {acc:.2%}")
    plot_tree(tree)
    import matplotlib.pyplot as plt

    plt.savefig("tree.svg")

    X, y = load_diabetes(return_X_y=True)
    # X = np.random.random((len(y), 4))
    # X = np.array([[1, 1, 2, 2], [1, 2, 3, 4]]).T
    # y = np.array([0, 0, 1, 1])
    # print(X)
    # print(y)

    tree = DecisionTreeRegressor(max_depth, min_samples_leaf)
    tree.fit(X, y)
    pred = tree.predict(X)
    mse = mean_squared_error(y, pred)
    print(f"tree -> MSE: {mse:.2f} baseline = {np.mean((y - np.mean(y)) ** 2):.2f}")
    # print_tree(tree._root_node)

    tree = DTR(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(X, y)
    pred = tree.predict(X)
    mse = mean_squared_error(y, pred)
    print(f"tree -> MSE: {mse:.2f} baseline = {np.mean((y - np.mean(y)) ** 2):.2f}")
    # print_tree(tree._root_node)
    return

    bagging = BaggingClassifier(
        lambda: DecisionTreeClassifier(max_depth, min_samples_leaf), 5, 0.5
    )
    bagging.fit(X, y)
    pred_proba = bagging.predict_proba(X)
    pred = bagging.predict(X)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    print(f"bagging -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")

    random_forest = RandomForestClassifier(5, 0.5, max_depth)
    random_forest.fit(X, y)
    pred_proba = random_forest.predict_proba(X)
    pred = random_forest.predict(X)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    print(f"random forest -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")

    adaboost = AdaboostClassifier(lambda: DecisionTreeClassifier(1, 0), 10, 0.5)
    adaboost.fit(X, y)
    pred_proba = adaboost.predict_proba(X)
    pred = adaboost.predict(X)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    print(f"AdaBoost -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")

    boost = GradientBoostingClassifier(lambda: DecisionTreeRegressor(1, 0), 10, 0.3)
    boost.fit(X, y)
    pred_proba = boost.predict_proba(X)
    pred = boost.predict(X)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    print(f"Gradient boosting -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")


if __name__ == "__main__":
    main()
