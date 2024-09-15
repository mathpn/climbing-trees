from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class LeafNode:
    value: int


@dataclass
class Node:
    feature_idx: int
    split_value: float
    left: Node | LeafNode
    right: Node | LeafNode


class Tree:
    def __init__(self) -> None:
        self.root = LeafNode()

    def fit(self, X, y) -> None:
        pass

    def predict(self, X) -> np.ndarray:
        pass


def _entropy(prob):
    prob = prob[prob > 0]
    return np.sum(-prob * np.log2(prob))


def _class_probabilities(labels):
    total_count = len(labels)
    return np.array(
        [label_count / total_count for label_count in Counter(labels).values()]
    )


def _find_best_split(X, y):

    min_split_entropy = 1e9
    best_split = best_feat = 0

    for feat_idx in range(X.shape[1]):
        feature = X[:, feat_idx]
        sort_idx = np.argsort(feature)
        feature_sort = feature[sort_idx]
        y_sort = y[sort_idx]

        for idx in range(1, len(sort_idx) - 1):
            entropy_l = _entropy(_class_probabilities(y_sort[:idx]))
            entropy_r = _entropy(_class_probabilities(y_sort[idx:]))
            p_l = (idx) / len(sort_idx)
            p_r = (len(sort_idx) - idx) / len(sort_idx)
            conditional_entropy = p_l * entropy_l + p_r * entropy_r
            if conditional_entropy < min_split_entropy:
                min_split_entropy = conditional_entropy
                best_split = feature_sort[idx]
                best_feat = feat_idx

    return best_feat, best_split


def split_node(leaf_node: LeafNode, X, y):
    feat_idx, best_split = _find_best_split(X, y)
    node = Node(feat_idx, best_split, LeafNode(0), LeafNode(1))
    return node


if __name__ == "__main__":
    X_0 = np.random.normal(0, size=100)
    X_1 = np.random.normal(2, size=100)
    X_2 = np.random.normal(0, size=200).reshape(-1, 1)
    X = np.concatenate((X_0, X_1)).reshape(-1, 1)
    X = np.concatenate((X_2, X), axis=1)
    y = np.repeat([0, 1], 100)
    node = LeafNode(0)
    node = split_node(node, X, y)
    print(node)
    pred = (X[:, node.feature_idx] >= node.split_value).astype(int)
    print(pred)
    print(y)
