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
    best_split = best_feat = left_label = right_label = 0
    best_left = best_right = None

    for feat_idx in range(X.shape[1]):
        feature = X[:, feat_idx]
        sort_idx = np.argsort(feature)
        feature_sort = feature[sort_idx]
        y_sort = y[sort_idx]

        for idx in range(1, len(sort_idx)):
            left = sort_idx[:idx]
            right = sort_idx[idx:]
            entropy_l = _entropy(_class_probabilities(y_sort[:idx]))
            entropy_r = _entropy(_class_probabilities(y_sort[idx:]))
            p_l = (idx) / len(sort_idx)
            p_r = (len(sort_idx) - idx) / len(sort_idx)
            conditional_entropy = p_l * entropy_l + p_r * entropy_r
            if conditional_entropy < min_split_entropy:
                min_split_entropy = conditional_entropy
                best_split = feature_sort[idx]
                best_feat = feat_idx
                best_left = left
                best_right = right
                left_label = (np.mean(y_sort[:idx]) >= 0.5).astype(int)
                right_label = (np.mean(y_sort[idx:]) >= 0.5).astype(int)

    return (
        min_split_entropy,
        best_feat,
        best_split,
        best_left,
        best_right,
        left_label,
        right_label,
    )


def split_node(node, X, y, value, depth, max_depth):
    if X.shape[0] <= 1 or depth >= max_depth:
        return LeafNode(value)

    prior_entropy = _entropy(_class_probabilities(y))
    if prior_entropy == 0:
        return LeafNode(value)

    split_entropy, feat_idx, best_split, left, right, left_label, right_label = (
        _find_best_split(X, y)
    )
    # print(f"information gain: {prior_entropy - split_entropy:.2f}")

    X_left = X[left, :]
    X_right = X[right, :]
    y_left = y[left]
    y_right = y[right]
    node = Node(feat_idx, best_split, LeafNode(left_label), LeafNode(right_label))

    node.left = split_node(node, X_left, y_left, left_label, depth + 1, max_depth)
    node.right = split_node(node, X_right, y_right, right_label, depth + 1, max_depth)
    return node


def print_tree(node, depth=0):
    indent = "  " * depth
    if isinstance(node, LeafNode):
        print(f"{indent}LeafNode(value={node.value})")
    else:
        print(
            f"{indent}Node(feature_idx={node.feature_idx}, split_value={node.split_value:.2f})"
        )
        print(f"{indent}Left:")
        print_tree(node.left, depth + 1)
        print(f"{indent}Right:")
        print_tree(node.right, depth + 1)


def predict(root: Node | LeafNode, X: np.ndarray) -> np.ndarray:
    y_pred = []
    for x in X:
        node = root

        while isinstance(node, Node):
            if x[node.feature_idx] < node.split_value:
                node = node.left
            else:
                node = node.right

        y_pred.append(node.value)

    return np.array(y_pred)


if __name__ == "__main__":
    X_0 = np.random.normal(0, size=100)
    X_1 = np.random.normal(2, size=100)
    X_2 = np.random.normal(0, size=200).reshape(-1, 1)
    X = np.concatenate((X_0, X_1)).reshape(-1, 1)
    X = np.concatenate((X_2, X), axis=1)
    y = np.repeat([0, 1], 100)
    node = LeafNode(0)
    node = split_node(node, X, y, 0, depth=0, max_depth=5)
    pred = predict(node, X)
    print(node)
    print_tree(node)
    print(pred)
    print(y)
