from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class LeafNode:
    prob: float


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


def _one_hot_encode(arr):
    one_hot = np.zeros((arr.size, arr.max() + 1), dtype=np.uint8)
    one_hot[np.arange(arr.size), arr] = 1
    return one_hot


def _find_best_split(X, y):

    min_split_entropy = 1e9
    best_split = best_feat = left_prob = right_prob = 0
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
                left_prob = np.mean(y_sort[:idx])
                right_prob = np.mean(y_sort[idx:])

    return (
        min_split_entropy,
        best_feat,
        best_split,
        best_left,
        best_right,
        left_prob,
        right_prob,
    )


def split_node(node, X, y, value, depth, max_depth, min_info_gain: float = 0):
    if X.shape[0] <= 1 or depth >= max_depth:
        return LeafNode(value)

    prior_entropy = _entropy(_class_probabilities(y))
    if prior_entropy == 0:
        return LeafNode(value)

    split_entropy, feat_idx, best_split, left, right, left_prob, right_prob = (
        _find_best_split(X, y)
    )
    info_gain = prior_entropy - split_entropy
    if info_gain < min_info_gain:
        return None

    X_left = X[left, :]
    X_right = X[right, :]
    y_left = y[left]
    y_right = y[right]
    node = Node(feat_idx, best_split, LeafNode(left_prob), LeafNode(right_prob))

    left = split_node(
        node, X_left, y_left, left_prob, depth + 1, max_depth, min_info_gain
    )
    right = split_node(
        node, X_right, y_right, right_prob, depth + 1, max_depth, min_info_gain
    )

    if left is not None:
        node.left = left
    if right is not None:
        node.right = right

    return node


def train_tree(X, y, max_depth: int, min_info_gain: float) -> Node | LeafNode:
    node = LeafNode(0)
    trained_node = split_node(
        node, X, y, 0, depth=0, max_depth=max_depth, min_info_gain=min_info_gain
    )
    node = trained_node if trained_node is not None else node
    return node


def print_tree(node, depth=0):
    indent = "  " * depth
    if isinstance(node, LeafNode):
        print(f"{indent}LeafNode(value={node.prob})")
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

        y_pred.append(node.prob)

    return np.array(y_pred)


def predict_class(root: Node | LeafNode, X: np.ndarray) -> np.ndarray:
    pred_prob = predict(root, X)
    pred = (pred_prob >= 0.5).astype(int)
    return pred


def count_nodes(node: Node | LeafNode):
    if isinstance(node, LeafNode):
        return 1
    else:
        return 1 + count_nodes(node.left) + count_nodes(node.right)


if __name__ == "__main__":
    # n = 1000
    # X_0 = np.random.normal(0, size=n)
    # X_1 = np.random.normal(2, size=n)
    # X_2 = np.random.normal(0, size=2 * n).reshape(-1, 1)
    # X = np.concatenate((X_0, X_1)).reshape(-1, 1)
    # X = np.concatenate((X_2, X), axis=1)
    # y = np.repeat([0, 1], n)

    X, y = load_breast_cancer(return_X_y=True)
    tree = train_tree(X, y, max_depth=4, min_info_gain=0.2)
    pred_proba = predict(tree, X)
    pred = predict_class(tree, X)
    score = f1_score(y, pred)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba)
    print_tree(tree)
    print(pred)
    print(y)
    print(
        f"F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f} with {count_nodes(tree)} nodes"
    )
