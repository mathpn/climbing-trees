from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

EPS = 1e-6


@dataclass
class LeafNode:
    prob: np.ndarray


@dataclass
class Node:
    feature_idx: int
    split_value: float
    left: Node | LeafNode
    right: Node | LeafNode


def _entropy(prob):
    prob = prob[prob > 0]
    return np.sum(-prob * np.log2(prob))


def _class_probabilities(labels, sample_weights=None):
    if sample_weights is None:
        return np.mean(labels, axis=0)

    return (sample_weights * labels).sum(axis=0) / np.sum(sample_weights)


def _one_hot_encode(arr):
    if arr.max() == 1:
        return arr.reshape(-1, 1)

    one_hot = np.zeros((arr.size, arr.max() + 1), dtype=np.uint8)
    one_hot[np.arange(arr.size), arr] = 1
    return one_hot


def _find_best_split(X, y, sample_weights=None):

    min_split_entropy = 1e9
    left_prob = right_prob = np.mean(y, axis=0)
    best_split = best_feat = 0
    best_left = best_right = None

    for feat_idx in range(X.shape[1]):
        feature = X[:, feat_idx]
        sort_idx = np.argsort(feature)
        feature_sort = feature[sort_idx]
        y_sort = y[sort_idx]

        for idx in range(1, len(sort_idx)):
            left = sort_idx[:idx]
            right = sort_idx[idx:]
            entropy_l = _entropy(_class_probabilities(y_sort[:idx], sample_weights))
            entropy_r = _entropy(_class_probabilities(y_sort[idx:], sample_weights))
            p_l = (idx) / len(sort_idx)
            p_r = (len(sort_idx) - idx) / len(sort_idx)
            conditional_entropy = p_l * entropy_l + p_r * entropy_r
            if conditional_entropy < min_split_entropy:
                min_split_entropy = conditional_entropy
                best_split = feature_sort[idx]
                best_feat = feat_idx
                best_left = left
                best_right = right
                left_prob = np.mean(y_sort[:idx], axis=0)
                right_prob = np.mean(y_sort[idx:], axis=0)

    return (
        min_split_entropy,
        best_feat,
        best_split,
        best_left,
        best_right,
        left_prob,
        right_prob,
    )


def split_node(
    node, X, y, value, depth, max_depth, min_info_gain: float = 0, sample_weights=None
):
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
        node,
        X_left,
        y_left,
        left_prob,
        depth + 1,
        max_depth,
        min_info_gain,
        sample_weights,
    )
    right = split_node(
        node,
        X_right,
        y_right,
        right_prob,
        depth + 1,
        max_depth,
        min_info_gain,
        sample_weights,
    )

    if left is not None:
        node.left = left
    if right is not None:
        node.right = right

    return node


def train_tree(
    X, y, max_depth: int, min_info_gain: float, sample_weights=None
) -> Node | LeafNode:
    y_oh = _one_hot_encode(y)
    node = LeafNode(np.zeros(y_oh.shape[1]))
    trained_node = split_node(
        node,
        X,
        y_oh,
        0,
        depth=0,
        max_depth=max_depth,
        min_info_gain=min_info_gain,
        sample_weights=sample_weights,
    )
    node = trained_node if trained_node is not None else node
    return node


def _sample(X, y, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.choice(range(X.shape[0]), size=sample_size, replace=True)

    X_sample = X[idx]
    y_sample = y[idx]

    return X_sample, y_sample


def train_tree_bagging(
    X, y, n_trees: int, sample_proportion: float, max_depth: int, min_info_gain: float
):
    n = math.floor(X.shape[0] * sample_proportion)
    return [
        train_tree(*_sample(X, y, n), max_depth, min_info_gain) for _ in range(n_trees)
    ]


def train_random_forest(
    X, y, n_trees: int, sample_proportion: float, max_depth: int, min_info_gain: float
):
    n = math.floor(X.shape[0] * sample_proportion)
    n_features = math.floor(math.sqrt(X.shape[1]))
    trees = []
    for _ in range(n_trees):
        idx = np.random.choice(np.arange(X.shape[1]), size=n_features, replace=False)
        X_sample = X[:, idx]
        tree = train_tree(*_sample(X_sample, y, n), max_depth, min_info_gain)
        trees.append((tree, idx))

    return trees


def _reweight_samples_adaboost(y, pred, sample_weights):
    err = 1 - np.sum(pred == y) / y.shape[0]
    n_classes = 1 + y.max() - y.min()  # TODO improve class count
    alpha = np.log((1 - err + EPS) / (err + EPS)) + np.log(n_classes - 1)
    sample_weights = sample_weights * np.exp(alpha * (pred != y))
    sample_weights = sample_weights / np.sum(sample_weights)
    return sample_weights


def train_adaboost(X, y, iterations: int):
    learners = []
    sample_weights = np.ones(X.shape[0]) / X.shape[0]
    for _ in range(iterations):
        # TODO parameters
        learner = train_tree(X, y, 2, 0, sample_weights=sample_weights)
        learners.append(learner)
        pred = prob_to_class(predict(learner, X))
        sample_weights = _reweight_samples_adaboost(y, pred, sample_weights)

    return learners


# XXX continue implementation
def train_gradient_boosting(X, y, iterations: int):
    learners = []
    residual = y
    for _ in range(iterations):
        # TODO parameters
        learner = train_tree(X, residual, 2, 0)
        learners.append(learner)
        pred = predict(learner, X)
        residual = residual - pred

    return learners


def print_tree(node, depth=0):
    indent = "  " * depth
    if isinstance(node, LeafNode):
        print(f"{indent}LeafNode(value={np.array_str(node.prob, precision=2)})")
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


def predict_ensemble(trees: list[Node | LeafNode], X: np.ndarray) -> np.ndarray:
    preds = [predict(root, X) for root in trees]
    pred_arr = np.stack(preds, axis=2)
    return pred_arr.mean(axis=2)


def predict_random_forest(
    trees: list[tuple[Node | LeafNode, np.ndarray]], X: np.ndarray
) -> np.ndarray:
    preds = [predict(root, X[:, idx]) for root, idx in trees]
    pred_arr = np.stack(preds, axis=2)
    return pred_arr.mean(axis=2)


def prob_to_class(prob: np.ndarray) -> np.ndarray:
    if prob.shape[1] > 1:
        return np.argmax(prob, axis=1)

    return (prob.squeeze(1) >= 0.5).astype(int)


def count_nodes(node: Node | LeafNode):
    if isinstance(node, LeafNode):
        return 1

    return 1 + count_nodes(node.left) + count_nodes(node.right)


if __name__ == "__main__":
    # n = 1000
    # X_0 = np.random.normal(0, size=n)
    # X_1 = np.random.normal(2, size=n)
    # X_2 = np.random.normal(0, size=2 * n).reshape(-1, 1)
    # X = np.concatenate((X_0, X_1)).reshape(-1, 1)
    # X = np.concatenate((X_2, X), axis=1)
    # y = np.repeat([0, 1], n)

    X, y = load_wine(return_X_y=True)
    max_depth = 4
    min_info_gain = 0.1

    tree = train_tree(X, y, max_depth=max_depth, min_info_gain=min_info_gain)
    pred_proba = predict(tree, X)
    pred = prob_to_class(pred_proba)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    print_tree(tree)
    print(
        f"tree -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f} with {count_nodes(tree)} nodes"
    )

    trees = train_tree_bagging(
        X, y, 5, 0.5, max_depth=max_depth, min_info_gain=min_info_gain
    )
    pred_proba = predict_ensemble(trees, X)
    pred = prob_to_class(pred_proba)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    # for i, tree in enumerate(trees):
    #     print(f"-> tree {i+1}")
    #     print_tree(tree)
    print(f"bagging -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")

    trees = train_random_forest(
        X, y, 5, 0.5, max_depth=max_depth, min_info_gain=min_info_gain
    )
    pred_proba = predict_random_forest(trees, X)
    pred = prob_to_class(pred_proba)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    # for i, tree in enumerate(trees):
    #     print(f"-> tree {i+1}")
    #     print_tree(tree)
    print(f"random forest -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")

    trees = train_adaboost(X, y, 10)
    pred_proba = predict_ensemble(trees, X)
    pred = prob_to_class(pred_proba)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    # for i, tree in enumerate(trees):
    #     print(f"-> tree {i+1}")
    #     print_tree(tree)
    print(f"AdaBoost -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")
