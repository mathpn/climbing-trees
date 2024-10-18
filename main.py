from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

EPS = 1e-6


@dataclass
class LeafNode:
    value: np.ndarray


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


def _entropy_criterion(y, sample_weights=None):
    return _entropy(_class_probabilities(y, sample_weights))


def _mse_criterion(y, sample_weights=None):
    if sample_weights is None:
        sample_weights = np.ones_like(y) / y.shape[0]

    value = (sample_weights * y) / sample_weights.sum()
    return np.mean(np.power(y - value, 2))


def _one_hot_encode(arr):
    if arr.max() == 1:
        return arr.reshape(-1, 1)

    one_hot = np.zeros((arr.size, arr.max() + 1), dtype=np.uint8)
    one_hot[np.arange(arr.size), arr] = 1
    return one_hot


def _find_best_split(X, y, criterion_fn, sample_weights):
    min_criterion = 1e9
    left_value = right_value = np.mean(y, axis=0)
    best_split = best_feat = 0
    best_left = best_right = None

    for feat_idx in range(X.shape[1]):
        feature = X[:, feat_idx]
        sort_idx = np.argsort(feature)
        feature_sort = feature[sort_idx]
        y_sort = y[sort_idx]
        weights_sort = sample_weights[sort_idx]

        for idx in range(1, len(sort_idx)):
            left = sort_idx[:idx]
            right = sort_idx[idx:]
            criterion_l = criterion_fn(y_sort[:idx], weights_sort[:idx])
            criterion_r = criterion_fn(y_sort[idx:], weights_sort[idx:])
            p_l = (idx) / len(sort_idx)
            p_r = (len(sort_idx) - idx) / len(sort_idx)
            criterion = p_l * criterion_l + p_r * criterion_r
            if criterion < min_criterion:
                min_criterion = criterion
                best_split = feature_sort[idx]
                best_feat = feat_idx
                best_left = left
                best_right = right
                left_value = np.mean(y_sort[:idx], axis=0)
                right_value = np.mean(y_sort[idx:], axis=0)

    return (
        min_criterion,
        best_feat,
        best_split,
        best_left,
        best_right,
        left_value,
        right_value,
    )


def split_node(
    node,
    X,
    y,
    value,
    depth,
    max_depth,
    criterion_fn,
    sample_weights=None,
    min_criterion_reduction: float = 0,
):
    if X.shape[0] <= 1 or depth >= max_depth:
        return LeafNode(value)

    if sample_weights is None:
        sample_weights = _uniform_sample_weights(X)

    prior_criterion = criterion_fn(y)
    if prior_criterion == 0:
        return LeafNode(value)

    split_criterion, feat_idx, best_split, left, right, left_prob, right_prob = (
        _find_best_split(X, y, criterion_fn, sample_weights)
    )
    info_gain = prior_criterion - split_criterion
    if info_gain < min_criterion_reduction:
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
        criterion_fn,
        sample_weights,
        min_criterion_reduction,
    )
    right = split_node(
        node,
        X_right,
        y_right,
        right_prob,
        depth + 1,
        max_depth,
        criterion_fn,
        sample_weights,
        min_criterion_reduction,
    )

    if left is not None:
        node.left = left
    if right is not None:
        node.right = right

    return node


class Estimator(Protocol):
    def fit(self, X, y, sample_weights=None) -> None: ...

    def predict(self, X) -> np.ndarray: ...

    def predict_proba(self, X) -> np.ndarray: ...


class DecisionTreeClassifier:
    def __init__(self, max_depth: int, min_info_gain: float = 0) -> None:
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self._root_node: Node | LeafNode | None = None

    def fit(self, X, y, sample_weights=None) -> None:
        y_oh = _one_hot_encode(y)
        node = LeafNode(np.zeros(y_oh.shape[1]))
        trained_node = split_node(
            node,
            X,
            y_oh,
            np.mean(y, axis=0),
            depth=0,
            max_depth=max_depth,
            criterion_fn=_entropy_criterion,
            sample_weights=sample_weights,
            min_criterion_reduction=min_info_gain,
        )
        node = trained_node if trained_node is not None else node
        self._root_node = node

    def predict(self, X) -> np.ndarray:
        return _prob_to_class(self.predict_proba(X))

    def predict_proba(self, X) -> np.ndarray:
        if self._root_node is None:
            raise ValueError("model must be trained before prediction")

        y_pred = []
        for x in X:
            node = self._root_node

            while isinstance(node, Node):
                if x[node.feature_idx] < node.split_value:
                    node = node.left
                else:
                    node = node.right

            y_pred.append(node.value)

        return np.array(y_pred)

    def print_tree(self) -> None:
        if self._root_node is None:
            print("model is not trained")
            return
        print_tree(self._root_node)

    def node_count(self) -> int:
        if self._root_node is None:
            return 0
        return count_nodes(self._root_node)


class BaggingClassifier:
    def __init__(
        self,
        estimator_constructor: Callable[[], Estimator],
        n_estimators: int,
        sample_proportion: float,
    ) -> None:
        self.estimator_constructor = estimator_constructor
        self.n_estimators = n_estimators
        self.sample_proportion = sample_proportion
        self._estimators: list[Estimator] = []

    def fit(self, X, y, sample_weights=None) -> None:
        n = math.floor(X.shape[0] * self.sample_proportion)
        self._estimators = []
        for _ in range(self.n_estimators):
            model = self.estimator_constructor()
            model.fit(*_sample(X, y, n), sample_weights=sample_weights)
            self._estimators.append(model)

    def predict(self, X) -> np.ndarray:
        return _prob_to_class(self.predict_proba(X))

    def predict_proba(self, X) -> np.ndarray:
        if not self._estimators:
            raise ValueError("model must be trained before prediction")

        preds = [model.predict_proba(X) for model in self._estimators]
        pred_arr = np.stack(preds, axis=2)
        return pred_arr.mean(axis=2)


def _uniform_sample_weights(X):
    return np.ones((X.shape[0], 1)) / X.shape[0]


def train_classification_tree(
    X, y, max_depth: int, min_info_gain: float, sample_weights=None
) -> Node | LeafNode:
    y_oh = _one_hot_encode(y)
    node = LeafNode(np.zeros(y_oh.shape[1]))
    trained_node = split_node(
        node,
        X,
        y_oh,
        np.mean(y, axis=0),
        depth=0,
        max_depth=max_depth,
        criterion_fn=_entropy_criterion,
        sample_weights=sample_weights,
        min_criterion_reduction=min_info_gain,
    )
    node = trained_node if trained_node is not None else node
    return node


def train_regression_tree(
    X, y, max_depth: int, min_info_gain: float, sample_weights=None
) -> Node | LeafNode:
    node = LeafNode(np.zeros(y.shape[1]))
    trained_node = split_node(
        node,
        X,
        y,
        np.mean(y, axis=0),
        depth=0,
        max_depth=max_depth,
        criterion_fn=_mse_criterion,
        min_criterion_reduction=min_info_gain,
        sample_weights=sample_weights,
    )
    node = trained_node if trained_node is not None else node
    return node


def _sample(X, y, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.choice(range(X.shape[0]), size=sample_size, replace=True)

    X_sample = X[idx]
    y_sample = y[idx]

    return X_sample, y_sample


def train_random_forest(
    X, y, n_trees: int, sample_proportion: float, max_depth: int, min_info_gain: float
):
    n = math.floor(X.shape[0] * sample_proportion)
    n_features = math.floor(math.sqrt(X.shape[1]))
    trees = []
    for _ in range(n_trees):
        idx = np.random.choice(np.arange(X.shape[1]), size=n_features, replace=False)
        X_sample = X[:, idx]
        tree = train_classification_tree(
            *_sample(X_sample, y, n), max_depth, min_info_gain
        )
        trees.append((tree, idx))

    return trees


def _reweight_samples_adaboost(y, pred, sample_weights):
    err = 1 - np.sum(pred == y) / y.shape[0]
    n_classes = 1 + y.max() - y.min()  # TODO improve class count
    alpha = np.log((1 - err + EPS) / (err + EPS)) + np.log(n_classes - 1)
    diff = (pred != y).reshape(-1, sample_weights.shape[1])
    sample_weights = sample_weights * np.exp(alpha * diff)
    sample_weights = sample_weights / np.sum(sample_weights)
    return sample_weights


def train_adaboost(X, y, iterations: int):
    learners = []
    sample_weights = np.ones((X.shape[0], 1)) / X.shape[0]
    for _ in range(iterations):
        # TODO parameters
        learner = train_classification_tree(X, y, 2, 0, sample_weights=sample_weights)
        learners.append(learner)
        pred = _prob_to_class(predict(learner, X))
        sample_weights = _reweight_samples_adaboost(y, pred, sample_weights)

    return learners


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_gradient_boosting(X, y, iterations: int, learning_rate: float):
    learners = []

    y_oh = _one_hot_encode(y)
    activation_fn = sigmoid if y_oh.shape[1] == 1 else softmax
    raw_pred = np.zeros_like(y_oh, dtype=np.float32)
    prob = activation_fn(raw_pred)

    for _ in range(iterations):
        # TODO parameters
        pseudo_residual = y_oh - prob
        learner = train_regression_tree(X, pseudo_residual, 1, 0)
        learner_pred = predict(learner, X)
        # TODO update predictions
        raw_pred += learning_rate * learner_pred
        prob = activation_fn(raw_pred)
        learners.append(learner)

    return (raw_pred, learning_rate, learners)


def print_tree(node: Node | LeafNode, depth: int = 0):
    indent = "  " * depth
    if isinstance(node, LeafNode):
        print(f"{indent}LeafNode(value={np.array_str(node.value, precision=2)})")
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


def predict_ensemble(trees: list[Node | LeafNode], X: np.ndarray) -> np.ndarray:
    preds = [predict(root, X) for root in trees]
    pred_arr = np.stack(preds, axis=2)
    return pred_arr.mean(axis=2)


def predict_random_forest(
    model: list[tuple[Node | LeafNode, np.ndarray]], X: np.ndarray
) -> np.ndarray:
    preds = [predict(root, X[:, idx]) for root, idx in model]
    pred_arr = np.stack(preds, axis=2)
    return pred_arr.mean(axis=2)


def predict_gradient_boosting(
    model: tuple[np.ndarray, float, list[Node | LeafNode]], X: np.ndarray
) -> np.ndarray:
    base_value, learning_rate, learners = model
    pred = base_value + learning_rate * np.sum(
        [predict(learner, X) for learner in learners], axis=0
    )
    if pred.shape[1] == 1:
        pred = sigmoid(pred)
    else:
        pred = softmax(pred)

    return pred


def _prob_to_class(prob: np.ndarray) -> np.ndarray:
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

    # X, y = load_wine(return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True)
    max_depth = 4
    min_info_gain = 0.1

    tree = DecisionTreeClassifier(max_depth, min_info_gain)
    tree.fit(X, y)
    pred_proba = tree.predict_proba(X)
    pred = tree.predict(X)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    tree.print_tree()
    print(
        f"tree -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f} with {tree.node_count()} nodes"
    )

    trees = BaggingClassifier(
        lambda: DecisionTreeClassifier(max_depth, min_info_gain), 5, 0.5
    )
    trees.fit(X, y)
    pred_proba = trees.predict_proba(X)
    pred = trees.predict(X)
    score = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    print(f"bagging -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")

    # trees = train_random_forest(
    #     X, y, 5, 0.5, max_depth=max_depth, min_info_gain=min_info_gain
    # )
    # pred_proba = predict_random_forest(trees, X)
    # pred = prob_to_class(pred_proba)
    # score = f1_score(y, pred, average="macro")
    # acc = accuracy_score(y, pred)
    # auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    # # for i, tree in enumerate(trees):
    # #     print(f"-> tree {i+1}")
    # #     print_tree(tree)
    # print(f"random forest -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")

    # trees = train_adaboost(X, y, 10)
    # pred_proba = predict_ensemble(trees, X)
    # pred = prob_to_class(pred_proba)
    # score = f1_score(y, pred, average="macro")
    # acc = accuracy_score(y, pred)
    # auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    # # for i, tree in enumerate(trees):
    # #     print(f"-> tree {i+1}")
    # #     print_tree(tree)
    # print(f"AdaBoost -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")

    # trees = train_gradient_boosting(X, y, 10, 0.3)
    # pred_proba = predict_gradient_boosting(trees, X)
    # pred = _prob_to_class(pred_proba)
    # score = f1_score(y, pred, average="macro")
    # acc = accuracy_score(y, pred)
    # auc = roc_auc_score(y, pred_proba, multi_class="ovr")
    # # for i, tree in enumerate(trees):
    # #     print(f"-> tree {i+1}")
    # #     print_tree(tree)
    # print(f"Gradient boosting -> F1: {score:.2f} accuracy: {acc:.2%} AUC {auc:.2f}")
