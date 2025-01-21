from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, NamedTuple, Protocol, TypeVar

import numpy as np

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


@dataclass
class BaseSplitStats:
    left_weight: float
    right_weight: float


S = TypeVar("S", bound=BaseSplitStats)


@dataclass
class ClassificationSplitStats(BaseSplitStats):
    left_class_count: np.ndarray
    right_class_count: np.ndarray


@dataclass
class SquaredLossSplitStats(BaseSplitStats):
    left_sum: np.ndarray
    right_sum: np.ndarray
    left_sum_squared: np.ndarray
    right_sum_squared: np.ndarray


class Criterion(Protocol, Generic[S]):
    def node_impurity(self, y: np.ndarray, sample_weights: np.ndarray) -> float: ...

    def node_optimal_value(self, y: np.ndarray) -> np.ndarray: ...

    def init_split_stats(self, y: np.ndarray, sample_weights: np.ndarray) -> S: ...

    def update_split_stats(
        self, stats: S, y_value: np.ndarray, weight: float
    ) -> None: ...

    def split_impurity(self, stats: S) -> float: ...


class ClassificationCriterion:
    def __init__(self, objective_fn: Callable[[np.ndarray], float]):
        self.objective = objective_fn

    def node_optimal_value(self, y: np.ndarray) -> np.ndarray:
        return np.mean(y, axis=0)

    def node_impurity(self, y: np.ndarray, sample_weights: np.ndarray) -> float:
        return self.objective(_class_probabilities(y, sample_weights))

    def init_split_stats(
        self, y: np.ndarray, sample_weights: np.ndarray
    ) -> ClassificationSplitStats:
        sample_weights = sample_weights.reshape((-1, 1))
        return ClassificationSplitStats(
            left_weight=0,
            right_weight=np.sum(sample_weights),
            left_class_count=np.zeros(y.shape[1], dtype=y.dtype),
            right_class_count=np.sum(y * sample_weights, axis=0, dtype=y.dtype),
        )

    def update_split_stats(
        self,
        stats: ClassificationSplitStats,
        y_value: np.ndarray,
        weight: float,
    ) -> None:
        stats.left_weight += weight
        stats.right_weight -= weight
        stats.left_class_count += y_value * weight
        stats.right_class_count -= y_value * weight

    def split_impurity(self, stats: ClassificationSplitStats) -> float:
        criterion_l = self.objective(stats.left_class_count / stats.left_weight)
        criterion_r = self.objective(stats.right_class_count / stats.right_weight)

        total_weight = stats.left_weight + stats.right_weight
        p_l = stats.left_weight / total_weight
        p_r = stats.right_weight / total_weight
        return float(p_l * criterion_l + p_r * criterion_r)


class SquaredLossCriterion(Criterion):
    def node_impurity(self, y: np.ndarray, sample_weights: np.ndarray) -> float:
        sample_weights = sample_weights.reshape(-1, 1)
        weighted_mean = np.average(y, weights=sample_weights)
        return float(np.average((y - weighted_mean) ** 2, weights=sample_weights))

    def node_optimal_value(self, y: np.ndarray) -> np.ndarray:
        return np.mean(y, axis=0)

    def init_split_stats(
        self, y: np.ndarray, sample_weights: np.ndarray
    ) -> SquaredLossSplitStats:
        sample_weights = sample_weights.reshape((-1, 1))
        return SquaredLossSplitStats(
            left_weight=0,
            right_weight=np.sum(sample_weights),
            left_sum=np.zeros(y.shape[1], dtype=y.dtype),
            right_sum=np.sum(y * sample_weights, axis=0),
            left_sum_squared=np.zeros(y.shape[1], dtype=y.dtype),
            right_sum_squared=np.sum(y * y, axis=0),
        )

    def update_split_stats(
        self,
        stats: SquaredLossSplitStats,
        y_value: np.ndarray,
        weight: float,
    ) -> None:
        stats.left_sum += weight * y_value
        stats.right_sum -= weight * y_value
        stats.left_weight += weight
        stats.right_weight -= weight
        stats.left_sum_squared += weight * y_value * y_value
        stats.right_sum_squared -= weight * y_value * y_value

    def split_impurity(self, stats: SquaredLossSplitStats) -> float:
        left_mean = stats.left_sum / stats.left_weight if stats.left_weight > 0 else 0
        right_mean = (
            stats.right_sum / stats.right_weight if stats.right_weight > 0 else 0
        )

        criterion_l = (
            np.sum(stats.left_sum_squared / stats.left_weight - left_mean * left_mean)
            if stats.left_weight > 0
            else 0
        )
        criterion_r = (
            np.sum(
                stats.right_sum_squared / stats.right_weight - right_mean * right_mean
            )
            if stats.right_weight > 0
            else 0
        )

        total_weight = stats.left_weight + stats.right_weight
        p_l = stats.left_weight / total_weight
        p_r = stats.right_weight / total_weight
        return float(p_l * criterion_l + p_r * criterion_r)


def entropy(prob):
    prob = prob[prob > 0]
    return np.sum(-prob * np.log2(prob))


def gini_impurity(prob):
    return 1 - np.sum(prob**2)


def _class_probabilities(labels, sample_weights=None):
    if sample_weights is None:
        return np.mean(labels, axis=0)

    sample_weights = sample_weights.reshape((-1, 1))
    return (sample_weights * labels).sum(axis=0) / np.sum(sample_weights)


# TODO improve
def _one_hot_encode(arr):
    # # XXX
    # if arr.max() == 1:
    #     return arr.reshape(-1, 1)

    one_hot = np.zeros((arr.size, arr.max() + 1), dtype=np.uint8)
    one_hot[np.arange(arr.size), arr] = 1
    return one_hot


class Split(NamedTuple):
    criterion: float
    feature_idx: int
    split_value: float
    left_index: np.ndarray
    right_index: np.ndarray
    left_value: np.ndarray
    right_value: np.ndarray


def _find_best_split(
    X, y, criterion: Criterion, sample_weights: np.ndarray
) -> Split | None:
    min_score = np.inf
    best_split = None

    for feat_idx in range(X.shape[1]):
        if np.issubdtype(X.iloc[:, feat_idx].dtype, np.number):
            sort_idx = np.argsort(X.iloc[:, feat_idx])
            x_sorted = X.iloc[sort_idx, feat_idx].values
            y_sorted = y[sort_idx]
            weights_sorted = sample_weights[sort_idx]

            stats = criterion.init_split_stats(y_sorted, weights_sorted)

            for i in range(1, len(y_sorted)):
                criterion.update_split_stats(
                    stats, y_sorted[i - 1], weights_sorted[i - 1]
                )
                if x_sorted[i] != x_sorted[i - 1]:
                    score = criterion.split_impurity(stats)
                    if score < min_score:
                        min_score = score
                        best_split = Split(
                            criterion=min_score,
                            feature_idx=feat_idx,
                            split_value=x_sorted[i - 1],
                            left_index=sort_idx[:i],
                            right_index=sort_idx[i:],
                            left_value=criterion.node_optimal_value(y_sorted[:i]),
                            right_value=criterion.node_optimal_value(y_sorted[i:]),
                        )
        else:
            unique_values = np.unique(X.iloc[:, feat_idx])
            for value in unique_values:
                left_index = X.iloc[:, feat_idx] == value
                right_index = ~left_index

                if np.sum(left_index) == 0 or np.sum(right_index) == 0:
                    continue

                y_left = y[left_index]
                y_right = y[right_index]
                weights_left = sample_weights[left_index]
                weights_right = sample_weights[right_index]

                # TODO improve
                criterion_l = criterion.node_impurity(y_left, weights_left)
                criterion_r = criterion.node_impurity(y_right, weights_right)
                score = np.sum(weights_left) * criterion_l + np.sum(
                    weights_right
                ) * criterion_r / (np.sum(weights_left) + np.sum(weights_right))
                if score < min_score:
                    min_score = score
                    best_split = Split(
                        criterion=min_score,
                        feature_idx=feat_idx,
                        split_value=value,
                        left_index=np.where(left_index)[0],
                        right_index=np.where(right_index)[0],
                        left_value=criterion.node_optimal_value(y_left),
                        right_value=criterion.node_optimal_value(y_right),
                    )

    return best_split


# FIXME min_samples_leaf must be past of split search?
def split_node(
    node,
    X,
    y,
    value,
    depth,
    criterion: Criterion,
    sample_weights: np.ndarray,
    max_depth: int = 0,
    min_samples_leaf: int = 0,
    min_criterion_reduction: float = 0,
) -> LeafNode | Node | None:
    if X.shape[0] <= 1 or (max_depth and depth >= max_depth):
        return LeafNode(value)

    prior_criterion = criterion.node_impurity(y, sample_weights)
    if prior_criterion == 0:
        return LeafNode(value)

    split = _find_best_split(X, y, criterion, sample_weights)
    if split is None:
        return None

    criterion_reduction = prior_criterion - split.criterion
    if criterion_reduction and criterion_reduction < min_criterion_reduction:
        return None

    X_left = X.iloc[split.left_index, :]
    X_right = X.iloc[split.right_index, :]
    y_left = y[split.left_index]
    y_right = y[split.right_index]

    if min_samples_leaf and (
        X_left.shape[0] < min_samples_leaf or X_right.shape[0] < min_samples_leaf
    ):
        return None

    node = Node(
        split.feature_idx,
        split.split_value,
        LeafNode(split.left_value),
        LeafNode(split.right_value),
    )

    left = split_node(
        node=node,
        X=X_left,
        y=y_left,
        value=split.left_value,
        depth=depth + 1,
        criterion=criterion,
        sample_weights=sample_weights[split.left_index],
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_criterion_reduction=min_criterion_reduction,
    )
    right = split_node(
        node=node,
        X=X_right,
        y=y_right,
        value=split.right_value,
        depth=depth + 1,
        criterion=criterion,
        sample_weights=sample_weights[split.right_index],
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_criterion_reduction=min_criterion_reduction,
    )

    if left is not None:
        node.left = left
    if right is not None:
        node.right = right

    return node


def print_tree(node: Node | LeafNode, depth: int = 0):
    indent = "  " * depth
    if isinstance(node, LeafNode):
        print(f"{indent}LeafNode(value={np.array_str(node.value, precision=2)})")
    else:
        if isinstance(node.split_value, (float, int)):
            print(
                f"{indent}Node(feature_idx={node.feature_idx}, split_value={node.split_value:.2f})"
            )
        else:
            print(
                f"{indent}Node(feature_idx={node.feature_idx}, split_value={node.split_value})"
            )

        print(f"{indent}Left:")
        print_tree(node.left, depth + 1)
        print(f"{indent}Right:")
        print_tree(node.right, depth + 1)


def _prob_to_class(prob: np.ndarray) -> np.ndarray:
    if prob.shape[1] > 1:
        return np.argmax(prob, axis=1)

    return (prob.squeeze(1) >= 0.5).astype(int)


def count_nodes(node: Node | LeafNode):
    if isinstance(node, LeafNode):
        return 1

    return 1 + count_nodes(node.left) + count_nodes(node.right)


class Regressor(Protocol):
    def fit(self, X, y, sample_weights=None) -> None: ...

    def predict(self, X) -> np.ndarray: ...


class Classifier(Protocol):
    def fit(self, X, y, sample_weights=None) -> None: ...

    def predict(self, X) -> np.ndarray: ...

    def predict_proba(self, X) -> np.ndarray: ...


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int = 0,
        min_criterion_reduction: float = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_criterion_reduction = min_criterion_reduction
        self._root_node: Node | LeafNode | None = None

    def fit(self, X, y, sample_weights: np.ndarray | None = None) -> None:
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0], dtype=int)

        y_oh = _one_hot_encode(y)
        node = LeafNode(np.mean(y_oh, axis=0))
        trained_node = split_node(
            node=node,
            X=X,
            y=y_oh,
            value=np.mean(y, axis=0),
            depth=0,
            criterion=ClassificationCriterion(gini_impurity),
            sample_weights=sample_weights,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_criterion_reduction=self.min_criterion_reduction,
        )
        node = trained_node if trained_node is not None else node
        self._root_node = node

    def predict(self, X) -> np.ndarray:
        return _prob_to_class(self.predict_proba(X))

    def predict_proba(self, X) -> np.ndarray:
        if self._root_node is None:
            raise ValueError("model must be trained before prediction")

        y_pred = []
        for _, x in X.iterrows():
            node = self._root_node

            while isinstance(node, Node):
                if x.iloc[node.feature_idx] <= node.split_value:
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


class DecisionTreeRegressor:
    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int = 0,
        min_criterion_reduction: float = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_criterion_reduction = min_criterion_reduction
        self._root_node: Node | LeafNode | None = None

    def fit(self, X, y, sample_weights=None) -> None:
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0], dtype=int)

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        node = LeafNode(np.mean(y, axis=0))
        trained_node = split_node(
            node=node,
            X=X,
            y=y,
            value=np.mean(y, axis=0),
            depth=0,
            criterion=SquaredLossCriterion(),
            sample_weights=sample_weights,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_criterion_reduction=self.min_criterion_reduction,
        )
        node = trained_node if trained_node is not None else node
        self._root_node = node

    def predict(self, X) -> np.ndarray:
        if self._root_node is None:
            raise ValueError("model must be trained before prediction")

        y_pred = []
        for x in X:
            node = self._root_node

            while isinstance(node, Node):
                if x[node.feature_idx] <= node.split_value:
                    node = node.left
                else:
                    node = node.right

            y_pred.append(node.value)

        return np.array(y_pred)
