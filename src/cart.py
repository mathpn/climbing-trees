from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic, NamedTuple, Protocol, TypeVar

import numpy as np
import pandas as pd


@dataclass
class LeafNode:
    value: np.ndarray


split_value = float | set


@dataclass
class Node:
    feature_idx: int
    split_value: split_value
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

    def make_stats_from_categorical_level(
        self, stats: S, y: np.ndarray, sample_weights: np.ndarray, is_left: bool
    ) -> S: ...


class ClassificationCriterion(Criterion):
    """Criterion for classification trees."""

    def __init__(self, objective_fn: Callable[[np.ndarray], float]):
        self.objective = objective_fn

    def node_optimal_value(self, y: np.ndarray) -> np.ndarray:
        return np.mean(y, axis=0)

    def node_impurity(self, y: np.ndarray, sample_weights: np.ndarray) -> float:
        if y.shape[1] == 1:
            y = np.hstack((y, 1 - y))
        return self.objective(_class_probabilities(y, sample_weights))

    def init_split_stats(
        self, y: np.ndarray, sample_weights: np.ndarray
    ) -> ClassificationSplitStats:
        sample_weights = sample_weights.reshape((-1, 1))

        # For binary classification with single column
        if y.shape[1] == 1:
            y = np.hstack((y, 1 - y))

        return ClassificationSplitStats(
            left_weight=0,
            right_weight=np.sum(sample_weights),
            left_class_count=np.zeros(y.shape[1], dtype=sample_weights.dtype),
            right_class_count=np.sum(
                y * sample_weights, axis=0, dtype=sample_weights.dtype
            ),
        )

    def update_split_stats(
        self,
        stats: ClassificationSplitStats,
        y_value: np.ndarray,
        weight: float,
    ) -> None:
        stats.left_weight += weight
        stats.right_weight -= weight

        # For binary classification with single column
        if len(y_value) == 1:
            y_value = np.hstack((y_value, 1 - y_value))

        stats.left_class_count += y_value * weight
        stats.right_class_count -= y_value * weight

    def split_impurity(self, stats: ClassificationSplitStats) -> float:
        criterion_l = self.objective(stats.left_class_count / stats.left_weight)
        criterion_r = self.objective(stats.right_class_count / stats.right_weight)

        total_weight = stats.left_weight + stats.right_weight
        p_l = stats.left_weight / total_weight
        p_r = stats.right_weight / total_weight
        return float(p_l * criterion_l + p_r * criterion_r)

    def make_stats_from_categorical_level(
        self,
        stats: ClassificationSplitStats,
        y: np.ndarray,
        sample_weights: np.ndarray,
        is_left: bool,
    ) -> ClassificationSplitStats:
        level_weights = sample_weights.reshape(-1, 1)
        level_weight = np.sum(sample_weights)

        # For binary classification with single column
        if y.shape[1] == 1:
            y = np.hstack((y, 1 - y))

        level_sum = np.sum(y * level_weights, axis=0)

        if is_left:
            stats = replace(
                stats,
                left_weight=level_weight,
                right_weight=stats.right_weight - level_weight,
                left_class_count=level_sum,
                right_class_count=stats.right_class_count - level_sum,
            )
        else:
            stats = replace(
                stats,
                left_weight=stats.left_weight - level_weight,
                right_weight=level_weight,
                left_class_count=stats.left_class_count - level_sum,
                right_class_count=level_sum,
            )
        return stats


class SquaredLossCriterion(Criterion):
    """Criterion for regression trees using squared loss."""

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
            right_sum_squared=np.sum(y * y * sample_weights, axis=0),
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

    def make_stats_from_categorical_level(
        self,
        stats: SquaredLossSplitStats,
        y: np.ndarray,
        sample_weights: np.ndarray,
        is_left: bool,
    ) -> SquaredLossSplitStats:
        level_weights = sample_weights.reshape(-1, 1)
        level_sum = np.sum(y * level_weights, axis=0)
        level_sum_squared = np.sum(y * y * level_weights, axis=0)
        level_weight = np.sum(sample_weights)

        if is_left:
            stats = replace(
                stats,
                left_weight=level_weight,
                right_weight=stats.right_weight - level_weight,
                left_sum=level_sum,
                right_sum=stats.right_sum - level_sum,
                left_sum_squared=level_sum_squared,
                right_sum_squared=stats.right_sum_squared - level_sum_squared,
            )
        else:
            stats = replace(
                stats,
                left_weight=stats.left_weight - level_weight,
                right_weight=level_weight,
                left_sum=stats.left_sum - level_sum,
                right_sum=level_sum,
                left_sum_squared=stats.left_sum_squared - level_sum_squared,
            )
        return stats


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


def one_hot_encode(arr: np.ndarray) -> np.ndarray:
    """Convert integer array to one-hot encoded matrix.

    For binary labels (max value 1), returns column vector.
    For multiclass labels, returns one-hot encoded matrix.
    """
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")

    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Input array must contain integers")

    if arr.min() < 0:
        raise ValueError("Input array cannot contain negative values")

    if arr.max() == 1:
        return arr.reshape(-1, 1)

    unique_classes = np.unique(arr)
    n_classes = len(unique_classes)
    one_hot = np.zeros((len(arr), n_classes), dtype=np.uint8)
    for i, label in enumerate(unique_classes):
        one_hot[arr == label, i] = 1
    return one_hot


class Split(NamedTuple):
    criterion: float
    feature_idx: int
    split_value: split_value
    left_index: np.ndarray
    right_index: np.ndarray
    left_value: np.ndarray
    right_value: np.ndarray


def _best_numerical_split(
    x: np.ndarray,
    y: np.ndarray,
    feat_idx: int,
    criterion: Criterion,
    sample_weights: np.ndarray,
    min_samples_leaf: int,
) -> tuple[float, Split | None]:
    min_score = np.inf
    best_split = None
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    weights_sorted = sample_weights[sort_idx]

    stats = criterion.init_split_stats(y_sorted, weights_sorted)

    n_samples = len(y_sorted)
    for i in range(1, n_samples):
        if i < min_samples_leaf or (n_samples - i) < min_samples_leaf:
            continue

        criterion.update_split_stats(stats, y_sorted[i - 1], weights_sorted[i - 1])
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

    return min_score, best_split


def _best_categorical_split(
    x: np.ndarray,
    y: np.ndarray,
    feat_idx: int,
    criterion: Criterion,
    sample_weights: np.ndarray,
    min_samples_leaf: int,
):
    min_score = np.inf
    best_split = None
    unique_values = np.unique(x)

    stats = criterion.init_split_stats(y.astype(np.float64), sample_weights)

    # Pre-compute category indices
    cat_indices = {}
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    weights_sorted = sample_weights[sort_idx]

    start_idx = 0
    for i in range(1, len(x_sorted) + 1):
        if i == len(x_sorted) or x_sorted[i] != x_sorted[start_idx]:
            cat_indices[x_sorted[start_idx]] = {
                "indices": sort_idx[start_idx:i],
                "y": y_sorted[start_idx:i],
                "weights": weights_sorted[start_idx:i],
            }
            start_idx = i

    stats = criterion.init_split_stats(y, sample_weights)

    n_samples = len(y_sorted)
    for value in unique_values:
        cat_data = cat_indices[value]
        level_size = len(cat_data["indices"])
        if level_size == 0 or level_size == len(x):
            continue

        if level_size < min_samples_leaf or (n_samples - level_size) < min_samples_leaf:
            continue

        level_stats = criterion.make_stats_from_categorical_level(
            stats, cat_data["y"], cat_data["weights"], is_left=True
        )
        score = criterion.split_impurity(level_stats)

        if score < min_score:
            min_score = score
            left_indices = cat_data["indices"]
            right_indices = np.setdiff1d(np.arange(len(x)), left_indices)
            best_split = Split(
                criterion=score,
                feature_idx=feat_idx,
                split_value=set([value]),
                left_index=left_indices,
                right_index=right_indices,
                left_value=criterion.node_optimal_value(y[left_indices]),
                right_value=criterion.node_optimal_value(y[right_indices]),
            )

    return min_score, best_split


def _best_categorical_optimal_partitioning(
    x: np.ndarray,
    y: np.ndarray,
    feat_idx: int,
    criterion: Criterion,
    sample_weights: np.ndarray,
    min_samples_leaf: int,
):
    """Find optimal binary split for categorical feature using Fisher's method.

    For binary classification, this orders categories by their positive response rate
    and finds the optimal split point that maximizes the difference between groups.
    """
    min_score = np.inf
    best_split = None

    df = pd.DataFrame({"x": x, "y": y.ravel(), "w": sample_weights})

    cat_stats = df.groupby("x").agg(
        y_avg=pd.NamedAgg(
            column="y", aggfunc=lambda x: np.average(x, weights=df.loc[x.index, "w"])
        ),
        y_count=pd.NamedAgg(column="y", aggfunc=len),
        w=pd.NamedAgg(column="w", aggfunc="sum"),
    )

    if len(cat_stats) <= 1:
        return min_score, None

    cat_stats = cat_stats.sort_values("y_avg")

    stats = criterion.init_split_stats(
        y.astype(np.float64), sample_weights.astype(np.float64)
    )

    cat_to_order = {cat: i for i, cat in enumerate(cat_stats.index)}
    x_ordered = np.vectorize(cat_to_order.get)(x)

    n_samples = len(y)
    for i in range(1, len(cat_stats)):
        left_cats = cat_stats.index[:i]
        left_mask = x_ordered < i

        if not (np.any(left_mask) and np.any(~left_mask)):
            continue

        level_size = cat_stats.iloc[i - 1]["y_count"]
        if level_size < min_samples_leaf or (n_samples - level_size) < min_samples_leaf:
            continue

        cat_y = cat_stats.iloc[i - 1 : i]["y_avg"].values
        cat_w = cat_stats.iloc[i - 1 : i]["w"].values
        criterion.update_split_stats(stats, cat_y, cat_w)

        score = criterion.split_impurity(stats)

        if score < min_score:
            min_score = score
            best_split = Split(
                criterion=score,
                feature_idx=feat_idx,
                split_value=set(left_cats),
                left_index=np.flatnonzero(left_mask),
                right_index=np.flatnonzero(~left_mask),
                left_value=criterion.node_optimal_value(y[left_mask]),
                right_value=criterion.node_optimal_value(y[~left_mask]),
            )

    return min_score, best_split


def _find_best_split(
    X: pd.DataFrame,
    y: np.ndarray,
    criterion: Criterion,
    sample_weights: np.ndarray,
    min_samples_leaf: int,
    feature_indices: np.ndarray,
) -> Split | None:
    min_score = np.inf
    best_split = None

    categorical_splitter = (
        _best_categorical_optimal_partitioning
        if y.shape[1] == 1
        else _best_categorical_split
    )

    feature_types = np.array(
        [np.issubdtype(X.iloc[:, i].dtype, np.number) for i in range(X.shape[1])]
    )
    feature_values = [X.iloc[:, i].values for i in range(X.shape[1])]

    for feat_idx in feature_indices:
        splitter = (
            _best_numerical_split if feature_types[feat_idx] else categorical_splitter
        )

        score, split = splitter(
            feature_values[feat_idx],
            y,
            feat_idx,
            criterion,
            sample_weights,
            min_samples_leaf,
        )

        if split is not None and score < min_score:
            min_score = score
            best_split = split

    return best_split


def _get_feature_indices(
    n_features: int, max_features: int | float | None = None
) -> np.ndarray:
    """Get indices of features to consider for splitting.

    Args:
        n_features: Total number of features
        max_features: If int, consider max_features features.
                     If float, consider max_features * n_features features.
                     If None, consider all features.
    """
    if max_features is None:
        return np.arange(n_features)

    if isinstance(max_features, float):
        if not 0.0 < max_features <= 1.0:
            raise ValueError("max_features must be in (0, 1]")
        max_features = int(max_features * n_features)

    max_features = min(max_features, n_features)
    return np.random.choice(n_features, size=max_features, replace=False)


def get_splitter(
    criterion: Criterion,
    max_depth: int = 0,
    min_samples_leaf: int = 0,
    min_criterion_reduction: float = 0,
    max_features: int | float | None = None,
):
    def split_node(
        node: Node | LeafNode,
        X: pd.DataFrame,
        y: np.ndarray,
        value: np.ndarray,
        sample_weights: np.ndarray,
        depth: int,
    ) -> LeafNode | Node | None:
        if X.shape[0] <= 1 or (max_depth and depth >= max_depth):
            return LeafNode(value)

        if X.shape[0] < 2 * min_samples_leaf:
            return LeafNode(value)

        prior_criterion = criterion.node_impurity(y, sample_weights)
        if np.isclose(prior_criterion, 0):
            return LeafNode(value)

        feature_indices = _get_feature_indices(X.shape[1], max_features)
        split = _find_best_split(
            X, y, criterion, sample_weights, min_samples_leaf, feature_indices
        )
        if split is None:
            return None

        criterion_reduction = prior_criterion - split.criterion
        if min_criterion_reduction and criterion_reduction < min_criterion_reduction:
            return None

        X_left = X.iloc[split.left_index, :]
        X_right = X.iloc[split.right_index, :]
        y_left = y[split.left_index]
        y_right = y[split.right_index]

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
            sample_weights=sample_weights[split.left_index],
            depth=depth + 1,
        )
        right = split_node(
            node=node,
            X=X_right,
            y=y_right,
            value=split.right_value,
            sample_weights=sample_weights[split.right_index],
            depth=depth + 1,
        )

        if left is not None:
            node.left = left
        if right is not None:
            node.right = right

        return node

    return split_node


def print_tree(node: Node | LeafNode | None, depth: int = 0):
    if node is None:
        print("tree is not trained")
        return

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
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray | None = None
    ) -> None: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class Classifier(Protocol):
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray | None = None
    ) -> None: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


T = TypeVar("T", bound=Classifier | Regressor)


class BaseDecisionTree(Generic[T], ABC):
    def __init__(
        self,
        max_depth: int = 0,
        min_samples_leaf: int = 0,
        min_criterion_reduction: float = 0,
        max_features: int | float | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_criterion_reduction = min_criterion_reduction
        self.max_features = max_features
        self._root_node: Node | LeafNode | None = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weights: np.ndarray | None,
        criterion: Criterion,
    ) -> None:
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0], dtype=int)

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        node = LeafNode(np.mean(y, axis=0))
        node_splitter = get_splitter(
            criterion,
            self.max_depth,
            self.min_samples_leaf,
            self.min_criterion_reduction,
            self.max_features,
        )
        trained_node = node_splitter(
            node=node,
            X=X,
            y=y,
            value=np.mean(y, axis=0),
            sample_weights=sample_weights,
            depth=0,
        )
        node = trained_node if trained_node is not None else node
        self._root_node = node

    def _traverse_tree(self, x: pd.Series) -> np.ndarray:
        node = self._root_node
        if node is None:
            raise ValueError("model must be trained before prediction")

        while isinstance(node, Node):
            feature_val = x.iloc[node.feature_idx]
            if isinstance(node.split_value, set):
                node = node.left if feature_val in node.split_value else node.right
            else:
                node = node.left if feature_val <= node.split_value else node.right

        return node.value


class DecisionTreeClassifier(BaseDecisionTree[Classifier], Classifier):
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray | None = None
    ) -> None:
        y_oh = one_hot_encode(y)
        self._fit(X, y_oh, sample_weights, ClassificationCriterion(gini_impurity))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _prob_to_class(self.predict_proba(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = np.array([self._traverse_tree(x) for _, x in X.iterrows()])
        return y_pred


class DecisionTreeRegressor(BaseDecisionTree[Regressor], Regressor):
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray | None = None
    ) -> None:
        y = y.astype(np.float64)
        self._fit(X, y, sample_weights, SquaredLossCriterion())
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0], dtype=int)

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        node = LeafNode(np.mean(y, axis=0))
        node_splitter = get_splitter(
            SquaredLossCriterion(),
            self.max_depth,
            self.min_samples_leaf,
            self.min_criterion_reduction,
            self.max_features,
        )
        trained_node = node_splitter(
            node=node,
            X=X,
            y=y,
            value=np.mean(y, axis=0),
            sample_weights=sample_weights,
            depth=0,
        )
        node = trained_node if trained_node is not None else node
        self._root_node = node

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = np.array([self._traverse_tree(x) for _, x in X.iterrows()])
        return y_pred
