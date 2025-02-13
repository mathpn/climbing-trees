from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import pandas as pd

from .cart import (
    Classifier,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Regressor,
)

T = TypeVar("T", DecisionTreeClassifier, DecisionTreeRegressor)


class BaseRandomForest(Generic[T], ABC):
    """Base class for Random Forest implementations."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 0,
        min_samples_leaf: int = 1,
        min_criterion_reduction: float = 0,
        max_features: int | float | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize Random Forest.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_leaf: Minimum samples required at leaf node
            min_criterion_reduction: Minimum criterion reduction required for splitting
            max_features: Number of features to consider for best split:
                         If int, consider max_features features
                         If float, consider max_features * n_features features
                         If None, consider sqrt(n_features) for classification
                         and n_features/3 for regression
            bootstrap: Whether to use bootstrap samples

            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_criterion_reduction = min_criterion_reduction
        self.max_features = max_features
        self.bootstrap = bootstrap

        self.random_state = random_state

        self._estimators: list[T] = []
        self.feature_importances_: np.ndarray | None = None

        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def _make_estimator(self) -> T:
        """Create a new decision tree instance."""
        pass

    def _build_trees(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weights: np.ndarray | None,
    ) -> list[T]:
        """Build trees sequentially."""
        trees = []
        for _ in range(self.n_estimators):
            tree = self._make_estimator()

            if self.bootstrap:
                indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
                sample_weight = (
                    None if sample_weights is None else sample_weights[indices]
                )
                tree.fit(X.iloc[indices], y[indices], sample_weight)
            else:
                tree.fit(X, y, sample_weights)

            trees.append(tree)

        return trees

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Fit the random forest."""
        self._n_features = X.shape[1]
        self._estimators = self._build_trees(X, y, sample_weights)


class RandomForestClassifier(BaseRandomForest[DecisionTreeClassifier], Classifier):
    """Random Forest Classifier implementation."""

    def _make_estimator(self) -> DecisionTreeClassifier:
        max_features = self.max_features
        if max_features is None:
            max_features = int(np.sqrt(self._n_features))

        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_criterion_reduction=self.min_criterion_reduction,
            max_features=max_features,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for X."""
        all_proba = np.array([tree.predict_proba(X) for tree in self._estimators])
        return np.mean(all_proba, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for X."""
        proba = self.predict_proba(X)
        return (
            (proba >= 0.5).astype(int)
            if proba.shape[1] == 1
            else np.argmax(proba, axis=1)
        )


class RandomForestRegressor(BaseRandomForest[DecisionTreeRegressor], Regressor):
    """Random Forest Regressor implementation."""

    def _make_estimator(self) -> DecisionTreeRegressor:
        max_features = self.max_features
        if max_features is None:
            max_features = max(1, int(self._n_features / 3))

        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_criterion_reduction=self.min_criterion_reduction,
            max_features=max_features,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regression target for X."""
        all_predictions = np.array([tree.predict(X) for tree in self._estimators])
        return np.mean(all_predictions, axis=0)
