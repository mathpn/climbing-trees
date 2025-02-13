from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np
import pandas as pd

from src.cart import (
    Classifier,
    Regressor,
)

T = TypeVar("T", bound=Classifier | Regressor)


class BaseBagging(Generic[T], ABC):
    """Base class for Bagging implementations."""

    def __init__(
        self,
        estimator_constructor: Callable[[], T],
        n_estimators: int = 100,
        random_state: int | None = None,
    ) -> None:
        """Initialize Bagging ensemble.

        Args:
            estimator_constructor: Function that returns a new estimator instance
            n_estimators: Number of estimators in the ensemble
            random_state: Random state for reproducibility
        """
        self.estimator_constructor = estimator_constructor
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._estimators: list[T] = []

        if random_state is not None:
            np.random.seed(random_state)

    def _build_estimators(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weights: np.ndarray | None,
    ) -> list[T]:
        """Build estimators sequentially."""
        estimators = []
        for _ in range(self.n_estimators):
            model = self.estimator_constructor()

            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            sample_weight = None if sample_weights is None else sample_weights[indices]
            model.fit(X.iloc[indices], y[indices], sample_weight)

            estimators.append(model)

        return estimators

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Fit the bagging ensemble."""
        self._estimators = self._build_estimators(X, y, sample_weights)


class BaggingClassifier(BaseBagging[Classifier], Classifier):
    """Bagging Classifier implementation."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for X."""
        all_proba = np.array([model.predict_proba(X) for model in self._estimators])
        return np.mean(all_proba, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for X."""
        proba = self.predict_proba(X)
        return (
            (proba >= 0.5).astype(int)
            if proba.shape[1] == 1
            else np.argmax(proba, axis=1)
        )


class BaggingRegressor(BaseBagging[Regressor], Regressor):
    """Bagging Regressor implementation."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regression target for X."""
        all_predictions = np.array([model.predict(X) for model in self._estimators])
        return np.mean(all_predictions, axis=0)
