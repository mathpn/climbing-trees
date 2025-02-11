from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
import numpy as np

from src.cart import (
    Classifier,
    Regressor,
    DecisionTreeClassifier,
    _prob_to_class,
)


class BaggingClassifier(Classifier):
    def __init__(
        self,
        estimator_constructor: Callable[[], Classifier],
        n_estimators: int,
    ) -> None:
        self.estimator_constructor = estimator_constructor
        self.n_estimators = n_estimators
        self._estimators: list[Classifier] = []

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray | None = None
    ) -> None:
        self._estimators = []
        for _ in range(self.n_estimators):
            model = self.estimator_constructor()
            model.fit(*_bootstrap_sample(X, y, sample_weights))
            self._estimators.append(model)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _prob_to_class(self.predict_proba(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._estimators:
            raise ValueError("model must be trained before prediction")

        preds = [model.predict_proba(X) for model in self._estimators]
        pred_arr = np.stack(preds, axis=2)
        return pred_arr.mean(axis=2)


class BaggingRegressor(Regressor):
    def __init__(
        self,
        estimator_constructor: Callable[[], Regressor],
        n_estimators: int,
    ) -> None:
        self.estimator_constructor = estimator_constructor
        self.n_estimators = n_estimators
        self._estimators: list[Regressor] = []

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray | None = None
    ) -> None:
        self._estimators = []
        for _ in range(self.n_estimators):
            model = self.estimator_constructor()
            model.fit(*_bootstrap_sample(X, y, sample_weights))
            self._estimators.append(model)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._estimators:
            raise ValueError("model must be trained before prediction")

        preds = [model.predict(X) for model in self._estimators]
        pred_arr = np.stack(preds, axis=2)
        return pred_arr.mean(axis=2)


def _bootstrap_sample(
    X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    idx = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)

    X_sample = X.iloc[idx]
    y_sample = y[idx]
    if sample_weights is not None:
        sample_weights = sample_weights[idx]

    return X_sample, y_sample, sample_weights
