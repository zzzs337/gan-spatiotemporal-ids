from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class TrafficPreprocessor:
    label_column: str
    categorical_columns: list[str]
    feature_columns: list[str] | None = None
    transformer: ColumnTransformer | None = None
    label_encoder: LabelEncoder | None = None

    def fit(self, frame: pd.DataFrame) -> "TrafficPreprocessor":
        frame = self._clean(frame)
        self.feature_columns = [c for c in frame.columns if c != self.label_column]
        categorical = [c for c in self.categorical_columns if c in self.feature_columns]
        numeric = [c for c in self.feature_columns if c not in categorical]
        self.transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "one_hot",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        self.label_encoder = LabelEncoder()
        self.transformer.fit(frame[self.feature_columns])
        self.label_encoder.fit(frame[self.label_column].astype(str))
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.transformer is None or self.label_encoder is None or self.feature_columns is None:
            raise RuntimeError("The preprocessor has not been fitted.")
        frame = self._clean(frame)
        missing = sorted(set(self.feature_columns + [self.label_column]) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        features = self.transformer.transform(frame[self.feature_columns]).astype(np.float32)
        labels = self.label_encoder.transform(frame[self.label_column].astype(str)).astype(np.int64)
        return features, labels

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None or self.feature_columns is None:
            raise RuntimeError("The preprocessor has not been fitted.")
        frame = self._clean(frame)
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        return self.transformer.transform(frame[self.feature_columns]).astype(np.float32)

    @property
    def classes(self) -> list[str]:
        if self.label_encoder is None:
            raise RuntimeError("The preprocessor has not been fitted.")
        return self.label_encoder.classes_.tolist()

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrafficPreprocessor":
        return joblib.load(path)

    @staticmethod
    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.replace([np.inf, -np.inf], np.nan).copy()


def make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) != len(labels):
        raise ValueError("Feature and label counts differ.")
    if len(features) < window_size:
        raise ValueError(f"At least {window_size} rows are required to create one window.")
    starts = range(0, len(features) - window_size + 1, stride)
    windows = np.stack([features[start : start + window_size] for start in starts])
    targets = np.asarray([labels[start + window_size - 1] for start in starts])
    return windows.astype(np.float32), targets.astype(np.int64)


def infer_categorical_columns(frame: pd.DataFrame, excluded: Iterable[str]) -> list[str]:
    excluded = set(excluded)
    return [
        column
        for column in frame.columns
        if column not in excluded
        and (pd.api.types.is_object_dtype(frame[column]) or isinstance(frame[column].dtype, pd.CategoricalDtype))
    ]

