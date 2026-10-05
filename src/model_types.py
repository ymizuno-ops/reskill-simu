"""
model_types.py
==============
学習済みモデル辞書（models.pkl の中身）の型定義。
辞書のキー名は models.pkl・model_meta.json の形式そのものなので変えないこと。
"""

from __future__ import annotations
from typing import Protocol, TypedDict, Union

import numpy as np
import pandas as pd

Target = Union[pd.Series, np.ndarray]


class Regressor(Protocol):
    """fit / predict を持つ sklearn 互換の回帰モデル"""

    def fit(self, X: pd.DataFrame, y: Target) -> "Regressor": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class ModelMeta(TypedDict, total=False):
    r2_train: float
    r2_cv_mean: float
    r2_cv_std: float
    mae_train: float
    features: list[str]
    base_models: list[str]


class ModelEntry(TypedDict):
    pipeline: Regressor
    meta: ModelMeta
    label: str
    desc: str
    uses_fe: bool


ModelDict = dict[str, ModelEntry]
MacroParams = dict[str, float]
