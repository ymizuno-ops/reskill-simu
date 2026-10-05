"""
model_types.py
==============
学習済みモデル辞書（models.pkl の中身）の型定義。
辞書のキー名は models.pkl・model_meta.json の形式そのものなので変えないこと。
"""

# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# typing: 型ヒント用の部品。Protocol = 必要なメソッドで型を表す、TypedDict = キーごとに値の型が決まった辞書、Union = いずれかの型。
from typing import Protocol, TypedDict, Union

# numpy: 数値計算のライブラリ。`as np` は短い別名で使うための書き方。
import numpy as np
# pandas: 表形式のデータ（DataFrame）を扱うライブラリ。
import pandas as pd

# 型の別名。正解データ（年収）は pandas の Series（1列分のデータ）か numpy の配列のどちらでもよい、という意味。
Target = Union[pd.Series, np.ndarray]


# Protocol を継承したクラス。「fit と predict を持っていれば Regressor とみなす」という型の約束事で、このクラス自体からオブジェクトは作らない。
class Regressor(Protocol):
    """fit / predict を持つ sklearn 互換の回帰モデル"""

    # 末尾の `...` は「中身は書かない」という意味。形（引数と戻り値の型）だけを示す。self はメソッドが属するオブジェクト自身。
    def fit(self, X: pd.DataFrame, y: Target) -> "Regressor": ...

    # 予測値の配列を返すメソッドの形。
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


# モデルの精度情報の辞書の型。total=False は、キーが全部そろっていなくてもよいという指定。
class ModelMeta(TypedDict, total=False):
    r2_train: float
    r2_cv_mean: float
    r2_cv_std: float
    mae_train: float
    features: list[str]
    base_models: list[str]


# models.pkl に保存する1モデル分の辞書の型。「キー: 値の型」を並べて宣言する。
class ModelEntry(TypedDict):
    pipeline: Regressor
    meta: ModelMeta
    label: str
    desc: str
    uses_fe: bool


# モデル名（"ridge" など）→ ModelEntry の辞書。アプリ全体でこの型を受け渡す。
ModelDict = dict[str, ModelEntry]
# マクロ経済パラメータ（macro_params.json の中身）の型。
MacroParams = dict[str, float]
