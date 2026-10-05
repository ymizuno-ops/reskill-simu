"""
model_wrappers.py
=================
sklearn 互換の Wrapper class と、StackingEnsemble が使う add_features。
pickle で保存できるよう、すべてモジュールレベルで定義する（step3_train.py から分離）。
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from sklearn.linear_model    import Ridge
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.base            import BaseEstimator, RegressorMixin

_HERE = os.path.dirname(os.path.abspath(__file__))
CATBOOST_TRAIN_DIR = os.path.join(_HERE, "..", "tmp", "catboost_info")


# ──────────────────────────────────────────────────────
# 共通前処理
# ──────────────────────────────────────────────────────
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    Xc["age_sq"]         = Xc["age"] ** 2 / 1000
    Xc["age_x_exp"]      = Xc["age"] * Xc["experience_years"] / 100
    Xc["exp_ratio"]      = Xc["experience_years"] / Xc["age"].clip(lower=1)
    Xc["prime_age_flag"] = ((Xc["age"] >= 35) & (Xc["age"] <= 54)).astype(float)
    return Xc


# ══════════════════════════════════════════════════════
# Wrapper クラス（モジュールレベル定義 ← pickle保存に必須）
# ══════════════════════════════════════════════════════

class LGBMWrapper(BaseEstimator, RegressorMixin):
    """
    LightGBM の sklearn 互換ラッパー。
    occupation を LabelEncoding してカテゴリ特徴として渡す。
    """
    def __init__(self,
                 n_estimators=500, learning_rate=0.05, num_leaves=63,
                 min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.1, reg_lambda=1.0, random_state=42):
        self.n_estimators    = n_estimators
        self.learning_rate   = learning_rate
        self.num_leaves      = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample       = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha       = reg_alpha
        self.reg_lambda      = reg_lambda
        self.random_state    = random_state

    def fit(self, X, y):
        import lightgbm as lgb
        self.le_ = LabelEncoder()
        Xc = X.copy()
        Xc["occupation"] = self.le_.fit_transform(Xc["occupation"].astype(str))
        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model_.fit(Xc, y, categorical_feature=[0])
        return self

    def predict(self, X):
        Xc = X.copy()
        known = set(self.le_.classes_)
        Xc["occupation"] = Xc["occupation"].apply(
            lambda v: v if v in known else self.le_.classes_[0]
        )
        Xc["occupation"] = self.le_.transform(Xc["occupation"].astype(str))
        return self.model_.predict(Xc)


class CatBoostWrapper(BaseEstimator, RegressorMixin):
    """
    CatBoost の sklearn 互換ラッパー。
    occupation を文字列のまま cat_features に指定できる。
    """
    def __init__(self,
                 iterations=500, learning_rate=0.05, depth=8,
                 l2_leaf_reg=3.0, min_data_in_leaf=10, random_state=42):
        self.iterations       = iterations
        self.learning_rate    = learning_rate
        self.depth            = depth
        self.l2_leaf_reg      = l2_leaf_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.random_state     = random_state

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        os.makedirs(CATBOOST_TRAIN_DIR, exist_ok=True)  # tmp/ が無いと CatBoost が作れず失敗する
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            min_data_in_leaf=self.min_data_in_leaf,
            random_state=self.random_state,
            verbose=0,
            thread_count=-1,
            train_dir=CATBOOST_TRAIN_DIR,
        )
        self.model_.fit(X, y, cat_features=["occupation"])
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ══════════════════════════════════════════════════════
# Stacking Ensemble
# ══════════════════════════════════════════════════════
class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    全ベースモデルのOOF（Out-of-Fold）予測をメタ特徴量として
    Ridgeメタモデルで最終予測する2層アンサンブル。

    設計:
      Layer1 (base models): 訓練済みの全モデル（FEの有無を自動判定）
      Layer2 (meta model) : Ridge回帰
        - 入力: 各ベースモデルの予測値 + age + experience_years
        - Ridge を使う理由: シンプルで過学習しにくく、
          各モデルへの重みを線形結合で学習できる

    FE_KEYS に含まれるモデルは predict 前に add_features を適用する。
    """

    FE_KEYS = frozenset({"custom", "xgboost", "elasticnet", "gradient_boosting"})

    def __init__(self, base_models: dict, n_splits: int = 5, meta_alpha: float = 1.0):
        """
        Parameters
        ----------
        base_models : dict
            step3_train.main() が返す models 辞書
            {"model_key": {"pipeline": ..., ...}, ...}
        n_splits    : OOFのfold数
        meta_alpha  : メタRidgeの正則化強度
        """
        self.base_models = base_models
        self.n_splits    = n_splits
        self.meta_alpha  = meta_alpha

    # ── 内部メソッド ──────────────────────────────
    def _prepare_X(self, X: pd.DataFrame, key: str) -> pd.DataFrame:
        """モデルキーに応じて特徴量エンジニアリングを適用"""
        return add_features(X) if key in self.FE_KEYS else X.copy()

    def _make_oof_matrix(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """
        全ベースモデルのOOF予測行列を作成する。
        shape: (n_samples, n_base_models)
        """
        n         = len(y)
        model_keys = list(self.base_models.keys())
        oof_matrix = np.zeros((n, len(model_keys)))
        kf         = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            X_tr  = X.iloc[tr_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
            y_tr  = y[tr_idx]

            for col_idx, key in enumerate(model_keys):
                import copy
                # モデルのディープコピーを fold ごとに再訓練
                model_entry = self.base_models[key]
                cloned = copy.deepcopy(model_entry["pipeline"])
                cloned.fit(self._prepare_X(X_tr, key), y_tr)
                oof_matrix[val_idx, col_idx] = cloned.predict(
                    self._prepare_X(X_val, key)
                )

        return oof_matrix

    def _make_meta_X(self, oof_or_pred: np.ndarray,
                     X: pd.DataFrame) -> np.ndarray:
        """
        メタ特徴量 = ベースモデル予測値 + age + experience_years
        age/experience_years を追加することで「年齢帯の系統誤差」を補正できる
        """
        structural = X[["age", "experience_years"]].values
        return np.hstack([oof_or_pred, structural])

    # ── 公開メソッド ──────────────────────────────
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        model_keys = list(self.base_models.keys())

        # Layer1: OOF予測行列を作成
        print(f"    [Stacking] OOF予測中 ({len(model_keys)}モデル × {self.n_splits}fold)...",
              end="", flush=True)
        oof_matrix = self._make_oof_matrix(X, y)
        print(" 完了")

        # Layer1: 全データでベースモデルを再訓練（最終予測用）
        self.fitted_bases_ = {}
        for key in model_keys:
            import copy
            cloned = copy.deepcopy(self.base_models[key]["pipeline"])
            cloned.fit(self._prepare_X(X, key), y)
            self.fitted_bases_[key] = cloned

        # Layer2: メタモデルを訓練
        meta_X = self._make_meta_X(oof_matrix, X)
        self.meta_model_ = Ridge(alpha=self.meta_alpha)
        self.meta_model_.fit(meta_X, y)
        self.model_keys_ = model_keys
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 各ベースモデルの予測を並べる
        base_preds = np.column_stack([
            self.fitted_bases_[key].predict(self._prepare_X(X, key))
            for key in self.model_keys_
        ])
        meta_X = self._make_meta_X(base_preds, X)
        return self.meta_model_.predict(meta_X)
