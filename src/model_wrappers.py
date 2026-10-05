"""
model_wrappers.py
=================
sklearn 互換の Wrapper class と、StackingEnsemble が使う addFeatures。
pickle で保存できるよう、すべてモジュールレベルで定義する（step3_train.py から分離）。
"""

from __future__ import annotations
import copy
import os
import pandas as pd
import numpy as np
from sklearn.linear_model    import Ridge
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.base            import BaseEstimator, RegressorMixin

from log_config import getLogger
from model_types import ModelDict, Regressor, Target

logger = getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
CATBOOST_TRAIN_DIR = os.path.join(_HERE, "..", "tmp", "catboost_info")
RANDOM_STATE = 42

# 特徴量エンジニアリングの係数
AGE_SQ_SCALE      = 1000   # age² を他の特徴量と同程度の桁にそろえる
AGE_X_EXP_SCALE   = 100
PRIME_AGE_FROM    = 35     # 給与ピーク帯
PRIME_AGE_TO      = 54
UNKNOWN_CATEGORY_INDEX = 0  # LightGBM: 未知の職種は先頭クラスとして扱う


# ──────────────────────────────────────────────────────
# 共通前処理
# ──────────────────────────────────────────────────────
def addFeatures(X: pd.DataFrame) -> pd.DataFrame:
    xc = X.copy()
    xc["age_sq"]         = xc["age"] ** 2 / AGE_SQ_SCALE
    xc["age_x_exp"]      = xc["age"] * xc["experience_years"] / AGE_X_EXP_SCALE
    xc["exp_ratio"]      = xc["experience_years"] / xc["age"].clip(lower=1)
    xc["prime_age_flag"] = ((xc["age"] >= PRIME_AGE_FROM) & (xc["age"] <= PRIME_AGE_TO)).astype(float)
    return xc


# ══════════════════════════════════════════════════════
# Wrapper クラス（モジュールレベル定義 ← pickle保存に必須）
# ══════════════════════════════════════════════════════

class LGBMWrapper(BaseEstimator, RegressorMixin):
    """
    LightGBM の sklearn 互換ラッパー。
    occupation を LabelEncoding してカテゴリ特徴として渡す。
    """
    def __init__(self,
                 n_estimators: int = 500, learning_rate: float = 0.05, num_leaves: int = 63,
                 min_child_samples: int = 10, subsample: float = 0.8, colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1, reg_lambda: float = 1.0, random_state: int = RANDOM_STATE) -> None:
        self.n_estimators    = n_estimators
        self.learning_rate   = learning_rate
        self.num_leaves      = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample       = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha       = reg_alpha
        self.reg_lambda      = reg_lambda
        self.random_state    = random_state

    def fit(self, X: pd.DataFrame, y: Target) -> "LGBMWrapper":
        import lightgbm as lgb
        self.le_ = LabelEncoder()
        xc = X.copy()
        xc["occupation"] = self.le_.fit_transform(xc["occupation"].astype(str))
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
        self.model_.fit(xc, y, categorical_feature=[0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        xc = X.copy()
        known = set(self.le_.classes_)
        fallback = self.le_.classes_[UNKNOWN_CATEGORY_INDEX]
        xc["occupation"] = xc["occupation"].apply(lambda v: v if v in known else fallback)
        xc["occupation"] = self.le_.transform(xc["occupation"].astype(str))
        return self.model_.predict(xc)


class CatBoostWrapper(BaseEstimator, RegressorMixin):
    """
    CatBoost の sklearn 互換ラッパー。
    occupation を文字列のまま cat_features に指定できる。
    """
    def __init__(self,
                 iterations: int = 500, learning_rate: float = 0.05, depth: int = 8,
                 l2_leaf_reg: float = 3.0, min_data_in_leaf: int = 10,
                 random_state: int = RANDOM_STATE) -> None:
        self.iterations       = iterations
        self.learning_rate    = learning_rate
        self.depth            = depth
        self.l2_leaf_reg      = l2_leaf_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.random_state     = random_state

    def fit(self, X: pd.DataFrame, y: Target) -> "CatBoostWrapper":
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
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

    FE_KEYS に含まれるモデルは predict 前に addFeatures を適用する。
    """

    FE_KEYS = frozenset({"custom", "xgboost", "elasticnet", "gradient_boosting"})

    def __init__(self, base_models: ModelDict, n_splits: int = 5, meta_alpha: float = 1.0) -> None:
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
    def _prepareX(self, X: pd.DataFrame, key: str) -> pd.DataFrame:
        """モデルキーに応じて特徴量エンジニアリングを適用"""
        return addFeatures(X) if key in self.FE_KEYS else X.copy()

    def _fitClone(self, key: str, X: pd.DataFrame, y: np.ndarray) -> Regressor:
        """ベースモデルをディープコピーして訓練する（元のモデルは変えない）"""
        cloned = copy.deepcopy(self.base_models[key]["pipeline"])
        cloned.fit(self._prepareX(X, key), y)
        return cloned

    def _makeOofMatrix(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """
        全ベースモデルのOOF予測行列を作成する。
        shape: (n_samples, n_base_models)
        """
        modelKeys = list(self.base_models.keys())
        oofMatrix = np.zeros((len(y), len(modelKeys)))
        kf        = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)

        for trainIdx, valIdx in kf.split(X):
            xTrain = X.iloc[trainIdx].reset_index(drop=True)
            xVal   = X.iloc[valIdx].reset_index(drop=True)

            for colIdx, key in enumerate(modelKeys):
                # モデルのディープコピーを fold ごとに再訓練
                cloned = self._fitClone(key, xTrain, y[trainIdx])
                oofMatrix[valIdx, colIdx] = cloned.predict(self._prepareX(xVal, key))

        return oofMatrix

    def _makeMetaX(self, oofOrPred: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """
        メタ特徴量 = ベースモデル予測値 + age + experience_years
        age/experience_years を追加することで「年齢帯の系統誤差」を補正できる
        """
        structural = X[["age", "experience_years"]].values
        return np.hstack([oofOrPred, structural])

    # ── 公開メソッド ──────────────────────────────
    def fit(self, X: pd.DataFrame, y: Target) -> "StackingEnsemble":
        yArr = np.asarray(y)
        modelKeys = list(self.base_models.keys())

        # Layer1: OOF予測行列を作成
        logger.info("    [Stacking] OOF予測中 (%dモデル × %dfold)...", len(modelKeys), self.n_splits)
        oofMatrix = self._makeOofMatrix(X, yArr)
        logger.info("    [Stacking] OOF予測 完了")

        # Layer1: 全データでベースモデルを再訓練（最終予測用）
        # 属性名は sklearn の規約（末尾 _）と models.pkl の互換のため変えない
        self.fitted_bases_ = {key: self._fitClone(key, X, yArr) for key in modelKeys}

        # Layer2: メタモデルを訓練
        self.meta_model_ = Ridge(alpha=self.meta_alpha)
        self.meta_model_.fit(self._makeMetaX(oofMatrix, X), yArr)
        self.model_keys_ = modelKeys
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 各ベースモデルの予測を並べる
        basePreds = np.column_stack([
            self.fitted_bases_[key].predict(self._prepareX(X, key))
            for key in self.model_keys_
        ])
        return self.meta_model_.predict(self._makeMetaX(basePreds, X))
