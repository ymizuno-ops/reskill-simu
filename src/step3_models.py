"""
step3_models.py
===============
Step3 で訓練する各モデル（sklearn 5モデル + LightGBM / CatBoost / XGBoost）の
定義と、CV 評価つきの訓練処理（step3_train.py から分離）。
"""

from __future__ import annotations
import time
import pandas as pd
import numpy as np
from sklearn.linear_model    import Ridge, ElasticNet
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics         import r2_score, mean_absolute_error

from log_config import getLogger
from model_types import ModelMeta, Regressor, Target
from model_wrappers import addFeatures, LGBMWrapper, CatBoostWrapper, RANDOM_STATE

logger = getLogger(__name__)

CV_FOLDS = 5
CV = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
ALL_CORES = -1
LABEL_WIDTH = 32
METRIC_DIGITS = 4
MAE_DIGITS = 2

BASE_FEATURES = ["occupation", "age", "experience_years"]
BASE_NUM_FEATURES = ["age", "experience_years"]
FE_NUM_FEATURES = ["age", "experience_years", "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
FE_FEATURES = ["occupation", *FE_NUM_FEATURES]

# CatBoost: CV は軽量版で高速化し、最終モデルのみ高精度で訓練する
CATBOOST_CV_PARAMS    = {"iterations": 200, "learning_rate": 0.1, "depth": 6}
CATBOOST_FINAL_PARAMS = {"iterations": 500, "learning_rate": 0.05, "depth": 8}


# ──────────────────────────────────────────────────────
# 共通処理
# ──────────────────────────────────────────────────────
def makeOhePreprocessor(numFeatures: list[str]) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["occupation"]),
        ("num", StandardScaler(), numFeatures),
    ])


def buildMeta(r2: float, cvScores: np.ndarray, mae: float, features: list[str]) -> ModelMeta:
    return {
        "r2_train":   round(r2, METRIC_DIGITS),
        "r2_cv_mean": round(cvScores.mean(), METRIC_DIGITS),
        "r2_cv_std":  round(cvScores.std(), METRIC_DIGITS),
        "mae_train":  round(mae, MAE_DIGITS),
        "features":   features,
    }


def logScore(label: str, r2: float, cvScores: np.ndarray, mae: float, startedAt: float) -> None:
    logger.info("  %s R²=%.4f  CV=%.4f±%.4f  MAE=%.1f万円  (%.1fs)",
                f"{label:<{LABEL_WIDTH}}", r2, cvScores.mean(), cvScores.std(), mae,
                time.time() - startedAt)


def _evaluate(model: Regressor, X: pd.DataFrame, y: Target, label: str,
              cvScores: np.ndarray, features: list[str], startedAt: float) -> ModelMeta:
    """訓練済みモデルの訓練データでの精度をログに出し、meta を返す"""
    pred = model.predict(X)
    r2, mae = r2_score(y, pred), mean_absolute_error(y, pred)
    logScore(label, r2, cvScores, mae, startedAt)
    return buildMeta(r2, cvScores, mae, features)


def _cvAndFit(pipe: Pipeline, X: pd.DataFrame, y: Target, label: str,
              features: list[str]) -> tuple[Pipeline, ModelMeta]:
    startedAt = time.time()
    cvScores = cross_val_score(pipe, X, y, cv=CV, scoring="r2")
    pipe.fit(X, y)
    return pipe, _evaluate(pipe, X, y, label, cvScores, features, startedAt)


def _feModel(model: Regressor, X: pd.DataFrame, y: Target, label: str) -> tuple[Pipeline, ModelMeta]:
    """特徴量エンジニアリング込みの OHE パイプラインで訓練する"""
    pipe = Pipeline([("pre", makeOhePreprocessor(FE_NUM_FEATURES)), ("model", model)])
    return _cvAndFit(pipe, addFeatures(X), y, label, FE_FEATURES)


def _baseModel(model: Regressor, X: pd.DataFrame, y: Target, label: str) -> tuple[Pipeline, ModelMeta]:
    pipe = Pipeline([("pre", makeOhePreprocessor(BASE_NUM_FEATURES)), ("model", model)])
    return _cvAndFit(pipe, X, y, label, BASE_FEATURES)


# ──────────────────────────────────────────────────────
# sklearn モデル
# ──────────────────────────────────────────────────────
def trainRidge(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    return _baseModel(Ridge(alpha=10.0), X, y, "Ridge Regression")


def trainRandomForest(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_leaf=3, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=ALL_CORES,
    )
    return _baseModel(model, X, y, "Random Forest")


def trainCustomRidge(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    return _feModel(Ridge(alpha=1.0), X, y, "Custom Ridge (+FE)")


def trainElasticnet(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    """
    L1（Lasso）+ L2（Ridge）の両正則化を組み合わせたモデル。
    不要な特徴量を自動で0にする効果（スパース性）があり解釈性が高い。
    特徴量エンジニアリング込みで使用する。
    """
    model = ElasticNet(
        alpha=0.001,     # 正則化強度（チューニング済み）
        l1_ratio=0.7,    # L1:L2 = 70:30（Lasso寄り・スパース性重視）
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    return _feModel(model, X, y, "ElasticNet (+FE)")


def trainGradientBoosting(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    """
    sklearn 標準の勾配ブースティング。追加インストール不要。
    XGBoost・LightGBMより低速だが安定性が高く、過学習に強い。
    特徴量エンジニアリング込みで使用する。
    """
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    return _feModel(model, X, y, "GradientBoosting (+FE)")


# ──────────────────────────────────────────────────────
# 勾配ブースティング系（追加ライブラリ）
# ──────────────────────────────────────────────────────
def trainLightgbm(X: pd.DataFrame, y: Target) -> tuple[LGBMWrapper, ModelMeta]:
    startedAt = time.time()
    logger.info("  %s CV中...", f"{'LightGBM':<{LABEL_WIDTH}}")
    wrapper = LGBMWrapper()
    cvScores = cross_val_score(wrapper, X, y, cv=CV, scoring="r2")
    wrapper.fit(X, y)
    return wrapper, _evaluate(wrapper, X, y, "LightGBM", cvScores, BASE_FEATURES, startedAt)


def trainCatboost(X: pd.DataFrame, y: Target) -> tuple[CatBoostWrapper, ModelMeta]:
    """CV は軽量版で高速化し、最終モデルのみ高精度パラメータで訓練"""
    startedAt = time.time()
    logger.info("  %s CV中（%diter）...", f"{'CatBoost':<{LABEL_WIDTH}}", CATBOOST_CV_PARAMS["iterations"])
    cvScores = cross_val_score(CatBoostWrapper(**CATBOOST_CV_PARAMS), X, y, cv=CV, scoring="r2")
    logger.info("  %s CV完了(%.0fs) → 最終訓練(%diter)...", f"{'CatBoost':<{LABEL_WIDTH}}",
                time.time() - startedAt, CATBOOST_FINAL_PARAMS["iterations"])

    final = CatBoostWrapper(**CATBOOST_FINAL_PARAMS)
    final.fit(X, y)
    return final, _evaluate(final, X, y, "CatBoost", cvScores, BASE_FEATURES, startedAt)


def trainXgboost(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=ALL_CORES,
        verbosity=0,
    )
    return _feModel(model, X, y, "XGBoost (+FE)")
