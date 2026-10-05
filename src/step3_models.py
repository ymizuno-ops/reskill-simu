"""
step3_models.py
===============
Step3 で訓練する各モデル（sklearn 5モデル + LightGBM / CatBoost / XGBoost）の
定義と、CV 評価つきの訓練処理（step3_train.py から分離）。
"""

# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# time: 時刻を扱う標準モジュール。学習にかかった秒数を測るのに使う。
import time
import pandas as pd
import numpy as np
# 線形回帰のモデル（予測を抑える仕組み = 正則化の種類が違う）。
from sklearn.linear_model    import Ridge, ElasticNet
# 決定木（条件分岐の木）を多数組み合わせるモデル。
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
# OneHotEncoder: 職種名を「その職種なら 1、他は 0」の列に展開する。StandardScaler: 数値を平均 0・ばらつき 1 にそろえる。
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
# Pipeline: 前処理とモデルを1つにつなげ、fit・predict を一度で呼べるようにする。
from sklearn.pipeline        import Pipeline
# ColumnTransformer: 列ごとに違う前処理をかける。
from sklearn.compose         import ColumnTransformer
# cross_val_score: 交差検証で精度を測る関数。
from sklearn.model_selection import cross_val_score, KFold
# R²（決定係数。1 に近いほど当てはまりが良い）と、MAE（予測と実際の差の絶対値の平均）を計算する関数。
from sklearn.metrics         import r2_score, mean_absolute_error

from log_config import getLogger
from model_types import ModelMeta, Regressor, Target
from model_wrappers import addFeatures, LGBMWrapper, CatBoostWrapper, RANDOM_STATE

logger = getLogger(__name__)

# 意味: 精度評価（交差検証 = データを分けて、学習に使っていない部分を当てられるか試す方法）の分割数。
# 影響: 増やすと精度の数値が安定するが、学習時間が延びる。画面の精度カードの CV の値が変わる。
CV_FOLDS = 5
# 全モデルで共通の分割方法。同じ分け方で比べるので、精度を公平に比較できる。
CV = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
# 意味: 学習に使う CPU コアの数（-1 = すべて使う）。
# 影響: 速さだけが変わり、結果は変わらない。他の作業が重くなる場合は 2 などに減らす。
ALL_CORES = -1
# 意味: 実行ログでモデル名を表示する幅（文字数）。見た目だけで結果は変わらない。
LABEL_WIDTH = 32
# 意味: model_meta.json と画面の精度カードに出す数値の小数点以下の桁数（R² と MAE）。
METRIC_DIGITS = 4
MAE_DIGITS = 2

# 意味: 予測に使う入力項目（職種・年齢・経験年数）。
# 注意: 学習データの CSV の列名そのもの。変えるとデータ・画面と合わなくなるので変えない。
BASE_FEATURES = ["occupation", "age", "experience_years"]
BASE_NUM_FEATURES = ["age", "experience_years"]
# 意味: 特徴量強化型のモデルが使う数値の項目（年齢² など、addFeatures が作る列を含む）。
# 注意: model_wrappers.py の addFeatures が作る列名と一致させる。
FE_NUM_FEATURES = ["age", "experience_years", "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
# * でリストを展開して、別のリストの中に並べる（["occupation", "age", ...] と書いたのと同じ）。
FE_FEATURES = ["occupation", *FE_NUM_FEATURES]

# CatBoost: CV は軽量版で高速化し、最終モデルのみ高精度で訓練する
# 意味: CatBoost の精度評価に使う軽い設定（木の数を減らして時間を短くする）。
# 影響: iterations を増やすと評価が正確になるが、学習時間が延びる。画面の予測には使わない。
CATBOOST_CV_PARAMS    = {"iterations": 200, "learning_rate": 0.1, "depth": 6}
# 意味: 画面の予測に使う CatBoost の最終モデルの設定。
# 影響: iterations・depth を増やすと細かく学習するが、時間が延び、過学習しやすくなる。
CATBOOST_FINAL_PARAMS = {"iterations": 500, "learning_rate": 0.05, "depth": 8}


# ──────────────────────────────────────────────────────
# 共通処理
# ──────────────────────────────────────────────────────
# 職種は OneHot に、数値の列は標準化する前処理を作る。
def makeOhePreprocessor(numFeatures: list[str]) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        # (名前, 前処理, 対象の列) の組。handle_unknown="ignore" で、学習時になかった職種が来てもエラーにせず全部 0 にする。sparse_output=False で普通の配列を返す。
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["occupation"]),
        ("num", StandardScaler(), numFeatures),
    ])


# 精度の数値を決めた桁数に丸め、辞書にまとめる。
def buildMeta(r2: float, cvScores: np.ndarray, mae: float, features: list[str]) -> ModelMeta:
    return {
        "r2_train":   round(r2, METRIC_DIGITS),
        "r2_cv_mean": round(cvScores.mean(), METRIC_DIGITS),
        "r2_cv_std":  round(cvScores.std(), METRIC_DIGITS),
        "mae_train":  round(mae, MAE_DIGITS),
        "features":   features,
    }


# 1モデル分の精度と所要時間をログに出す。
def logScore(label: str, r2: float, cvScores: np.ndarray, mae: float, startedAt: float) -> None:
    logger.info("  %s R²=%.4f  CV=%.4f±%.4f  MAE=%.1f万円  (%.1fs)",
                # {label:<32} は左寄せで32文字分の幅をとる書式。幅を変数にするときは {LABEL_WIDTH} のように入れ子にする。
                f"{label:<{LABEL_WIDTH}}", r2, cvScores.mean(), cvScores.std(), mae,
                time.time() - startedAt)


# 訓練データでの予測を1回だけ計算し、R² と MAE を求めて、ログと精度情報を作る。
def _evaluate(model: Regressor, X: pd.DataFrame, y: Target, label: str,
              cvScores: np.ndarray, features: list[str], startedAt: float) -> ModelMeta:
    """訓練済みモデルの訓練データでの精度をログに出し、meta を返す"""
    pred = model.predict(X)
    # 右辺の2つの値を、左辺の2つの変数にそれぞれ代入する。
    r2, mae = r2_score(y, pred), mean_absolute_error(y, pred)
    logScore(label, r2, cvScores, mae, startedAt)
    return buildMeta(r2, cvScores, mae, features)


# 交差検証で精度を測ってから、全データで学習し直す。
def _cvAndFit(pipe: Pipeline, X: pd.DataFrame, y: Target, label: str,
              features: list[str]) -> tuple[Pipeline, ModelMeta]:
    # 現在の時刻（秒）。後で差を取って所要時間を出す。
    startedAt = time.time()
    # 分割ごとに学習と検証を繰り返し、各回の R² を配列で返す（分割数の個数だけ値が並ぶ）。
    cvScores = cross_val_score(pipe, X, y, cv=CV, scoring="r2")
    pipe.fit(X, y)
    return pipe, _evaluate(pipe, X, y, label, cvScores, features, startedAt)


# 特徴量を追加したデータで、OneHot の前処理つきパイプラインを学習する共通の処理。
def _feModel(model: Regressor, X: pd.DataFrame, y: Target, label: str) -> tuple[Pipeline, ModelMeta]:
    """特徴量エンジニアリング込みの OHE パイプラインで訓練する"""
    # (名前, 部品) の組を順に並べる。"pre" で前処理し、その結果を "model" に渡す。
    pipe = Pipeline([("pre", makeOhePreprocessor(FE_NUM_FEATURES)), ("model", model)])
    return _cvAndFit(pipe, addFeatures(X), y, label, FE_FEATURES)


# 追加の特徴量なし（職種・年齢・経験年数だけ）で学習する共通の処理。
def _baseModel(model: Regressor, X: pd.DataFrame, y: Target, label: str) -> tuple[Pipeline, ModelMeta]:
    pipe = Pipeline([("pre", makeOhePreprocessor(BASE_NUM_FEATURES)), ("model", model)])
    return _cvAndFit(pipe, X, y, label, BASE_FEATURES)


# ──────────────────────────────────────────────────────
# sklearn モデル
# ──────────────────────────────────────────────────────
# 各 train〜 関数は、(学習済みのモデル, 精度情報) の組を返す。
def trainRidge(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    # 意味: alpha = 予測を極端にしない抑えの強さ（正則化）。
    # 影響: 大きくすると予測がなだらかになり、職種ごとの年収の差が小さく出る。
    return _baseModel(Ridge(alpha=10.0), X, y, "Ridge Regression")


def trainRandomForest(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    model = RandomForestRegressor(
        # 意味: n_estimators = 木の数、max_depth = 木の深さ、min_samples_leaf = 1つの枝に必要な最少のデータ件数。
        # 影響: 木を深く・最少件数を小さくすると訓練データに細かく合わせ、新しい条件で外れやすくなる。木を増やすと安定するが遅くなる。
        n_estimators=200, max_depth=12,
        min_samples_leaf=3, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=ALL_CORES,
    )
    return _baseModel(model, X, y, "Random Forest")


def trainCustomRidge(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    # 意味: alpha = 予測を極端にしない抑えの強さ（正則化）。
    # 影響: 大きくすると予測がなだらかになり、年齢カーブの曲がり方が弱く出る。
    return _feModel(Ridge(alpha=1.0), X, y, "Custom Ridge (+FE)")


def trainElasticnet(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    """
    L1（Lasso）+ L2（Ridge）の両正則化を組み合わせたモデル。
    不要な特徴量を自動で0にする効果（スパース性）があり解釈性が高い。
    特徴量エンジニアリング込みで使用する。
    """
    model = ElasticNet(
        # 影響: alpha を大きくすると、効き目の小さい項目の重みが 0 になりやすく、予測が単純になる。
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
        # 意味: n_estimators = 木の数、learning_rate = 1本ごとの学習の歩幅、max_depth = 木の深さ、subsample = 木ごとに使うデータの割合。
        # 影響: 歩幅を下げて木を増やすと精度が上がりやすいが遅くなる。木を深くすると過学習しやすくなる。
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
# LightGBM はパイプラインを使わず、ラッパークラスの中で職種を番号に変換する。
def trainLightgbm(X: pd.DataFrame, y: Target) -> tuple[LGBMWrapper, ModelMeta]:
    startedAt = time.time()
    logger.info("  %s CV中...", f"{'LightGBM':<{LABEL_WIDTH}}")
    # 既定のパラメータでラッパーを作る。
    wrapper = LGBMWrapper()
    cvScores = cross_val_score(wrapper, X, y, cv=CV, scoring="r2")
    wrapper.fit(X, y)
    return wrapper, _evaluate(wrapper, X, y, "LightGBM", cvScores, BASE_FEATURES, startedAt)


def trainCatboost(X: pd.DataFrame, y: Target) -> tuple[CatBoostWrapper, ModelMeta]:
    """CV は軽量版で高速化し、最終モデルのみ高精度パラメータで訓練"""
    startedAt = time.time()
    logger.info("  %s CV中（%diter）...", f"{'CatBoost':<{LABEL_WIDTH}}", CATBOOST_CV_PARAMS["iterations"])
    # **辞書 で、辞書のキーと値をキーワード引数として渡す（iterations=200, ... と書いたのと同じ）。
    cvScores = cross_val_score(CatBoostWrapper(**CATBOOST_CV_PARAMS), X, y, cv=CV, scoring="r2")
    logger.info("  %s CV完了(%.0fs) → 最終訓練(%diter)...", f"{'CatBoost':<{LABEL_WIDTH}}",
                time.time() - startedAt, CATBOOST_FINAL_PARAMS["iterations"])

    final = CatBoostWrapper(**CATBOOST_FINAL_PARAMS)
    final.fit(X, y)
    return final, _evaluate(final, X, y, "CatBoost", cvScores, BASE_FEATURES, startedAt)


# XGBoost も関数の中で import する（入っていない環境でもファイルを読み込めるようにするため）。
def trainXgboost(X: pd.DataFrame, y: Target) -> tuple[Pipeline, ModelMeta]:
    import xgboost as xgb
    model = xgb.XGBRegressor(
        # 意味: 木の数・歩幅・深さは GradientBoosting と同じ意味。reg_alpha / reg_lambda = 予測を極端にしない抑えの強さ、colsample_bytree = 木ごとに使う項目の割合。
        # 影響: 抑えを強くすると過学習しにくくなるが、細かな差を捉えにくくなる。
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
