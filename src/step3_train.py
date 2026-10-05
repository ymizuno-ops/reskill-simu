"""
step3_train.py
==============
全モデルを訓練し、models/models.pkl と models/model_meta.json に保存する。

各モデルの訓練処理は step3_models.py、
Wrapper class（LGBMWrapper・CatBoostWrapper・StackingEnsemble）と addFeatures は
model_wrappers.py に置いている。ここから import するので、step3_train.LGBMWrapper の名前でも引ける。
"""

from __future__ import annotations
import os, json, pickle, time, warnings
from collections.abc import Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics         import r2_score, mean_absolute_error

from log_config import getLogger
from model_types import ModelDict, ModelEntry, ModelMeta, Regressor, Target
from model_wrappers import LGBMWrapper, CatBoostWrapper, StackingEnsemble, RANDOM_STATE  # noqa: F401
from step3_models import (
    CV_FOLDS, BASE_FEATURES, LABEL_WIDTH, buildMeta, logScore,
    trainRidge, trainRandomForest, trainCustomRidge, trainElasticnet, trainGradientBoosting,
    trainLightgbm, trainCatboost, trainXgboost,
)

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

_HERE      = os.path.dirname(os.path.abspath(__file__))
# 意味: 学習データ（ml_dataset.csv）を読み込む場所。
# 注意: step2_to_master.py の OUT_DIR と同じ場所にする。
MASTER_DIR = os.path.join(_HERE, "..", "data", "master")
# 意味: 学習済みモデル（models.pkl）と精度情報（model_meta.json）の保存先。
# 注意: main.py の MODEL_DIR と同じ場所にする（画面がここを読む）。
MODEL_DIR  = os.path.join(_HERE, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 意味: インストールされていれば使う追加ライブラリ。
# 影響: インストールされていないライブラリのモデルは学習を飛ばし、画面のモデル選択肢からも消える。
# 注意: 名前は BOOSTING_MODELS の左端のキーと一致させる。
OPTIONAL_LIBS = ["lightgbm", "catboost", "xgboost"]
# 意味: Stacking の精度評価（評価の中でさらに分割するネストCV）の内側の分割数。
# 影響: 増やすと評価の時間が大きく延びる（今の設定でも5〜15分かかる）。
STACKING_INNER_SPLITS = 4     # ネストCVの内側の fold 数
# 意味: 各モデルの予測を混ぜる Ridge の抑えの強さ（正則化）。
# 影響: 大きくすると特定のモデルに偏らず、均等寄りに混ぜる。
STACKING_META_ALPHA   = 1.0
# 意味: 実行ログの精度ランキングに出す棒（█）の最大の長さ。見た目だけで結果は変わらない。
RANKING_BAR_WIDTH     = 20    # 精度ランキングの棒の長さ（R²=1.0 のとき）
SECTION_RULE_WIDTH    = 65

Trainer = Callable[[pd.DataFrame, Target], tuple[Regressor, ModelMeta]]

# 意味: 常に学習するモデルの一覧。表示名と説明は model_meta.json に保存され、画面の精度カードに出る。
# 影響: 表示名・説明を書き換えて step3 を実行し直すと、画面の表示が変わる。
# 注意: 左端のキー（"ridge" など）は画面・シミュレーションと共通の名前なので変えない。
# (キー, 訓練関数, 表示名, 説明, FE の要否)
SKLEARN_MODELS: list[tuple[str, Trainer, str, str, bool]] = [
    ("ridge", trainRidge, "Ridge Regression（安定型）",
     "過学習を抑えた線形回帰。標準的なキャリアパスの推計に最適。", False),
    ("random_forest", trainRandomForest, "Random Forest（変動型）",
     "職種固有の昇給パターンを細かく学習。上振れ・下振れ確認に。", False),
    ("custom", trainCustomRidge, "Custom Ridge（特徴量強化型）",
     "年齢²・交互作用項を追加した高精度線形モデル。", True),
    ("elasticnet", trainElasticnet, "ElasticNet（L1+L2正則化）",
     "RidgeとLassoの融合。不要な特徴量を自動で除外し解釈性が高い。", True),
    ("gradient_boosting", trainGradientBoosting, "Gradient Boosting（sklearn標準）",
     "追加インストール不要の勾配ブースティング。安定性と精度のバランスが良い。", True),
]
# 意味: 追加ライブラリがあるときだけ学習するモデルの一覧。表示名・説明の扱いは上の一覧と同じ。
# (キー = ライブラリ名, 訓練関数, 表示名, 説明, FE の要否, ログ用の名前)
BOOSTING_MODELS: list[tuple[str, Trainer, str, str, bool, str]] = [
    ("lightgbm", trainLightgbm, "LightGBM（高速ブースティング）",
     "カテゴリ変数ネイティブ対応。高速かつ高精度。", False, "LightGBM"),
    ("catboost", trainCatboost, "CatBoost（カテゴリ変数特化）",
     "職種名をそのまま入力可。チューニング不要で高精度。", False, "CatBoost"),
    ("xgboost", trainXgboost, "XGBoost（勾配ブースティング標準）",
     "業界標準モデル。特徴量エンジニアリング込みで高精度。", True, "XGBoost"),
]


def _checkLibs() -> dict[str, bool]:
    """追加ライブラリごとに import できるかを返す（ない場合はそのモデルを飛ばすため例外にしない）"""
    isAvailable: dict[str, bool] = {}
    for lib in OPTIONAL_LIBS:
        try:
            __import__(lib)
            isAvailable[lib] = True
        except ImportError:
            isAvailable[lib] = False
    return isAvailable


def _entry(model: Regressor, meta: ModelMeta, label: str, desc: str, usesFe: bool) -> ModelEntry:
    # キー名は models.pkl / model_meta.json の形式なので変えない
    return {"pipeline": model, "meta": meta, "label": label, "desc": desc, "uses_fe": usesFe}


# ──────────────────────────────────────────────────────
# Stacking Ensemble 訓練
# ──────────────────────────────────────────────────────
def trainStacking(X: pd.DataFrame, y: Target, baseModels: ModelDict) -> tuple[StackingEnsemble, ModelMeta]:
    """
    全ベースモデルを使った Stacking Ensemble を訓練する。

    Parameters
    ----------
    baseModels : 訓練済みモデル辞書（"stacking"自身は含まない）
    """
    startedAt = time.time()
    logger.info("  %s OOF訓練中...", f"{'Stacking Ensemble':<{LABEL_WIDTH}}")

    # CV: StackingEnsemble 自体を5-fold評価
    # （内部でさらにOOFを使うためネストCVになる → 計算コスト大のため簡易評価）
    cvScores = []
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    yArr = np.asarray(y)
    for foldNo, (trainIdx, valIdx) in enumerate(kf.split(X), 1):
        logger.info("    CV fold %d/%d...", foldNo, CV_FOLDS)
        foldModel = StackingEnsemble(base_models=baseModels, n_splits=STACKING_INNER_SPLITS,
                                     meta_alpha=STACKING_META_ALPHA)
        foldModel.fit(X.iloc[trainIdx].reset_index(drop=True), yArr[trainIdx])
        score = r2_score(yArr[valIdx], foldModel.predict(X.iloc[valIdx].reset_index(drop=True)))
        cvScores.append(score)
        logger.info("    CV fold %d/%d R²=%.4f", foldNo, CV_FOLDS, score)

    # 全データで最終訓練
    logger.info("    最終訓練中（全データ）...")
    stacking = StackingEnsemble(base_models=baseModels, n_splits=CV_FOLDS, meta_alpha=STACKING_META_ALPHA)
    stacking.fit(X, yArr)
    logger.info("    最終訓練 完了")

    pred = stacking.predict(X)
    r2, mae = r2_score(yArr, pred), mean_absolute_error(yArr, pred)
    cvArr = np.array(cvScores)
    logScore("Stacking Ensemble", r2, cvArr, mae, startedAt)

    meta = buildMeta(r2, cvArr, mae, BASE_FEATURES)
    meta["base_models"] = list(baseModels.keys())
    return stacking, meta


# ──────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────
def _trainAll(X: pd.DataFrame, y: Target) -> ModelDict:
    isAvailable = _checkLibs()
    mark = {True: "✅", False: "❌"}
    logger.info("利用可能ライブラリ: LightGBM=%s / CatBoost=%s / XGBoost=%s\n",
                mark[isAvailable["lightgbm"]], mark[isAvailable["catboost"]], mark[isAvailable["xgboost"]])

    # ── sklearn モデル（常に訓練）──
    logger.info("[sklearn モデル]")
    models: ModelDict = {}
    for key, train, label, desc, usesFe in SKLEARN_MODELS:
        models[key] = _entry(*train(X, y), label, desc, usesFe)

    # ── 勾配ブースティング系（ライブラリがあれば）──
    if any(isAvailable.values()):
        logger.info("\n[勾配ブースティング系モデル]")
    for key, train, label, desc, usesFe, displayName in BOOSTING_MODELS:
        if not isAvailable[key]:
            logger.warning("  ⚠ %s スキップ（pip install %s）", displayName, key)
            continue
        models[key] = _entry(*train(X, y), label, desc, usesFe)

    # ── Stacking Ensemble（全ベースモデルが揃ってから訓練）──
    logger.info("\n[Stacking Ensemble]")
    logger.info("  ※ ネストCVのため時間がかかります（5〜15分）")
    try:
        stackingModel, stackingMeta = trainStacking(X, y, models)
        models["stacking"] = _entry(
            stackingModel, stackingMeta, "Stacking Ensemble（全モデル統合）",
            f"全{len(models)}ベースモデルのOOF予測をRidgeで統合。最高精度を目指す。",
            False,   # StackingEnsemble内部で処理するため不要
        )
    except Exception:
        logger.exception("  ⚠ Stacking スキップ（他のモデルは保存を続ける）")
    return models


def _save(models: ModelDict) -> None:
    pklPath = os.path.join(MODEL_DIR, "models.pkl")
    with open(pklPath, "wb") as f:
        pickle.dump(models, f)

    metaOut = {
        k: v["meta"] | {"label": v["label"], "desc": v["desc"]}
        for k, v in models.items()
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metaOut, f, ensure_ascii=False, indent=2)

    logger.info("\n  ✅ %d モデルを保存 → %s", len(models), pklPath)


def _logRanking(models: ModelDict) -> None:
    logger.info("\n[精度ランキング（CV R²降順）]")
    ranked = sorted(models.values(), key=lambda v: v["meta"]["r2_cv_mean"], reverse=True)
    for rank, entry in enumerate(ranked, 1):
        cvMean = entry["meta"]["r2_cv_mean"]
        bar = "█" * int(cvMean * RANKING_BAR_WIDTH)
        logger.info("  %d. %s CV R²=%.4f %s", rank, f"{entry['label']:<30}", cvMean, bar)


def main() -> ModelDict:
    np.random.seed(RANDOM_STATE)

    rule = "=" * SECTION_RULE_WIDTH
    logger.info("\n%s\n  Step3: モデル訓練（sklearn 5モデル + LightGBM / CatBoost / XGBoost）\n%s\n", rule, rule)

    df = pd.read_csv(os.path.join(MASTER_DIR, "ml_dataset.csv"))
    logger.info("訓練データ: %s サンプル, %d 職種\n", f"{len(df):,}", df["occupation"].nunique())

    models = _trainAll(df[BASE_FEATURES], df["annual_income"])
    _save(models)
    _logRanking(models)

    logger.info("\n%s\n  Step3 完了 → models/\n%s\n", rule, rule)
    return models


if __name__ == "__main__":
    main()
