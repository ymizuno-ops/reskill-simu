"""
step3_train.py
==============
Wrapper class（LGBMWrapper・CatBoostWrapper・StackingEnsemble）と add_features は
model_wrappers.py に置いている。ここから import するので、step3_train.LGBMWrapper の名前でも引ける。
"""

from __future__ import annotations
import os, json, pickle, time, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model    import Ridge, ElasticNet
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics         import r2_score, mean_absolute_error

from model_wrappers import add_features, LGBMWrapper, CatBoostWrapper, StackingEnsemble

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(_HERE, "..", "data", "master")
MODEL_DIR  = os.path.join(_HERE, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CV = KFold(n_splits=5, shuffle=True, random_state=42)


# ──────────────────────────────────────────────────────
# ライブラリ有無チェック
# ──────────────────────────────────────────────────────
def _check_libs() -> dict[str, bool]:
    available = {}
    for lib in ["lightgbm", "catboost", "xgboost"]:
        try:
            __import__(lib)
            available[lib] = True
        except ImportError:
            available[lib] = False
    return available


# ──────────────────────────────────────────────────────
# 共通前処理
# ──────────────────────────────────────────────────────
def make_ohe_preprocessor(num_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["occupation"]),
        ("num", StandardScaler(), num_features),
    ])


def _cv_and_fit(pipe, X, y, label: str) -> tuple:
    t0  = time.time()
    cv  = cross_val_score(pipe, X, y, cv=CV, scoring="r2")
    pipe.fit(X, y)
    r2  = r2_score(y, pipe.predict(X))
    mae = mean_absolute_error(y, pipe.predict(X))
    elapsed = time.time() - t0
    print(f"  {label:<32} R²={r2:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}"
          f"  MAE={mae:.1f}万円  ({elapsed:.1f}s)")
    return pipe, {
        "r2_train":   round(r2, 4),
        "r2_cv_mean": round(cv.mean(), 4),
        "r2_cv_std":  round(cv.std(), 4),
        "mae_train":  round(mae, 2),
    }


# ──────────────────────────────────────────────────────
# sklearn モデル
# ──────────────────────────────────────────────────────
def train_ridge(X, y):
    pre  = make_ohe_preprocessor(["age", "experience_years"])
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=10.0))])
    pipe, meta = _cv_and_fit(pipe, X, y, "Ridge Regression")
    meta["features"] = ["occupation", "age", "experience_years"]
    return pipe, meta


def train_random_forest(X, y):
    pre  = make_ohe_preprocessor(["age", "experience_years"])
    pipe = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=3, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )),
    ])
    pipe, meta = _cv_and_fit(pipe, X, y, "Random Forest")
    meta["features"] = ["occupation", "age", "experience_years"]
    return pipe, meta


def train_custom_ridge(X, y):
    Xc  = add_features(X)
    num = ["age", "experience_years", "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    pre = make_ohe_preprocessor(num)
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    pipe, meta = _cv_and_fit(pipe, Xc, y, "Custom Ridge (+FE)")
    meta["features"] = ["occupation", "age", "experience_years",
                        "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    return pipe, meta


# ──────────────────────────────────────────────────────
# ElasticNet
# ──────────────────────────────────────────────────────
def train_elasticnet(X, y):
    """
    L1（Lasso）+ L2（Ridge）の両正則化を組み合わせたモデル。
    不要な特徴量を自動で0にする効果（スパース性）があり解釈性が高い。
    特徴量エンジニアリング込みで使用する。
    """
    Xc  = add_features(X)
    num = ["age", "experience_years", "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    pre = make_ohe_preprocessor(num)
    pipe = Pipeline([
        ("pre", pre),
        ("model", ElasticNet(
            alpha=0.001,     # 正則化強度（チューニング済み）
            l1_ratio=0.7,    # L1:L2 = 70:30（Lasso寄り・スパース性重視）
            max_iter=5000,
            random_state=42,
        )),
    ])
    pipe, meta = _cv_and_fit(pipe, Xc, y, "ElasticNet (+FE)")
    meta["features"] = ["occupation", "age", "experience_years",
                        "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    return pipe, meta


# ──────────────────────────────────────────────────────
# Gradient Boosting (sklearn)
# ──────────────────────────────────────────────────────
def train_gradient_boosting(X, y):
    """
    sklearn 標準の勾配ブースティング。追加インストール不要。
    XGBoost・LightGBMより低速だが安定性が高く、過学習に強い。
    特徴量エンジニアリング込みで使用する。
    """
    Xc  = add_features(X)
    num = ["age", "experience_years", "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    pre = make_ohe_preprocessor(num)
    pipe = Pipeline([
        ("pre", pre),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )),
    ])
    pipe, meta = _cv_and_fit(pipe, Xc, y, "GradientBoosting (+FE)")
    meta["features"] = ["occupation", "age", "experience_years",
                        "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    return pipe, meta


# ──────────────────────────────────────────────────────
# LightGBM
# ──────────────────────────────────────────────────────
def train_lightgbm(X, y):
    t0 = time.time()
    print(f"  {'LightGBM':<32} CV中...", end="", flush=True)
    wrapper = LGBMWrapper()
    cv = cross_val_score(wrapper, X, y, cv=CV, scoring="r2")
    wrapper.fit(X, y)
    r2  = r2_score(y, wrapper.predict(X))
    mae = mean_absolute_error(y, wrapper.predict(X))
    print(f"\r  {'LightGBM':<32} R²={r2:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}"
          f"  MAE={mae:.1f}万円  ({time.time()-t0:.1f}s)")
    meta = {
        "r2_train": round(r2,4), "r2_cv_mean": round(cv.mean(),4),
        "r2_cv_std": round(cv.std(),4), "mae_train": round(mae,2),
        "features": ["occupation", "age", "experience_years"],
    }
    return wrapper, meta


# ──────────────────────────────────────────────────────
# CatBoost
# ──────────────────────────────────────────────────────
def train_catboost(X, y):
    """CV は軽量版（iterations=200）で高速化し、最終モデルのみ500iterで訓練"""
    t0 = time.time()
    print(f"  {'CatBoost':<32} CV中（200iter）...", end="", flush=True)

    # CV: 軽量版パラメータ
    cv_wrapper = CatBoostWrapper(iterations=200, learning_rate=0.1, depth=6)
    cv = cross_val_score(cv_wrapper, X, y, cv=CV, scoring="r2")
    print(f" CV完了({time.time()-t0:.0f}s) → 最終訓練(500iter)...", end="", flush=True)

    # 最終モデル: 高精度パラメータ
    final = CatBoostWrapper(iterations=500, learning_rate=0.05, depth=8)
    final.fit(X, y)
    r2  = r2_score(y, final.predict(X))
    mae = mean_absolute_error(y, final.predict(X))
    print(f"\r  {'CatBoost':<32} R²={r2:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}"
          f"  MAE={mae:.1f}万円  ({time.time()-t0:.1f}s)")
    meta = {
        "r2_train": round(r2,4), "r2_cv_mean": round(cv.mean(),4),
        "r2_cv_std": round(cv.std(),4), "mae_train": round(mae,2),
        "features": ["occupation", "age", "experience_years"],
    }
    return final, meta


# ──────────────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────────────
def train_xgboost(X, y):
    import xgboost as xgb
    Xc  = add_features(X)
    num = ["age", "experience_years", "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    pre = make_ohe_preprocessor(num)
    pipe = Pipeline([
        ("pre", pre),
        ("model", xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ])
    pipe, meta = _cv_and_fit(pipe, Xc, y, "XGBoost (+FE)")
    meta["features"] = ["occupation", "age", "experience_years",
                        "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    return pipe, meta


# ──────────────────────────────────────────────────────
# Stacking Ensemble 訓練
# ──────────────────────────────────────────────────────
def train_stacking(X: pd.DataFrame, y, base_models: dict):
    """
    全ベースモデルを使った Stacking Ensemble を訓練する。

    Parameters
    ----------
    base_models : 訓練済みモデル辞書（"stacking"自身は含まない）
    """
    import time
    t0 = time.time()
    print(f"  {'Stacking Ensemble':<32} OOF訓練中...")

    stacking = StackingEnsemble(
        base_models=base_models,
        n_splits=5,
        meta_alpha=1.0,
    )

    # CV: StackingEnsemble 自体を5-fold評価
    # （内部でさらにOOFを使うためネストCVになる → 計算コスト大のため簡易評価）
    # 簡易CV: 各foldでfitしてOOF R²を計算
    cv_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_arr = np.asarray(y)
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"    CV fold {fold_i}/5...", end="", flush=True)
        s_fold = StackingEnsemble(base_models=base_models, n_splits=4, meta_alpha=1.0)
        s_fold.fit(X.iloc[tr_idx].reset_index(drop=True), y_arr[tr_idx])
        pred = s_fold.predict(X.iloc[val_idx].reset_index(drop=True))
        from sklearn.metrics import r2_score
        score = r2_score(y_arr[val_idx], pred)
        cv_scores.append(score)
        print(f" R²={score:.4f}")

    cv_arr = np.array(cv_scores)

    # 全データで最終訓練
    print(f"    最終訓練中（全データ）...", end="", flush=True)
    stacking.fit(X, y_arr)
    print(" 完了")

    r2  = r2_score(y_arr, stacking.predict(X))
    mae = mean_absolute_error(y_arr, stacking.predict(X))
    elapsed = time.time() - t0

    print(f"  {'Stacking Ensemble':<32} R²={r2:.4f}  CV={cv_arr.mean():.4f}±{cv_arr.std():.4f}"
          f"  MAE={mae:.1f}万円  ({elapsed:.1f}s)")

    meta = {
        "r2_train":   round(r2, 4),
        "r2_cv_mean": round(cv_arr.mean(), 4),
        "r2_cv_std":  round(cv_arr.std(), 4),
        "mae_train":  round(mae, 2),
        "features":   ["occupation", "age", "experience_years"],
        "base_models": list(base_models.keys()),
    }
    return stacking, meta


# ──────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("\n" + "=" * 65)
    print("  Step3: モデル訓練（sklearn 5モデル + LightGBM / CatBoost / XGBoost）")
    print("=" * 65 + "\n")

    df = pd.read_csv(os.path.join(MASTER_DIR, "ml_dataset.csv"))
    print(f"訓練データ: {len(df):,} サンプル, {df['occupation'].nunique()} 職種\n")

    X = df[["occupation", "age", "experience_years"]]
    y = df["annual_income"]

    libs = _check_libs()
    print(f"利用可能ライブラリ: "
          f"LightGBM={'✅' if libs['lightgbm'] else '❌'} / "
          f"CatBoost={'✅' if libs['catboost'] else '❌'} / "
          f"XGBoost={'✅' if libs['xgboost'] else '❌'}\n")

    # ── sklearn モデル（常に訓練）──
    print("[sklearn モデル]")
    ridge_pipe,  ridge_meta  = train_ridge(X, y)
    rf_pipe,     rf_meta     = train_random_forest(X, y)
    custom_pipe, custom_meta = train_custom_ridge(X, y)
    en_pipe,     en_meta     = train_elasticnet(X, y)
    gb_pipe,     gb_meta     = train_gradient_boosting(X, y)

    models = {
        "ridge": {
            "pipeline": ridge_pipe, "meta": ridge_meta,
            "label": "Ridge Regression（安定型）",
            "desc":  "過学習を抑えた線形回帰。標準的なキャリアパスの推計に最適。",
            "uses_fe": False,
        },
        "random_forest": {
            "pipeline": rf_pipe, "meta": rf_meta,
            "label": "Random Forest（変動型）",
            "desc":  "職種固有の昇給パターンを細かく学習。上振れ・下振れ確認に。",
            "uses_fe": False,
        },
        "custom": {
            "pipeline": custom_pipe, "meta": custom_meta,
            "label": "Custom Ridge（特徴量強化型）",
            "desc":  "年齢²・交互作用項を追加した高精度線形モデル。",
            "uses_fe": True,
        },
        "elasticnet": {
            "pipeline": en_pipe, "meta": en_meta,
            "label": "ElasticNet（L1+L2正則化）",
            "desc":  "RidgeとLassoの融合。不要な特徴量を自動で除外し解釈性が高い。",
            "uses_fe": True,
        },
        "gradient_boosting": {
            "pipeline": gb_pipe, "meta": gb_meta,
            "label": "Gradient Boosting（sklearn標準）",
            "desc":  "追加インストール不要の勾配ブースティング。安定性と精度のバランスが良い。",
            "uses_fe": True,
        },
    }

    # ── 勾配ブースティング系（ライブラリがあれば）──
    if any(libs.values()):
        print()
        print("[勾配ブースティング系モデル]")

    if libs["lightgbm"]:
        lgbm_model, lgbm_meta = train_lightgbm(X, y)
        models["lightgbm"] = {
            "pipeline": lgbm_model, "meta": lgbm_meta,
            "label": "LightGBM（高速ブースティング）",
            "desc":  "カテゴリ変数ネイティブ対応。高速かつ高精度。",
            "uses_fe": False,
        }
    else:
        print("  ⚠ LightGBM スキップ（pip install lightgbm）")

    if libs["catboost"]:
        cb_model, cb_meta = train_catboost(X, y)
        models["catboost"] = {
            "pipeline": cb_model, "meta": cb_meta,
            "label": "CatBoost（カテゴリ変数特化）",
            "desc":  "職種名をそのまま入力可。チューニング不要で高精度。",
            "uses_fe": False,
        }
    else:
        print("  ⚠ CatBoost スキップ（pip install catboost）")

    if libs["xgboost"]:
        xgb_model, xgb_meta = train_xgboost(X, y)
        models["xgboost"] = {
            "pipeline": xgb_model, "meta": xgb_meta,
            "label": "XGBoost（勾配ブースティング標準）",
            "desc":  "業界標準モデル。特徴量エンジニアリング込みで高精度。",
            "uses_fe": True,
        }
    else:
        print("  ⚠ XGBoost スキップ（pip install xgboost）")

    # ── Stacking Ensemble（全ベースモデルが揃ってから訓練）──
    print()
    print("[Stacking Ensemble]")
    print("  ※ ネストCVのため時間がかかります（5〜15分）")
    try:
        stacking_model, stacking_meta = train_stacking(X, y, models)
        models["stacking"] = {
            "pipeline": stacking_model,
            "meta":     stacking_meta,
            "label":    "Stacking Ensemble（全モデル統合）",
            "desc":     f"全{len(models)}ベースモデルのOOF予測をRidgeで統合。最高精度を目指す。",
            "uses_fe":  False,   # StackingEnsemble内部で処理するため不要
        }
    except Exception as e:
        print(f"  ⚠ Stacking スキップ: {e}")

    # ── 保存 ──
    pkl_path = os.path.join(MODEL_DIR, "models.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(models, f)

    meta_out = {
        k: v["meta"] | {"label": v["label"], "desc": v["desc"]}
        for k, v in models.items()
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ {len(models)} モデルを保存 → {pkl_path}")

    # ── 精度ランキング ──
    print("\n[精度ランキング（CV R²降順）]")
    ranked = sorted(models.items(), key=lambda x: x[1]["meta"]["r2_cv_mean"], reverse=True)
    for rank, (k, v) in enumerate(ranked, 1):
        m   = v["meta"]
        bar = "█" * int(m["r2_cv_mean"] * 20)
        print(f"  {rank}. {v['label']:<30} CV R²={m['r2_cv_mean']:.4f} {bar}")

    print("\n" + "=" * 65)
    print("  Step3 完了 → models/")
    print("=" * 65 + "\n")

    return models


if __name__ == "__main__":
    main()
