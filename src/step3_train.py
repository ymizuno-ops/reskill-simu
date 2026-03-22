"""
step3_train.py
==============
pickleで保存できるようにWrapper classはすべてモジュールレベルで定義する。
"""

from __future__ import annotations
import os, json, pickle, time, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics         import r2_score, mean_absolute_error
from sklearn.base            import BaseEstimator, RegressorMixin

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


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    Xc["age_sq"]         = Xc["age"] ** 2 / 1000
    Xc["age_x_exp"]      = Xc["age"] * Xc["experience_years"] / 100
    Xc["exp_ratio"]      = Xc["experience_years"] / Xc["age"].clip(lower=1)
    Xc["prime_age_flag"] = ((Xc["age"] >= 35) & (Xc["age"] <= 54)).astype(float)
    return Xc


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
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            min_data_in_leaf=self.min_data_in_leaf,
            random_state=self.random_state,
            verbose=0,
            thread_count=-1,
        )
        self.model_.fit(X, y, cat_features=["occupation"])
        return self

    def predict(self, X):
        return self.model_.predict(X)


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
# メイン
# ──────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("\n" + "=" * 65)
    print("  Step3: モデル訓練（sklearn + LightGBM / CatBoost / XGBoost）")
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
