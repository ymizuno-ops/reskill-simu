"""
step3_train.py
==============
data/master/ml_dataset.csv を使い、3つのモデルを訓練して
models/ に保存する。

訓練モデル:
  1. Ridge Regression       - 安定型・平均的シナリオ
  2. Random Forest          - 変動型・職種固有の昇給パターン
  3. Custom (Ridge + 特徴量エンジニアリング) - 精度強化版

出力:
  models/
    models.pkl        # 3モデルのpipeline辞書
    model_meta.json   # R²スコア・CV平均等のメタ情報
"""

from __future__ import annotations
import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics         import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(_HERE, "..", "data", "master")
MODEL_DIR  = os.path.join(_HERE, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CV = KFold(n_splits=5, shuffle=True, random_state=42)


# ──────────────────────────────────────────────────────
# 前処理パーツ
# ──────────────────────────────────────────────────────
def make_preprocessor(num_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["occupation"]),
        ("num", StandardScaler(), num_features),
    ])


def add_custom_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    カスタムモデル用の特徴量エンジニアリング:
      - age_sq        : 年齢^2 / 1000  (昇給の逓減・ピーク後の低下を捉える)
      - age_x_exp     : 年齢 × 経験年数 / 100  (シニア+ベテランの相乗効果)
      - exp_ratio     : 経験年数 / 年齢  (年齢に占める経験割合)
      - prime_age_flag: 35〜54歳フラグ  (給与ピーク帯)
    """
    Xc = X.copy()
    Xc["age_sq"]         = Xc["age"] ** 2 / 1000
    Xc["age_x_exp"]      = Xc["age"] * Xc["experience_years"] / 100
    Xc["exp_ratio"]      = Xc["experience_years"] / Xc["age"].clip(lower=1)
    Xc["prime_age_flag"] = ((Xc["age"] >= 35) & (Xc["age"] <= 54)).astype(float)
    return Xc


# ──────────────────────────────────────────────────────
# モデル訓練
# ──────────────────────────────────────────────────────
def train_ridge(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    num_feats = ["age", "experience_years"]
    pre = make_preprocessor(num_feats)
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=10.0))])

    cv_scores = cross_val_score(pipe, X, y, cv=CV, scoring="r2")
    pipe.fit(X, y)
    r2  = r2_score(y, pipe.predict(X))
    mae = mean_absolute_error(y, pipe.predict(X))

    meta = {
        "r2_train": round(r2, 4),
        "r2_cv_mean": round(cv_scores.mean(), 4),
        "r2_cv_std":  round(cv_scores.std(),  4),
        "mae_train":  round(mae, 2),
        "features": ["occupation", "age", "experience_years"],
    }
    print(f"  Ridge     R²={r2:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  MAE={mae:.1f}万円")
    return pipe, meta


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    num_feats = ["age", "experience_years"]
    pre = make_preprocessor(num_feats)
    pipe = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    cv_scores = cross_val_score(pipe, X, y, cv=CV, scoring="r2")
    pipe.fit(X, y)
    r2  = r2_score(y, pipe.predict(X))
    mae = mean_absolute_error(y, pipe.predict(X))

    meta = {
        "r2_train": round(r2, 4),
        "r2_cv_mean": round(cv_scores.mean(), 4),
        "r2_cv_std":  round(cv_scores.std(),  4),
        "mae_train":  round(mae, 2),
        "features": ["occupation", "age", "experience_years"],
    }
    print(f"  RandomForest R²={r2:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  MAE={mae:.1f}万円")
    return pipe, meta


def train_custom(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    Xc = add_custom_features(X)
    num_feats = ["age", "experience_years", "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"]
    pre = make_preprocessor(num_feats)
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])

    cv_scores = cross_val_score(pipe, Xc, y, cv=CV, scoring="r2")
    pipe.fit(Xc, y)
    r2  = r2_score(y, pipe.predict(Xc))
    mae = mean_absolute_error(y, pipe.predict(Xc))

    meta = {
        "r2_train": round(r2, 4),
        "r2_cv_mean": round(cv_scores.mean(), 4),
        "r2_cv_std":  round(cv_scores.std(),  4),
        "mae_train":  round(mae, 2),
        "features": ["occupation", "age", "experience_years",
                     "age_sq", "age_x_exp", "exp_ratio", "prime_age_flag"],
    }
    print(f"  Custom    R²={r2:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  MAE={mae:.1f}万円")
    return pipe, meta


# ──────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("  Step3: モデル訓練")
    print("=" * 60 + "\n")

    # データ読み込み
    df = pd.read_csv(os.path.join(MASTER_DIR, "ml_dataset.csv"))
    print(f"訓練データ: {len(df):,} サンプル, {df['occupation'].nunique()} 職種\n")

    X = df[["occupation", "age", "experience_years"]]
    y = df["annual_income"]

    print("[モデル精度]")
    ridge_pipe, ridge_meta  = train_ridge(X, y)
    rf_pipe,    rf_meta     = train_random_forest(X, y)
    custom_pipe, custom_meta = train_custom(X, y)

    # 保存
    models = {
        "ridge": {
            "pipeline": ridge_pipe,
            "meta":     ridge_meta,
            "label":    "Ridge Regression（安定型）",
            "desc":     "過学習を抑えた線形回帰。平均的・標準的なキャリアパスの推計に最適。",
        },
        "random_forest": {
            "pipeline": rf_pipe,
            "meta":     rf_meta,
            "label":    "Random Forest（変動型）",
            "desc":     "職種固有の昇給パターンを細かく学習。上振れ・下振れシナリオの確認に。",
        },
        "custom": {
            "pipeline": custom_pipe,
            "meta":     custom_meta,
            "label":    "Custom Ridge（特徴量強化型）",
            "desc":     "年齢²・交互作用項を追加した高精度モデル。",
        },
    }

    pkl_path = os.path.join(MODEL_DIR, "models.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(models, f)

    # メタ情報をJSONでも保存（人間が読めるように）
    meta_out = {k: v["meta"] | {"label": v["label"], "desc": v["desc"]}
                for k, v in models.items()}
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ models.pkl     → {pkl_path}")
    print(f"  ✅ model_meta.json → {meta_path}")

    print("\n" + "=" * 60)
    print("  Step3 完了 → models/")
    print("=" * 60 + "\n")

    return models


if __name__ == "__main__":
    main()
