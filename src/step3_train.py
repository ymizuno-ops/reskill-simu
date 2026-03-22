"""
step3_train.py
==============
pickleで保存できるようにWrapper classはすべてモジュールレベルで定義する。
"""

from __future__ import annotations
import os, json, pickle, time, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model    import Ridge, ElasticNet
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
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
