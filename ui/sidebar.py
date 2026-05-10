from __future__ import annotations
from typing import Any
import streamlit as st
from occupation import OCCUPATION_CATEGORIES, build_category_occ_map

_ALL_MODEL_OPTIONS: dict[str, str] = {
    "Ridge（安定型）": "ridge",
    "ElasticNet（L1+L2正則化）": "elasticnet",
    "Custom Ridge（特徴量強化型）": "custom",
    "Random Forest（変動型）": "random_forest",
    "Gradient Boosting（sklearn標準）": "gradient_boosting",
    "LightGBM（高速ブースティング）": "lightgbm",
    "CatBoost（カテゴリ変数特化）": "catboost",
    "XGBoost（勾配ブースティング）": "xgboost",
    "Stacking Ensemble（全モデル統合）": "stacking",
}
_DEFAULT_MODEL_LABEL = "Custom Ridge（特徴量強化型）"
_DEFAULT_CURRENT_OCC = "販売店員"
_DEFAULT_TARGET_OCC = "システムコンサルタント・設計者"


def render_sidebar(
    occ_list: Any,
    models: dict[str, Any],
    macro: dict[str, Any],
) -> tuple:
    occs = sorted(occ_list["occupation"].tolist())
    cat_occ_map = build_category_occ_map(occs)
    all_cats = ["（すべて）"] + sorted(OCCUPATION_CATEGORIES.keys())

    submitted = st.sidebar.button("🚀 シミュレーション実行", type="primary", use_container_width=True)

    st.sidebar.markdown("### 👤 プロフィール設定")
    current_age = st.sidebar.number_input("現在の年齢", 20, 65, 30, step=1)
    current_exp = st.sidebar.number_input("現在の勤続年数", 0, 40, 5, step=1)
    current_income = st.sidebar.number_input("現在の年収（万円）", 100, 3000, 450, step=10)

    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 予測モデル選択")
    available_models = {label: key for label, key in _ALL_MODEL_OPTIONS.items() if key in models}
    default_label = _DEFAULT_MODEL_LABEL if _DEFAULT_MODEL_LABEL in available_models else list(available_models.keys())[-1]
    model_label = st.sidebar.selectbox(
        "使用するAIモデル",
        list(available_models.keys()),
        index=list(available_models.keys()).index(default_label),
    )
    model_key = available_models[model_label]

    st.sidebar.divider()
    st.sidebar.markdown("### 💼 キャリア選択")

    cur_cat = st.sidebar.selectbox("現職の大分類", all_cats, index=0, key="cur_cat")
    cur_occs = sorted(cat_occ_map.get(cur_cat, [])) if cur_cat != "（すべて）" else occs
    if not cur_occs:
        cur_occs = occs
    default_cur = cur_occs.index(_DEFAULT_CURRENT_OCC) if _DEFAULT_CURRENT_OCC in cur_occs else 0
    current_occ = st.sidebar.selectbox("現職名", cur_occs, index=default_cur, key="cur_occ")

    tgt_cat = st.sidebar.selectbox("目標の大分類", all_cats, index=0, key="tgt_cat")
    tgt_occs = sorted(cat_occ_map.get(tgt_cat, [])) if tgt_cat != "（すべて）" else occs
    if not tgt_occs:
        tgt_occs = occs
    default_tgt = tgt_occs.index(_DEFAULT_TARGET_OCC) if _DEFAULT_TARGET_OCC in tgt_occs else 0
    target_occ = st.sidebar.selectbox("目標職種名", tgt_occs, index=default_tgt, key="tgt_occ")

    skill_transfer = st.sidebar.slider("経験引継ぎ率（%）", 0, 100, 20, help="0%＝完全未経験、100%＝即戦力。") / 100

    st.sidebar.divider()
    st.sidebar.markdown("### 💰 投資設定")
    learning_cost = st.sidebar.number_input("自己投資費用（万円）", 0, 500, 50, step=5)
    gdp_growth = st.sidebar.slider(
        "期待GDP成長率（%）", -3.0, 3.0,
        float(round(macro["avg_gdp_growth_10yr"] * 100, 2)), step=0.05,
    )
    future_cpi = st.sidebar.slider("将来のCPI（物価指数）", 80, 150, 105, step=1)
    nominal_raise = max(gdp_growth / 100 + (future_cpi - 100) / 100 * 0.3, 0.0)

    st.sidebar.divider()
    st.sidebar.markdown("### 🎯 リアリティ補正")
    raise_suppression = st.sidebar.slider("転職後の昇給抑制（%）", 0, 50, 0, step=5) / 100
    career_risk = st.sidebar.slider("キャリアリスク係数（%）", 0, 30, 0, step=5) / 100

    return (
        current_occ, target_occ, current_age, current_exp, current_income,
        skill_transfer, learning_cost, model_key, model_label,
        nominal_raise, gdp_growth, future_cpi, raise_suppression, career_risk,
        submitted,
    )
