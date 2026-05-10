from __future__ import annotations
import os
import sys
import json
import pickle
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from step3_train import LGBMWrapper, CatBoostWrapper, StackingEnsemble  # noqa: F401
except ImportError:
    pass

from simulation import simulate, calc_roi
from ui.sidebar import render_sidebar
from ui.charts import plot_main_plotly, plot_all_models_plotly
from ui.guides import render_pre_sim_guides, render_post_sim_guides
from ui.results import render_analysis_results

_HERE = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(_HERE, "data", "master")
MODEL_DIR = os.path.join(_HERE, "models")
AGE_ALL_PATH = os.path.join(_HERE, "data", "processed", "age_wage_all.csv")

_MODEL_KEY_ORDER = [
    "ridge", "elasticnet", "custom", "random_forest", "gradient_boosting",
    "lightgbm", "catboost", "xgboost", "stacking",
]
_MODEL_LABEL_MAP = {
    "ridge": "Ridge", "elasticnet": "ElasticNet", "custom": "Custom Ridge",
    "random_forest": "Random Forest", "gradient_boosting": "GradientBoosting",
    "lightgbm": "LightGBM", "catboost": "CatBoost", "xgboost": "XGBoost",
    "stacking": "Stacking",
}

st.set_page_config(
    page_title="リスキリングによる年収シミュレーター",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.result-section {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1rem 1.3rem 0.8rem;
    margin-bottom: 0.8rem;
}
.result-section h4 { font-size: 0.95rem; font-weight: 600; color: var(--text-color); opacity: 0.85; margin: 0 0 0.5rem; }
.result-section ul { margin: 0; padding-left: 1.2rem; list-style: disc; }
.result-section li { font-size: 0.88rem; color: var(--text-color); margin-bottom: 0.25rem; line-height: 1.5; }
.result-section li span.val { font-weight: 700; color: var(--text-color); }
.result-section li span.pos, .pos { color: #00c04b !important; font-weight: 700; }
.result-section li span.neg, .neg { color: #ff4b4b !important; font-weight: 700; }
.result-section li span.neu, .neu { color: #1c83e1 !important; font-weight: 700; }
.lifetime-highlight { font-size: 2rem; font-weight: 700; margin: 0.3rem 0 0; }
[data-testid="stSidebar"] > div:first-child > div:first-child > div:first-child > [data-testid="stButton"],
[data-testid="stSidebarContent"] > [data-testid="stButton"]:first-child,
[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type {
    position: -webkit-sticky; position: sticky; top: 0; z-index: 9999;
    background-color: var(--secondary-background-color);
    padding-top: 12px; padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color); margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="モデルを読み込み中...")
def load_assets():
    pkl = os.path.join(MODEL_DIR, "models.pkl")
    if not os.path.exists(pkl):
        with st.spinner("初回起動: データ加工 & モデル訓練中（1〜2 分）"):
            from step1_to_processed import main as s1
            from step2_to_master import main as s2
            from step3_train import main as s3
            s1(); s2(); s3()

    with open(pkl, "rb") as f:
        models = pickle.load(f)

    occ_list = pd.read_csv(os.path.join(MASTER_DIR, "occupation_list.csv"))
    age_curve = pd.read_csv(os.path.join(MASTER_DIR, "age_curve.csv"))
    with open(os.path.join(MASTER_DIR, "macro_params.json"), encoding="utf-8") as f:
        macro = json.load(f)

    return models, occ_list, age_curve, macro


@st.dialog("⚠️ ご利用にあたっての注意事項")
def _show_disclaimer() -> None:
    st.markdown(
        """
**本アプリをご利用いただく前に、以下の注意事項をご確認ください。**

**📊 シミュレーションの性質について**
- 本アプリはあくまでシミュレーションとなります。
- 表示される年収はAIモデルによる統計的推計であり、実際の年収を保証するものではありません。
- 予測結果は厚生労働省「賃金構造基本統計調査」等の統計データに基づいており、将来の経済状況・労働市場の変化により大きく異なる場合があります。

**👤 個人差について**
- 実際の年収は、個人のスキル・学歴・企業規模・地域・交渉力など、本アプリが考慮していない多数の要因によって左右されます。
- 同一職種であっても、企業や雇用形態によって年収は大きく異なります。

**💼 意思決定における注意**
- 本アプリの結果のみをもって転職・キャリア変更等の重要な意思決定を行わないでください。
- キャリアに関する重要な判断は、キャリアアドバイザーや専門家へのご相談を推奨します。

**🔒 免責事項**
- 本アプリの利用により生じた損害・不利益について、開発者は一切の責任を負いません。
- 本アプリが提供する情報は参考目的に限定されるものであり、開発者はその正確性・完全性・最新性を保証しません。
- 本アプリは予告なく内容の変更・サービスの停止を行う場合があり、それに伴う損害についても開発者は責任を負いません。
"""
    )
    if st.button("上記に同意して始める", type="primary", use_container_width=True):
        st.session_state["disclaimer_accepted"] = True
        st.rerun()


def _run_all_model_simulations(models, p, age_curve):
    model_keys = [k for k in _MODEL_KEY_ORDER if k in models]
    sq_all, cc_all, roi_all = [], [], []
    for key in model_keys:
        s, c = simulate(
            models, key,
            p["current_occ"], p["target_occ"],
            p["current_age"], p["current_exp"], p["current_income"],
            p["skill_transfer"], p["nominal_raise"], age_curve,
            age_all_path=AGE_ALL_PATH,
            raise_suppression=p.get("raise_suppression", 0.0),
            career_risk=p.get("career_risk", 0.0),
        )
        sq_all.append(s)
        cc_all.append(c)
        roi_all.append(calc_roi(s, c, p["learning_cost"]))
    return model_keys, sq_all, cc_all, roi_all


def main() -> None:
    st.title("📊 リスキリングによる年収シミュレーター")

    if not st.session_state.get("disclaimer_accepted", False):
        _show_disclaimer()
        st.stop()

    try:
        models, occ_list, age_curve, macro = load_assets()
    except Exception as e:
        st.error(f"起動エラー: {e}")
        st.stop()

    (
        current_occ, target_occ, current_age, current_exp, current_income,
        skill_transfer, learning_cost, model_key, model_label,
        nominal_raise, gdp_growth, future_cpi, raise_suppression, career_risk,
        submitted,
    ) = render_sidebar(occ_list, models, macro)

    if submitted:
        st.session_state["sim_done"] = True
        st.session_state["sim_params"] = dict(
            current_occ=current_occ, target_occ=target_occ,
            current_age=current_age, current_exp=current_exp, current_income=current_income,
            skill_transfer=skill_transfer, learning_cost=learning_cost,
            model_key=model_key, model_label=model_label,
            nominal_raise=nominal_raise, raise_suppression=raise_suppression,
            career_risk=career_risk, gdp_growth=gdp_growth, future_cpi=future_cpi,
        )

    if not st.session_state.get("sim_done", False):
        st.info("👈 サイドバーで条件を設定し、「シミュレーション実行」ボタンを押してください。")
        render_pre_sim_guides(MODEL_DIR)
        return

    p = st.session_state["sim_params"]
    status_quo, career_change = simulate(
        models, p["model_key"],
        p["current_occ"], p["target_occ"],
        p["current_age"], p["current_exp"], p["current_income"],
        p["skill_transfer"], p["nominal_raise"], age_curve,
        age_all_path=AGE_ALL_PATH,
        raise_suppression=p.get("raise_suppression", 0.0),
        career_risk=p.get("career_risk", 0.0),
    )

    model_keys, sq_all, cc_all, roi_all = _run_all_model_simulations(models, p, age_curve)

    st.markdown("### 📋 分析結果")
    render_analysis_results(
        status_quo, career_change,
        p["current_age"], p["current_occ"], p["target_occ"],
        p["current_income"], p["skill_transfer"], p["learning_cost"],
    )

    st.markdown("### 📈 年収推移グラフ")
    st.plotly_chart(
        plot_main_plotly(status_quo, career_change, p["current_age"], p["current_occ"], p["target_occ"], p["learning_cost"]),
        use_container_width=True, theme="streamlit",
    )

    st.markdown("### 📉 全モデル比較グラフ")
    st.plotly_chart(
        plot_all_models_plotly(sq_all, cc_all, p["current_age"], [r[0] for r in roi_all], [_MODEL_LABEL_MAP[k] for k in model_keys]),
        use_container_width=True, theme="streamlit",
    )

    st.markdown("### 📊 年次詳細（選択モデル・5年刻み）")
    st.dataframe(
        pd.DataFrame([
            {
                "年齢": f"{p['current_age'] + i}歳",
                "現状維持（万円）": f"{status_quo[i]:,.0f}",
                "転職後（万円）": f"{career_change[i]:,.0f}",
                "年間差（万円）": f"{career_change[i] - status_quo[i]:+,.0f}",
            }
            for i in range(0, 50, 5)
        ]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    render_post_sim_guides(
        models, p["model_key"], p["target_occ"],
        p["current_age"], p["current_exp"], p["current_income"],
        AGE_ALL_PATH, MODEL_DIR,
    )

    st.markdown("---")
    rs, cr = p.get("raise_suppression", 0.0), p.get("career_risk", 0.0)
    st.caption(
        f"📌 使用モデル: {p['model_label']} ／ "
        f"GDP: {p.get('gdp_growth', 0.0):+.2f}% ／ CPI: {p.get('future_cpi', 105)} ／ "
        f"昇給抑制: {rs * 100:.0f}% ／ キャリアリスク: {cr * 100:.0f}% ／ "
        "本シミュレーションは厚生労働省「賃金構造基本統計調査」・GDP・CPI をもとにした統計的推計です。"
        "個人の実際の収入を保証するものではありません。"
    )


if __name__ == "__main__":
    main()
