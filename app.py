# app.py ─ リスキリング年収シミュレーター (全ガイド実装版)

from __future__ import annotations
import os
import sys
import json
import pickle
import warnings
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from step3_train import LGBMWrapper, CatBoostWrapper, StackingEnsemble  # noqa: F401
except ImportError:
    pass

# ── パス ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(_HERE, "data", "master")
MODEL_DIR = os.path.join(_HERE, "models")
AGE_ALL_PATH = os.path.join(_HERE, "data", "processed", "age_wage_all.csv")

# ══════════════════════════════════════════════════════
# ページ設定 & CSS注入
# ══════════════════════════════════════════════════════
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
.result-section h4 {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.85;
    margin: 0 0 0.5rem;
}
.result-section ul {
    margin: 0;
    padding-left: 1.2rem;
    list-style: disc;
}
.result-section li {
    font-size: 0.88rem;
    color: var(--text-color);
    margin-bottom: 0.25rem;
    line-height: 1.5;
}
.result-section li span.val {
    font-weight: 700;
    color: var(--text-color);
}
.result-section li span.pos, .pos { color: #00c04b !important; font-weight: 700; }
.result-section li span.neg, .neg { color: #ff4b4b !important; font-weight: 700; }
.result-section li span.neu, .neu { color: #1c83e1 !important; font-weight: 700; }
.lifetime-highlight {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.3rem 0 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════
# アセット読み込み
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner="モデルを読み込み中...")
def load_assets() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    pkl = os.path.join(MODEL_DIR, "models.pkl")
    if not os.path.exists(pkl):
        with st.spinner("初回起動: データ加工 & モデル訓練中（1〜2 分）"):
            from step1_to_processed import main as s1
            from step2_to_master import main as s2
            from step3_train import main as s3

            s1()
            s2()
            s3()

    with open(pkl, "rb") as f:
        models = pickle.load(f)

    occ_list = pd.read_csv(os.path.join(MASTER_DIR, "occupation_list.csv"))
    age_curve = pd.read_csv(os.path.join(MASTER_DIR, "age_curve.csv"))
    with open(os.path.join(MASTER_DIR, "macro_params.json"), encoding="utf-8") as f:
        macro = json.load(f)

    return models, occ_list, age_curve, macro


# ══════════════════════════════════════════════════════
# 予測・シミュレーション
# ══════════════════════════════════════════════════════
def _add_features(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    Xc["age_sq"] = Xc["age"] ** 2 / 1000
    Xc["age_x_exp"] = Xc["age"] * Xc["experience_years"] / 100
    Xc["exp_ratio"] = Xc["experience_years"] / Xc["age"].clip(lower=1)
    Xc["prime_age_flag"] = ((Xc["age"] >= 35) & (Xc["age"] <= 54)).astype(float)
    return Xc


_FE_MODELS = {"custom", "xgboost", "elasticnet", "gradient_boosting"}


def predict(
    models: Dict[str, Any],
    model_key: str,
    occupation: str,
    age: float,
    experience: float,
) -> float:
    m = models[model_key]
    X = pd.DataFrame(
        [
            {
                "occupation": occupation,
                "age": float(age),
                "experience_years": float(experience),
            }
        ]
    )
    if model_key != "stacking" and model_key in _FE_MODELS:
        X = _add_features(X)
    return float(m["pipeline"].predict(X)[0])


_AGE_MIDS = [18.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 67.0]
_AGE_LABELS = [
    "〜19歳",
    "20〜24歳",
    "25〜29歳",
    "30〜34歳",
    "35〜39歳",
    "40〜44歳",
    "45〜49歳",
    "50〜54歳",
    "55〜59歳",
    "60〜64歳",
    "65〜69歳",
]


def _get_one_step_down_income(
    occ_name: str, current_age: float, age_all_path: str, year: int = 2024
) -> Tuple[float, str]:
    try:
        current_mid = min(_AGE_MIDS, key=lambda m: abs(m - current_age))
        idx = _AGE_MIDS.index(current_mid)
        lower_idx = max(0, idx - 1)
        lower_mid = _AGE_MIDS[lower_idx]
        lower_label = _AGE_LABELS[lower_idx]

        age_all = pd.read_csv(age_all_path)
        latest = age_all[
            (age_all["year"] == year) & (age_all["occupation"] == occ_name)
        ]
        row = latest[latest["age_mid"] == lower_mid]
        if len(row) > 0:
            return float(row["annual_income"].mean()), lower_label

        all_row = age_all[(age_all["year"] == year) & (age_all["age_mid"] == lower_mid)]
        if len(all_row) > 0:
            return float(all_row["annual_income"].mean()), lower_label
    except Exception:
        pass
    return 300.0, "〜"


def simulate(
    models: Dict[str, Any],
    model_key: str,
    current_occ: str,
    target_occ: str,
    current_age: int,
    current_exp: float,
    current_income: float,
    skill_transfer: float,
    nominal_raise: float,
    age_curve: pd.DataFrame,
    years: int = 50,
    age_all_path: str = "",
    raise_suppression: float = 0.0,
    career_risk: float = 0.0,
) -> Tuple[List[float], List[float]]:

    raise_by_age = dict(zip(age_curve["age_mid"], age_curve["raise_rate"]))
    base_pred = predict(models, model_key, current_occ, current_age, current_exp)
    correction = current_income / max(base_pred, 1)

    status_quo_incomes = []
    income = current_income
    for i in range(years):
        age = current_age + i
        if age >= 65:
            income *= 0.97
        else:
            pred = predict(models, model_key, current_occ, age, current_exp + i)
            income = pred * correction * (1 + nominal_raise)
        status_quo_incomes.append(max(income, 0))

    base_income, base_label = _get_one_step_down_income(
        target_occ, current_age, age_all_path
    )
    experienced_income = predict(
        models, model_key, target_occ, current_age, current_exp
    )
    first_income = max(
        base_income + (experienced_income - base_income) * skill_transfer,
        base_income * 0.8,
    )

    current_mid = min(_AGE_MIDS, key=lambda m: abs(m - current_age))
    idx = _AGE_MIDS.index(current_mid)
    lower_mid = _AGE_MIDS[max(0, idx - 1)]
    base_pred_at_lower = predict(models, model_key, target_occ, lower_mid, 0)
    corr2 = first_income / max(base_pred_at_lower, 1)

    career_change_incomes = []
    for i in range(years):
        age = current_age + i
        if age >= 65:
            career_change_incomes.append(max(career_change_incomes[-1] * 0.97, 0))
        else:
            effective_exp = i
            pred = predict(models, model_key, target_occ, age, effective_exp)
            suppression_factor = 1.0 - raise_suppression * max(0, (10 - i) / 10)
            risk_decay = 1.0 - career_risk * max(0, (i - 10) / 40)
            income = (
                pred
                * corr2
                * suppression_factor
                * risk_decay
                * (1 + nominal_raise * i * 0.05)
            )
            career_change_incomes.append(max(income, 0))

    return status_quo_incomes, career_change_incomes


def calc_roi(
    status_quo_incomes: List[float], career_change_incomes: List[float], cost: float
) -> Tuple[Optional[int], float]:
    cumulative, breakeven_month = 0, None
    for i, (s, c) in enumerate(zip(status_quo_incomes, career_change_incomes)):
        annual_diff = c - s
        cumulative += annual_diff
        monthly_diff = annual_diff / 12
        if monthly_diff > 0 and breakeven_month is None:
            months_to_break = (-cumulative + annual_diff + cost) / monthly_diff
            if months_to_break <= 12:
                breakeven_month = i * 12 + int(months_to_break)
    lifetime = sum(c - s for s, c in zip(status_quo_incomes, career_change_incomes))
    return breakeven_month, lifetime


# ══════════════════════════════════════════════════════
# Plotly グラフ描画
# ══════════════════════════════════════════════════════
def plot_main_plotly(
    status_quo_incomes: List[float],
    career_change_incomes: List[float],
    current_age: int,
    current_occ: str,
    target_occ: str,
    cost: float,
) -> go.Figure:
    ages = [current_age + i for i in range(len(status_quo_incomes))]
    cumul = np.cumsum(
        [c - s for s, c in zip(status_quo_incomes, career_change_incomes)]
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=ages,
            y=status_quo_incomes,
            name="現状維持",
            line=dict(color="#4F8EF7", width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=career_change_incomes,
            name="転職後",
            line=dict(color="#FF5B5B", width=3),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=ages,
            y=cumul,
            name="累積収支差額",
            line=dict(color="#43a047", width=2, dash="dash"),
            fill="tozeroy",
            opacity=0.3,
        ),
        secondary_y=True,
    )

    if cost > 0:
        fig.add_hline(
            y=-cost,
            line_dash="dot",
            line_color="#FB8C00",
            annotation_text=f"学習コスト ▲{cost:,}万円",
            secondary_y=True,
        )

    fig.update_layout(
        title=f"年収推移シミュレーション ({current_occ} vs {target_occ})",
        xaxis_title="年齢",
        yaxis_title="年収（万円）",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="累積収支差額（万円）", secondary_y=True)

    return fig


def plot_all_models_plotly(
    sq_all: List[List[float]],
    cc_all: List[List[float]],
    current_age: int,
    breakevens: List[Optional[int]],
    model_labels: List[str],
) -> go.Figure:
    n = len(sq_all)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=model_labels, shared_xaxes=True
    )
    ages = [current_age + i for i in range(len(sq_all[0]))]

    for i, (sq, cc, be) in enumerate(zip(sq_all, cc_all, breakevens)):
        row = (i // ncols) + 1
        col = (i % ncols) + 1

        fig.add_trace(
            go.Scatter(
                x=ages,
                y=sq,
                line=dict(color="#4F8EF7", width=2),
                showlegend=(i == 0),
                name="現状維持",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=ages,
                y=cc,
                line=dict(color="#FF5B5B", width=2),
                showlegend=(i == 0),
                name="転職後",
            ),
            row=row,
            col=col,
        )

        if be:
            be_age = current_age + be / 12
            fig.add_vline(
                x=be_age,
                line_dash="dash",
                line_color="gold",
                row=row,
                col=col,
                annotation_text=f"回収",
                annotation_position="top right",
            )

    fig.update_layout(
        height=300 * nrows, title_text="モデル別 年収推移比較", showlegend=True
    )
    return fig


# ══════════════════════════════════════════════════════
# 職種データ
# ══════════════════════════════════════════════════════
OCCUPATION_CATEGORIES = {
    "管理職": ["管理的職業従事者", "男管理的職業従事者", "女管理的職業従事者"],
    "専門職・技術職（IT・理工系）": [
        "研究者",
        "システムコンサルタント・設計者",
        "ソフトウェア作成者",
        "電気・電子・電気通信技術者（通信ネットワーク技術者を除く）",
        "機械技術者",
        "輸送用機器技術者",
        "金属技術者",
        "化学技術者",
        "建築技術者",
        "土木技術者",
        "測量技術者",
        "他に分類されない技術者",
        "その他の情報処理・通信技術者",
    ],
    "専門職・技術職（医療・福祉）": [
        "医師",
        "歯科医師",
        "獣医師",
        "薬剤師",
        "保健師",
        "助産師",
        "看護師",
        "准看護師",
        "看護助手",
        "男看護助手",
        "女看護助手",
        "理学療法士，作業療法士，言語聴覚士，視能訓練士",
        "臨床検査技師",
        "診療放射線技師",
        "歯科衛生士",
        "歯科技工士",
        "栄養士",
        "介護職員（医療・福祉施設等）",
        "訪問介護従事者",
        "介護支援専門員（ケアマネージャー）",
        "その他の保健医療従事者",
        "その他の保健医療サービス職業従事者",
        "その他の社会福祉専門職業従事者",
    ],
    "専門職・技術職（教育・文化）": [
        "大学教授（高専含む）",
        "大学准教授（高専含む）",
        "大学講師・助教（高専含む）",
        "高等学校教員",
        "小・中学校教員",
        "幼稚園教員，保育教諭",
        "保育士",
        "個人教師",
        "その他の教員",
        "著述家，記者，編集者",
        "美術家，写真家，映像撮影者",
        "音楽家，舞台芸術家",
        "デザイナー",
        "宗教家",
        "他に分類されない専門的職業従事者",
    ],
    "専門職・技術職（法務・経営・金融）": [
        "公認会計士，税理士",
        "法務従事者",
        "その他の経営・金融・保険専門職業従事者",
    ],
    "事務職": [
        "総合事務員",
        "庶務・人事事務員",
        "企画事務員",
        "会計事務従事者",
        "営業・販売事務従事者",
        "生産関連事務従事者",
        "外勤事務従事者",
        "運輸・郵便事務従事者",
        "事務用機器操作員",
        "秘書",
        "受付・案内事務員",
        "電話応接事務員",
        "その他の一般事務従事者",
    ],
    "営業・販売職": [
        "販売店員",
        "販売類似職業従事者",
        "機械器具・通信・システム営業職業従事者（自動車を除く）",
        "自動車営業職業従事者",
        "保険営業職業従事者",
        "金融営業職業従事者",
        "その他の営業職業従事者",
        "その他の商品販売従事者",
    ],
    "サービス職": [
        "飲食物調理従事者",
        "飲食物給仕従事者",
        "娯楽場等接客員",
        "理容・美容師",
        "美容サービス・浴場従事者（美容師を除く）",
        "航空機客室乗務員",
        "身の回り世話従事者",
        "居住施設・ビル等管理人",
        "ビル・建物清掃員",
        "清掃員（ビル・建物を除く），廃棄物処理従事者",
        "その他のサービス職業従事者",
    ],
    "保安職": ["警備員", "その他の保安職業従事者"],
    "農林漁業": ["農林漁業従事者"],
    "生産・製造職（機械・金属）": [
        "機械検査従事者",
        "自動車整備・修理従事者",
        "自動車組立従事者",
        "はん用・生産用・業務用機械器具組立従事者",
        "はん用・生産用・業務用機械器具・電気機械器具整備・修理従事者",
        "電気機械器具組立従事者",
        "金属工作機械作業従事者",
        "金属プレス従事者",
        "金属溶接・溶断従事者",
        "鉄工，製缶従事者",
        "板金従事者",
        "鋳物製造・鍛造従事者",
        "金属彫刻・表面処理従事者",
        "製銑・製鋼・非鉄金属製錬従事者",
        "製品検査従事者（金属製品）",
        "その他の機械組立従事者",
        "その他の機械整備・修理従事者",
    ],
    "生産・製造職（その他）": [
        "食料品・飲料・たばこ製造従事者",
        "化学製品製造従事者",
        "ゴム・プラスチック製品製造従事者",
        "窯業・土石製品製造従事者",
        "木・紙製品製造従事者",
        "紡織・衣服・繊維製品製造従事者",
        "印刷・製本従事者",
        "製品検査従事者（金属製品を除く）",
        "製図その他生産関連・生産類似作業従事者",
        "包装従事者",
        "その他の製品製造・加工処理従事者（金属製品）",
        "その他の製品製造・加工処理従事者（金属製品を除く）",
    ],
    "建設・土木職": [
        "大工",
        "建設躯体工事従事者",
        "土木従事者，鉄道線路工事従事者",
        "建設・さく井機械運転従事者",
        "ダム・トンネル掘削従事者，採掘従事者",
        "配管従事者",
        "電気工事従事者",
        "画工，塗装・看板制作従事者",
        "その他の建設従事者",
    ],
    "輸送・機械運転職": [
        "鉄道運転従事者",
        "バス運転者",
        "タクシー運転者",
        "乗用自動車運転者（タクシー運転者を除く）",
        "営業用大型貨物自動車運転者",
        "営業用貨物自動車運転者（大型車を除く）",
        "自家用貨物自動車運転者",
        "航空機操縦士",
        "車掌",
        "クレーン・ウインチ運転従事者",
        "その他の定置・建設機械運転従事者",
        "その他の自動車運転従事者",
        "他に分類されない輸送従事者",
    ],
    "運搬・清掃・その他": [
        "船内・沿岸荷役従事者",
        "その他の運搬従事者",
        "クリーニング職，洗張職",
        "発電員，変電員",
        "他に分類されない運搬・清掃・包装等従事者",
    ],
}
_OCC_TO_CATEGORY = {
    occ: cat for cat, occs in OCCUPATION_CATEGORIES.items() for occ in occs
}


def get_category(occ: str) -> str:
    return _OCC_TO_CATEGORY.get(occ, "その他")


# ══════════════════════════════════════════════════════
# UI描画ロジック
# ══════════════════════════════════════════════════════
def render_sidebar(
    occ_list: pd.DataFrame, models: Dict[str, Any], macro: Dict[str, Any]
) -> Tuple:
    occs = sorted(occ_list["occupation"].tolist())
    st.sidebar.markdown("### 👤 プロフィール設定")
    current_age = st.sidebar.number_input("現在の年齢", 20, 65, 30, step=1)
    current_exp = st.sidebar.number_input("現在の勤続年数", 0, 40, 5, step=1)
    current_income = st.sidebar.number_input(
        "現在の年収（万円）", 100, 3000, 450, step=10
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 予測モデル選択")
    _all_model_options = {
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
    available_models = {
        label: key for label, key in _all_model_options.items() if key in models
    }
    default_model = (
        "Custom Ridge（特徴量強化型）"
        if "Custom Ridge（特徴量強化型）" in available_models
        else list(available_models.keys())[-1]
    )
    model_label = st.sidebar.selectbox(
        "使用するAIモデル",
        list(available_models.keys()),
        index=list(available_models.keys()).index(default_model),
    )
    model_key = available_models[model_label]

    st.sidebar.divider()
    st.sidebar.markdown("### 💼 キャリア選択")
    all_cats = ["（すべて）"] + sorted(OCCUPATION_CATEGORIES.keys())

    cur_cat = st.sidebar.selectbox("現職の大分類", all_cats, index=0, key="cur_cat")
    cur_occs = (
        occs
        if cur_cat == "（すべて）"
        else sorted([o for o in occs if get_category(o) == cur_cat]) or occs
    )
    default_current = cur_occs.index("販売店員") if "販売店員" in cur_occs else 0
    current_occ = st.sidebar.selectbox(
        "現職名", cur_occs, index=default_current, key="cur_occ"
    )

    tgt_cat = st.sidebar.selectbox("目標の大分類", all_cats, index=0, key="tgt_cat")
    tgt_occs = (
        occs
        if tgt_cat == "（すべて）"
        else sorted([o for o in occs if get_category(o) == tgt_cat]) or occs
    )
    default_target = (
        tgt_occs.index("システムコンサルタント・設計者")
        if "システムコンサルタント・設計者" in tgt_occs
        else 0
    )
    target_occ = st.sidebar.selectbox(
        "目標職種名", tgt_occs, index=default_target, key="tgt_occ"
    )

    skill_transfer = (
        st.sidebar.slider(
            "経験引継ぎ率（%）", 0, 100, 20, help="0%＝完全未経験、100%＝即戦力。"
        )
        / 100
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 💰 投資設定")
    learning_cost = st.sidebar.number_input("自己投資費用（万円）", 0, 500, 50, step=5)
    gdp_growth = st.sidebar.slider(
        "期待GDP成長率（%）",
        -3.0,
        3.0,
        float(round(macro["avg_gdp_growth_10yr"] * 100, 2)),
        step=0.05,
    )
    future_cpi = st.sidebar.slider("将来のCPI（物価指数）", 80, 150, 105, step=1)
    nominal_raise = max(gdp_growth / 100 + (future_cpi - 100) / 100 * 0.3, 0.0)

    st.sidebar.divider()
    st.sidebar.markdown("### 🎯 リアリティ補正")
    raise_suppression = (
        st.sidebar.slider("転職後の昇給抑制（%）", 0, 50, 0, step=5) / 100
    )
    career_risk = st.sidebar.slider("キャリアリスク係数（%）", 0, 30, 0, step=5) / 100

    return (
        current_occ,
        target_occ,
        current_age,
        current_exp,
        current_income,
        skill_transfer,
        learning_cost,
        model_key,
        model_label,
        nominal_raise,
        gdp_growth,
        future_cpi,
        raise_suppression,
        career_risk,
    )


# ══════════════════════════════════════════════════════
# 各種ガイド描画関数
# ══════════════════════════════════════════════════════
def _render_skill_transfer_table(
    models: Dict[str, Any],
    model_key: str,
    target_occ: str,
    current_age: int,
    current_exp: float,
    current_income: float,
    age_all_path: str,
) -> None:
    base_income, base_label = _get_one_step_down_income(
        target_occ, current_age, age_all_path
    )
    exp_income = predict(models, model_key, target_occ, current_age, current_exp)

    rates = [0, 20, 40, 60, 80, 100]
    meanings = [
        "完全未経験スタート（1段下の年齢階級相当）",
        "前職の汎用スキルが少し評価される",
        "ドメイン知識がある程度活かせる",
        "近接領域・類似業務からの転職",
        "かなりのスキルが転用できる",
        "即戦力（資格・経験が完全移行）",
    ]

    data = []
    for rate, meaning in zip(rates, meanings):
        first = max(
            base_income + (exp_income - base_income) * (rate / 100), base_income * 0.8
        )
        diff = first - current_income
        diff_str = f"▲ {abs(diff):.0f}万円" if diff < 0 else f"+{diff:.0f}万円"
        data.append(
            {
                "引継ぎ率": f"{rate}%",
                "意味": meaning,
                f"初年度（現職→{target_occ[:12]}）": f"{first:.0f}万円 ({diff_str})",
            }
        )

    df = pd.DataFrame(data)
    with st.expander(
        f"📊 スキル引継ぎ率ガイド　※ベースライン: {target_occ[:15]} の {base_label} 平均年収 {base_income:.0f}万円",
        expanded=False,
    ):
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_macro_guide() -> None:
    gdp_data = [
        {
            "期待GDP成長率": "-2.0%",
            "シナリオの意味": "深刻な不況。賃金水準全体が低下する悲観シナリオ。",
        },
        {
            "期待GDP成長率": "0.0%",
            "シナリオの意味": "ゼロ成長。経済が停滞し、昇給は年齢・経験カーブのみに依存。",
        },
        {
            "期待GDP成長率": "1.0%",
            "シナリオの意味": "緩やかな成長。過去10年間の日本経済の平均的な水準（標準）。",
        },
        {
            "期待GDP成長率": "2.0%〜",
            "シナリオの意味": "安定成長。インフレに連動した健全なベースアップが見込める楽観シナリオ。",
        },
    ]
    cpi_data = [
        {
            "将来のCPI": "90",
            "シナリオの意味": "デフレ進行。物価が下落し、名目賃金の上昇が強く抑えられる。",
        },
        {
            "将来のCPI": "105",
            "シナリオの意味": "現状維持。直近の緩やかなインフレ傾向の継続（標準）。",
        },
        {
            "将来のCPI": "120",
            "シナリオの意味": "マイルドインフレ。物価上昇に合わせて賃金への還元が進む。",
        },
        {
            "将来のCPI": "140",
            "シナリオの意味": "高インフレ。急激な物価高騰による名目賃金の上押し圧力。",
        },
    ]
    with st.expander("🌍 マクロ経済シナリオガイド（GDP・CPI）", expanded=False):
        st.markdown("将来の日本全体の名目賃金上昇率に影響を与えます。")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                pd.DataFrame(gdp_data), use_container_width=True, hide_index=True
            )
        with col2:
            st.dataframe(
                pd.DataFrame(cpi_data), use_container_width=True, hide_index=True
            )


def _render_risk_guide() -> None:
    raise_data = [
        {
            "転職後の昇給抑制": "0%",
            "シナリオの意味": "抑制なし。AI予測通りの標準的な昇給を享受できる（理想的）。",
        },
        {
            "転職後の昇給抑制": "10%",
            "シナリオの意味": "軽微なビハインド。キャッチアップ期間として序盤の昇給がやや遅れる。",
        },
        {
            "転職後の昇給抑制": "30%",
            "シナリオの意味": "現実的な壁。未経験領域の評価構築に時間がかかり、昇給ペースが3割減速。",
        },
        {
            "転職後の昇給抑制": "50%",
            "シナリオの意味": "厳しい下積み。最初の数年間はベースアップがほぼ期待できない保守的シナリオ。",
        },
    ]
    risk_data = [
        {
            "キャリアリスク係数": "0%",
            "シナリオの意味": "リスクなし。生涯にわたり継続的に最新スキルをキャッチアップできる前提。",
        },
        {
            "キャリアリスク係数": "10%",
            "シナリオの意味": "標準的な陳腐化。中堅層以降で技術変化により若干の年収頭打ちが発生。",
        },
        {
            "キャリアリスク係数": "20%",
            "シナリオの意味": "シニア期の停滞。プレイングマネージャーとしての給与限界に直面する。",
        },
        {
            "キャリアリスク係数": "30%",
            "シナリオの意味": "スキルの陳腐化。技術パラダイムシフト等で10年後以降に市場価値が目減りする。",
        },
    ]
    with st.expander(
        "🛡️ リアリティ補正シナリオガイド（昇給抑制・キャリアリスク）", expanded=False
    ):
        st.markdown(
            "AIモデルが算出した「理想的な予測」に対して、現実的な下方修正を加えます。"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                pd.DataFrame(raise_data), use_container_width=True, hide_index=True
            )
        with col2:
            st.dataframe(
                pd.DataFrame(risk_data), use_container_width=True, hide_index=True
            )


# ══════════════════════════════════════════════════════
# 結果描画
# ══════════════════════════════════════════════════════
def render_analysis_results(
    sq,
    cc,
    current_age,
    current_occ,
    target_occ,
    current_income,
    skill_transfer,
    learning_cost,
):
    ages_to_65 = max(0, 65 - current_age)

    current_monthly = current_income / 14
    first_monthly = cc[0] / 14
    idx5 = min(5, len(sq) - 1)
    sq5_annual = sq[idx5]
    cc5_annual = cc[idx5]
    sq5_monthly = sq5_annual / 14
    cc5_monthly = cc5_annual / 14
    sq_lifetime = sum(sq[:ages_to_65])
    cc_lifetime = sum(cc[:ages_to_65])
    net_benefit = cc_lifetime - sq_lifetime - learning_cost
    breakeven_month, _ = calc_roi(sq, cc, learning_cost)
    roi_pct = (
        (net_benefit / max(learning_cost, 1)) * 100
        if learning_cost > 0
        else float("inf")
    )

    def v(val, fmt=".1f", unit="万円", cls="val"):
        return f"<span class='{cls}'>{val:{fmt}}{unit}</span>"

    def diff_span(val, unit="万円", fmt=".1f"):
        cls = "pos" if val >= 0 else "neg"
        sign = "+" if val >= 0 else ""
        return f"<span class='{cls}'>{sign}{val:{fmt}}{unit}</span>"

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            f"<div class='result-section'>"
            f"<h4>1. 現状（現在年齢）</h4><ul>"
            f"<li>月収: {v(current_monthly)}</li>"
            f"<li>年収: {v(current_income)}</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='result-section'>"
            f"<h4>2. 転職直後（初年度）</h4><ul>"
            f"<li>月収: {v(first_monthly)}</li>"
            f"<li>年収: {v(cc[0])}</li>"
            f"<li style='font-size:.78rem;opacity:0.7;list-style:none;margin-left:-1.2rem;margin-top:4px'>※スキル引継ぎ率 {int(skill_transfer*100)}% を適用済み</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )

        net_cls = "pos" if net_benefit >= 0 else "neg"
        st.markdown(
            f"<div class='result-section'>"
            f"<h4>4. 65歳時点での生涯年収</h4><ul>"
            f"<li>転職しなかった場合: {v(sq_lifetime, fmt=',.0f')}</li>"
            f"<li>転職した場合: {v(cc_lifetime, fmt=',.0f')}</li>"
            f"</ul>"
            f"<div style='font-size:.78rem;opacity:0.7;margin-top:.4rem'>生涯差額（投資コスト控除後）</div>"
            f"<div class='lifetime-highlight {net_cls}'>{net_benefit:+,.0f} 万円</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            f"<div class='result-section'>"
            f"<h4>3. 転職から5年後</h4><ul>"
            f"<li>転職前の月収: {v(sq5_monthly)}</li>"
            f"<li>転職後の月収: {v(cc5_monthly)}</li>"
            f"<li>月収差額: {diff_span(cc5_monthly - sq5_monthly)}</li>"
            f"</ul><ul style='margin-top:.5rem'>"
            f"<li>転職前の年収: {v(sq5_annual, fmt=',.1f')}</li>"
            f"<li>転職後の年収: {v(cc5_annual, fmt=',.1f')}</li>"
            f"<li>年収差額: {diff_span(cc5_annual - sq5_annual, fmt=',.1f')}</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )

        if breakeven_month:
            be_str = f"{breakeven_month//12} ヶ月（{'1年以内' if breakeven_month <= 12 else f'{breakeven_month//12}年{breakeven_month%12}か月'}）"
        else:
            be_str = "回収困難"
        roi_str = f"{roi_pct:,.1f} %" if roi_pct != float("inf") else "∞（費用0円）"

        st.markdown(
            f"<div class='result-section'>"
            f"<h4>5. 費用対効果</h4><ul>"
            f"<li>投資コスト回収までの月数: {v(be_str, fmt='', unit='', cls='pos' if breakeven_month and breakeven_month <= 12 else 'neu')}</li>"
            f"<li>生涯年収ベースのROI: <span class='{'pos' if roi_pct > 0 else 'neg'}'>{roi_str}</span></li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.title("📊 リスキリングによる年収シミュレーター")

    try:
        models, occ_list, age_curve, macro = load_assets()
    except Exception as e:
        st.error(f"起動エラー: {e}")
        st.stop()

    (
        current_occ,
        target_occ,
        current_age,
        current_exp,
        current_income,
        skill_transfer,
        learning_cost,
        model_key,
        model_label,
        nominal_raise,
        gdp_growth,
        future_cpi,
        raise_suppression,
        career_risk,
    ) = render_sidebar(occ_list, models, macro)

    if st.button("🚀 シミュレーション実行", type="primary"):
        st.session_state["sim_done"] = True
        st.session_state["sim_params"] = dict(
            current_occ=current_occ,
            target_occ=target_occ,
            current_age=current_age,
            current_exp=current_exp,
            current_income=current_income,
            skill_transfer=skill_transfer,
            learning_cost=learning_cost,
            model_key=model_key,
            model_label=model_label,
            nominal_raise=nominal_raise,
            raise_suppression=raise_suppression,
            career_risk=career_risk,
            gdp_growth=gdp_growth,
            future_cpi=future_cpi,
        )

    if not st.session_state.get("sim_done"):
        st.info(
            "👈 サイドバーで条件を設定し、「シミュレーション実行」ボタンを押してください。"
        )
        return

    p = st.session_state["sim_params"]
    status_quo_incomes, career_change_incomes = simulate(
        models,
        p["model_key"],
        p["current_occ"],
        p["target_occ"],
        p["current_age"],
        p["current_exp"],
        p["current_income"],
        p["skill_transfer"],
        p["nominal_raise"],
        age_curve,
        age_all_path=AGE_ALL_PATH,
        raise_suppression=p.get("raise_suppression", 0.0),
        career_risk=p.get("career_risk", 0.0),
    )

    MODEL_KEYS = [
        k
        for k in [
            "ridge",
            "elasticnet",
            "custom",
            "random_forest",
            "gradient_boosting",
            "lightgbm",
            "catboost",
            "xgboost",
            "stacking",
        ]
        if k in models
    ]
    MODEL_LABELS_MAP = {
        "ridge": "Ridge",
        "elasticnet": "ElasticNet",
        "custom": "Custom Ridge",
        "random_forest": "Random Forest",
        "gradient_boosting": "GradientBoosting",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
        "stacking": "Stacking",
    }
    sq_all, cc_all, roi_all = [], [], []
    for key in MODEL_KEYS:
        s, c = simulate(
            models,
            key,
            p["current_occ"],
            p["target_occ"],
            p["current_age"],
            p["current_exp"],
            p["current_income"],
            p["skill_transfer"],
            p["nominal_raise"],
            age_curve,
            age_all_path=AGE_ALL_PATH,
            raise_suppression=p.get("raise_suppression", 0.0),
            career_risk=p.get("career_risk", 0.0),
        )
        be, benefit = calc_roi(s, c, p["learning_cost"])
        sq_all.append(s)
        cc_all.append(c)
        roi_all.append((be, benefit))

    st.markdown("### 📋 分析結果")
    render_analysis_results(
        status_quo_incomes,
        career_change_incomes,
        p["current_age"],
        p["current_occ"],
        p["target_occ"],
        p["current_income"],
        p["skill_transfer"],
        p["learning_cost"],
    )

    st.markdown("### 📈 年収推移グラフ")
    fig_main = plot_main_plotly(
        status_quo_incomes,
        career_change_incomes,
        p["current_age"],
        p["current_occ"],
        p["target_occ"],
        p["learning_cost"],
    )
    st.plotly_chart(fig_main, use_container_width=True, theme="streamlit")

    with st.expander("🤖 モデル精度情報（クリックで展開）"):
        meta_path = os.path.join(MODEL_DIR, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta_all = json.load(f)
            n, ncols = len(meta_all), min(3, len(meta_all))
            items, best = list(meta_all.items()), max(
                v["r2_train"] for v in meta_all.values()
            )
            for row_start in range(0, n, ncols):
                for col, (key, m) in zip(
                    st.columns(len(items[row_start : row_start + ncols])),
                    items[row_start : row_start + ncols],
                ):
                    with col.container(border=True):
                        st.markdown(f"**{m['label']}**")
                        st.metric("R² Score", f"{m['r2_train']}")
                        st.caption(
                            f"CV={m['r2_cv_mean']}±{m['r2_cv_std']} / MAE={m['mae_train']}万円"
                        )

        st.markdown("---")
        st.markdown("#### 📖 各モデルの特徴と使い分け")
        MODEL_DESCRIPTIONS = [
            (
                "🔵 Ridge Regression",
                "linear",
                "線形回帰にL2正則化を加えたシンプルモデル。職種・年齢・経験年数の主効果を線形に捉える。過学習しにくく安定した予測が特徴。標準的なキャリアパスの基準として使いやすい。",
            ),
            (
                "🔵 ElasticNet",
                "linear",
                "RidgeとLassoを融合した線形モデル。不要な特徴量の係数を自動でゼロに近づける（スパース性）。解釈性が高く、影響の強い特徴量を絞り込んで学習する。",
            ),
            (
                "🔵 Custom Ridge",
                "linear",
                "年齢²・年齢×経験年数の交互作用項など独自特徴量を追加した線形モデル。給与ピーク帯（35〜54歳）のフラグも組み込み、年収カーブの非線形な動きを線形モデルで近似する。",
            ),
            (
                "🟢 Random Forest",
                "tree",
                "複数の決定木を組み合わせたバギングモデル。職種ごとの細かい条件分岐を学習するが、外挿（訓練データ範囲外の予測）が苦手。CV精度は低めだが、特定職種の上振れ・下振れシナリオ確認に有効。",
            ),
            (
                "🟢 Gradient Boosting",
                "tree",
                "弱い決定木を順番に積み上げてエラーを修正する勾配ブースティング。sklearnの標準実装で追加インストール不要。XGBoostより低速だが安定性が高く、過学習に強い。",
            ),
            (
                "🟡 LightGBM",
                "boosting",
                "MicrosoftのLightGBMは葉ごとに成長する「Leaf-wise」戦略で高速。職種名をカテゴリ変数としてネイティブに処理でき、OHEが不要。大規模データでも実用的な速度で学習できる。",
            ),
            (
                "🟡 CatBoost",
                "boosting",
                "Yandexのカテゴリ特化ブースティング。Target Encodingを対称木とともに最適化する独自手法で職種名を扱う。ハイパーパラメータのデフォルト値が優秀でチューニング不要でも高精度。",
            ),
            (
                "🟡 XGBoost",
                "boosting",
                "勾配ブースティングの業界標準。L1/L2正則化・欠損値の自動処理・並列化など多くの最適化を備える。特徴量エンジニアリング（FE）を組み合わせることでさらに精度が向上。",
            ),
            (
                "🏆 Stacking Ensemble",
                "ensemble",
                "全ベースモデルのOOF（Out-of-Fold）予測をメタ特徴量としてRidgeで統合する2層アンサンブル。単一モデルでは捉えられない「モデル間の誤差の相補関係」を学習し、理論上最高精度を目指す。ただし訓練時間は最長。",
            ),
        ]
        type_color = {
            "linear": "#4F8EF7",
            "tree": "#4CAF50",
            "boosting": "#FF9800",
            "ensemble": "#E040FB",
        }
        type_label = {
            "linear": "線形系",
            "tree": "ツリー系",
            "boosting": "ブースティング系",
            "ensemble": "アンサンブル",
        }

        for model_name, mtype, desc in MODEL_DESCRIPTIONS:
            clr = type_color[mtype]
            lbl = type_label[mtype]
            st.markdown(
                f"<div style='border-left:3px solid {clr};padding:.4rem .8rem;margin:.35rem 0;'>"
                f"<span style='font-weight:700;color:{clr}'>{model_name}</span>"
                f"<span style='font-size:.7rem;background:{clr}22;color:{clr}; border-radius:4px;padding:1px 6px;margin-left:8px'>{lbl}</span>"
                f"<div style='font-size:.82rem;margin-top:.25rem;opacity:0.9;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ガイド類の展開（3つ並べる）
    _render_skill_transfer_table(
        models,
        p["model_key"],
        p["target_occ"],
        p["current_age"],
        p["current_exp"],
        p["current_income"],
        AGE_ALL_PATH,
    )
    _render_macro_guide()
    _render_risk_guide()

    st.markdown("### 📉 全モデル比較グラフ")
    fig3 = plot_all_models_plotly(
        sq_all,
        cc_all,
        p["current_age"],
        [r[0] for r in roi_all],
        [MODEL_LABELS_MAP[k] for k in MODEL_KEYS],
    )
    st.plotly_chart(fig3, use_container_width=True, theme="streamlit")

    st.markdown("### 📊 年次詳細（選択モデル・5年刻み）")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "年齢": f"{p['current_age'] + i}歳",
                    "現状維持（万円）": f"{status_quo_incomes[i]:,.0f}",
                    "転職後（万円）": f"{career_change_incomes[i]:,.0f}",
                    "年間差（万円）": f"{career_change_incomes[i] - status_quo_incomes[i]:+,.0f}",
                }
                for i in range(0, 50, 5)
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    rs = p.get("raise_suppression", 0.0)
    cr = p.get("career_risk", 0.0)
    st.caption(
        f"📌 使用モデル: {p['model_label']} ／ "
        f"GDP: {p.get('gdp_growth', 0.0):+.2f}% ／ CPI: {p.get('future_cpi', 105)} ／ "
        f"昇給抑制: {rs*100:.0f}% ／ キャリアリスク: {cr*100:.0f}% ／ "
        "本シミュレーションは厚生労働省「賃金構造基本統計調査」・GDP・CPI をもとにした統計的推計です。"
        "個人の実際の収入を保証するものではありません。"
    )


if __name__ == "__main__":
    main()
