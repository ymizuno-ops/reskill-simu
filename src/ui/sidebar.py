from __future__ import annotations
from typing import NamedTuple
import pandas as pd
import streamlit as st
from model_types import MacroParams, ModelDict
from occupation import OCCUPATION_CATEGORIES, buildCategoryOccMap
from simulation import RETIREMENT_AGE

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
_ALL_CATEGORIES = "（すべて）"

# 入力欄の (最小, 最大, 初期値, 刻み)
# 年齢の上限は退職年齢の前年（初年度が退職年齢以上だとシミュレーションの前提が成り立たない）
AGE_INPUT = (20, RETIREMENT_AGE - 1, 30, 1)
EXP_INPUT = (0, 40, 5, 1)
INCOME_INPUT = (100, 3000, 450, 10)              # 万円
SKILL_TRANSFER_INPUT = (0, 100, 20)              # % (最小, 最大, 初期値)
LEARNING_COST_INPUT = (0, 500, 50, 5)            # 万円
GDP_GROWTH_INPUT = (-3.0, 3.0, 0.05)             # % (最小, 最大, 刻み)。初期値は過去10年平均
FUTURE_CPI_INPUT = (80, 150, 105, 1)
RAISE_SUPPRESSION_INPUT = (0, 50, 0, 5)          # %
CAREER_RISK_INPUT = (0, 30, 0, 5)                # %

PERCENT = 100
CPI_BASE = 100
CPI_TO_RAISE_WEIGHT = 0.3   # CPI の上昇分のうち名目昇給に反映する割合
GDP_DIGITS = 2


class SidebarInputs(NamedTuple):
    currentOcc: str
    targetOcc: str
    currentAge: int
    currentExp: int
    currentIncome: int
    skillTransfer: float
    learningCost: int
    modelKey: str
    modelLabel: str
    nominalRaise: float
    gdpGrowth: float
    futureCpi: int
    raiseSuppression: float
    careerRisk: float
    isSubmitted: bool


def _selectOccupation(label: str, key: str, cats: list[str], catOccMap: dict[str, list[str]],
                      occs: list[str], defaultOcc: str, occLabel: str, occKey: str) -> str:
    """大分類 → 職種名の順に選ばせる。大分類に職種がなければ全職種から選ぶ"""
    cat = st.sidebar.selectbox(label, cats, index=0, key=key)
    catOccs = sorted(catOccMap.get(cat, [])) if cat != _ALL_CATEGORIES else occs
    if not catOccs:
        catOccs = occs
    defaultIdx = catOccs.index(defaultOcc) if defaultOcc in catOccs else 0
    return st.sidebar.selectbox(occLabel, catOccs, index=defaultIdx, key=occKey)


def renderSidebar(
    occList: pd.DataFrame,
    models: ModelDict,
    macro: MacroParams,
) -> SidebarInputs:
    occs = sorted(occList["occupation"].tolist())
    catOccMap = buildCategoryOccMap(occs)
    allCats = [_ALL_CATEGORIES] + sorted(OCCUPATION_CATEGORIES.keys())

    isSubmitted = st.sidebar.button("🚀 シミュレーション実行", type="primary", use_container_width=True)

    st.sidebar.markdown("### 👤 プロフィール設定")
    currentAge = st.sidebar.number_input("現在の年齢", *AGE_INPUT[:3], step=AGE_INPUT[3])
    currentExp = st.sidebar.number_input("現在の勤続年数", *EXP_INPUT[:3], step=EXP_INPUT[3])
    currentIncome = st.sidebar.number_input("現在の年収（万円）", *INCOME_INPUT[:3], step=INCOME_INPUT[3])

    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 予測モデル選択")
    availableModels = {label: key for label, key in _ALL_MODEL_OPTIONS.items() if key in models}
    modelLabels = list(availableModels.keys())
    defaultLabel = _DEFAULT_MODEL_LABEL if _DEFAULT_MODEL_LABEL in availableModels else modelLabels[-1]
    modelLabel = st.sidebar.selectbox("使用するAIモデル", modelLabels, index=modelLabels.index(defaultLabel))
    modelKey = availableModels[modelLabel]

    st.sidebar.divider()
    st.sidebar.markdown("### 💼 キャリア選択")
    currentOcc = _selectOccupation("現職の大分類", "cur_cat", allCats, catOccMap, occs,
                                   _DEFAULT_CURRENT_OCC, "現職名", "cur_occ")
    targetOcc = _selectOccupation("目標の大分類", "tgt_cat", allCats, catOccMap, occs,
                                  _DEFAULT_TARGET_OCC, "目標職種名", "tgt_occ")

    skillTransfer = st.sidebar.slider(
        "経験引継ぎ率（%）", *SKILL_TRANSFER_INPUT, help="0%＝完全未経験、100%＝即戦力。"
    ) / PERCENT

    st.sidebar.divider()
    st.sidebar.markdown("### 💰 投資設定")
    learningCost = st.sidebar.number_input("自己投資費用（万円）", *LEARNING_COST_INPUT[:3], step=LEARNING_COST_INPUT[3])
    gdpMin, gdpMax, gdpStep = GDP_GROWTH_INPUT
    gdpGrowth = st.sidebar.slider(
        "期待GDP成長率（%）", gdpMin, gdpMax,
        float(round(macro["avg_gdp_growth_10yr"] * PERCENT, GDP_DIGITS)), step=gdpStep,
    )
    futureCpi = st.sidebar.slider("将来のCPI（物価指数）", *FUTURE_CPI_INPUT[:3], step=FUTURE_CPI_INPUT[3])
    nominalRaise = max(gdpGrowth / PERCENT + (futureCpi - CPI_BASE) / PERCENT * CPI_TO_RAISE_WEIGHT, 0.0)

    st.sidebar.divider()
    st.sidebar.markdown("### 🎯 リアリティ補正")
    raiseSuppression = st.sidebar.slider(
        "転職後の昇給抑制（%）", *RAISE_SUPPRESSION_INPUT[:3], step=RAISE_SUPPRESSION_INPUT[3]
    ) / PERCENT
    careerRisk = st.sidebar.slider(
        "キャリアリスク係数（%）", *CAREER_RISK_INPUT[:3], step=CAREER_RISK_INPUT[3]
    ) / PERCENT

    return SidebarInputs(
        currentOcc, targetOcc, currentAge, currentExp, currentIncome,
        skillTransfer, learningCost, modelKey, modelLabel,
        nominalRaise, gdpGrowth, futureCpi, raiseSuppression, careerRisk,
        isSubmitted,
    )
