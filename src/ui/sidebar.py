# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# NamedTuple: 各要素に名前を付けたタプル。p.currentAge のように名前で値を取り出せる。
from typing import NamedTuple
import pandas as pd
# Streamlit: Python だけで Web 画面を作るライブラリ。st.〜 で画面の部品を置く。
import streamlit as st
from model_types import MacroParams, ModelDict
from occupation import OCCUPATION_CATEGORIES, buildCategoryOccMap
from simulation import RETIREMENT_AGE

# 意味: 画面のモデル選択肢の表示名（左）と、学習済みモデルの名前（右）の対応。
# 影響: 左の表示名は自由に変えてよい。注意: 右の名前は変えない。表示名を変えたら _DEFAULT_MODEL_LABEL も同じ表記にする。
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
# 意味: 画面を開いたときに選ばれているモデル。
# 注意: _ALL_MODEL_OPTIONS の表示名と一字一句同じにする。違うと一覧の最後のモデルが選ばれる。
_DEFAULT_MODEL_LABEL = "Custom Ridge（特徴量強化型）"
# 意味: 画面を開いたときの「現職名」の初期値。
# 注意: 職種マスタ（data/master/occupation_list.csv）の職種名と一字一句同じにする。違うと一覧の先頭の職種が選ばれる。
_DEFAULT_CURRENT_OCC = "販売店員"
# 意味: 画面を開いたときの「目標職種名」の初期値。注意: 上と同じく、職種マスタの職種名と一致させる。
_DEFAULT_TARGET_OCC = "システムコンサルタント・設計者"
# 意味: 大分類の選択肢のうち「絞り込まない」を表す文言。変えると表示の文言だけが変わる。
_ALL_CATEGORIES = "（すべて）"

# 入力欄の (最小, 最大, 初期値, 刻み)
# 年齢の上限は退職年齢の前年（初年度が退職年齢以上だとシミュレーションの前提が成り立たない）
# 意味: 「現在の年齢」の下限・上限・初期値・刻み（歳）。
# 注意: 上限は退職年齢から自動で決まるため、数字を書き込まない。
AGE_INPUT = (20, RETIREMENT_AGE - 1, 30, 1)
# 意味: 「現在の勤続年数」の下限・上限・初期値・刻み（年）。現状維持の年収予測の起点になる。
EXP_INPUT = (0, 40, 5, 1)
# 意味: 「現在の年収」の下限・上限・初期値・刻み（万円）。
# 注意: 下限・上限は、学習データの職種の年収範囲（step2_to_master.py の 100〜3000万円）とそろえている。
INCOME_INPUT = (100, 3000, 450, 10)              # 万円
# 意味: 「経験引継ぎ率」スライダーの下限・上限・初期値（%）。
SKILL_TRANSFER_INPUT = (0, 100, 20)              # % (最小, 最大, 初期値)
# 意味: 「自己投資費用」の下限・上限・初期値・刻み（万円）。
# 影響: 初期値を変えると、開いたときの回収期間と ROI の表示が変わる。
LEARNING_COST_INPUT = (0, 500, 50, 5)            # 万円
# 意味: 「期待GDP成長率」スライダーの下限・上限・刻み（%）。初期値は macro_params.json の過去10年平均を使う。
GDP_GROWTH_INPUT = (-3.0, 3.0, 0.05)             # % (最小, 最大, 刻み)。初期値は過去10年平均
# 意味: 「将来のCPI」（2020年 = 100 とした物価指数）スライダーの下限・上限・初期値・刻み。
FUTURE_CPI_INPUT = (80, 150, 105, 1)
# 意味: 「転職後の昇給抑制」スライダーの下限・上限・初期値・刻み（%）。
RAISE_SUPPRESSION_INPUT = (0, 50, 0, 5)          # %
# 意味: 「キャリアリスク係数」スライダーの下限・上限・初期値・刻み（%）。
CAREER_RISK_INPUT = (0, 30, 0, 5)                # %

PERCENT = 100
# 意味: CPI の基準値（2020年 = 100）。統計の定義なので変えない。
CPI_BASE = 100
# 意味: 物価の上昇分のうち、賃金の昇給に反映される割合（0.3 = 3割）。
# 影響: 大きくすると、CPI スライダーを上げたときの名目昇給率が大きくなり、現状維持・転職後の両方の年収が上がる。
CPI_TO_RAISE_WEIGHT = 0.3   # CPI の上昇分のうち名目昇給に反映する割合
GDP_DIGITS = 2


# サイドバーの入力値をまとめて返すための型。「名前: 型」を並べるだけで定義できる。
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


# 「大分類 → 職種名」の2段階の選択欄を置き、選ばれた職種名を返す。
def _selectOccupation(label: str, key: str, cats: list[str], catOccMap: dict[str, list[str]],
                      occs: list[str], defaultOcc: str, occLabel: str, occKey: str) -> str:
    """大分類 → 職種名の順に選ばせる。大分類に職種がなければ全職種から選ぶ"""
    # selectbox はプルダウン。index は最初に選ばれている位置、key は Streamlit が入力値を覚えておくための名前。
    cat = st.sidebar.selectbox(label, cats, index=0, key=key)
    catOccs = sorted(catOccMap.get(cat, [])) if cat != _ALL_CATEGORIES else occs
    if not catOccs:
        catOccs = occs
    defaultIdx = catOccs.index(defaultOcc) if defaultOcc in catOccs else 0
    return st.sidebar.selectbox(occLabel, catOccs, index=defaultIdx, key=occKey)


# サイドバー全体を描画し、入力値をまとめて返す。Streamlit は画面を操作するたびに、この関数を含む画面全体を上から実行し直す。
def renderSidebar(
    occList: pd.DataFrame,
    models: ModelDict,
    macro: MacroParams,
) -> SidebarInputs:
    # tolist で列をリストに変換し、sorted で並べる。
    occs = sorted(occList["occupation"].tolist())
    catOccMap = buildCategoryOccMap(occs)
    # 「リスト + リスト」でつなげる。先頭に「すべて」を置く。
    allCats = [_ALL_CATEGORIES] + sorted(OCCUPATION_CATEGORIES.keys())

    # ボタン。押された直後の実行のときだけ True が返る。
    isSubmitted = st.sidebar.button("🚀 シミュレーション実行", type="primary", use_container_width=True)

    st.sidebar.markdown("### 👤 プロフィール設定")
    # [:3] は先頭から3つ（最小・最大・初期値）。* で展開して、位置の順に引数として渡す。
    currentAge = st.sidebar.number_input("現在の年齢", *AGE_INPUT[:3], step=AGE_INPUT[3])
    currentExp = st.sidebar.number_input("現在の勤続年数", *EXP_INPUT[:3], step=EXP_INPUT[3])
    currentIncome = st.sidebar.number_input("現在の年収（万円）", *INCOME_INPUT[:3], step=INCOME_INPUT[3])

    # 区切りの線を引く。
    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 予測モデル選択")
    # 辞書内包表記に if を付けて、学習済みのモデルだけに絞る。
    availableModels = {label: key for label, key in _ALL_MODEL_OPTIONS.items() if key in models}
    modelLabels = list(availableModels.keys())
    defaultLabel = _DEFAULT_MODEL_LABEL if _DEFAULT_MODEL_LABEL in availableModels else modelLabels[-1]
    # 最初に選ばれている位置を、表示名のリストの中の位置（index）で指定する。
    modelLabel = st.sidebar.selectbox("使用するAIモデル", modelLabels, index=modelLabels.index(defaultLabel))
    modelKey = availableModels[modelLabel]

    st.sidebar.divider()
    st.sidebar.markdown("### 💼 キャリア選択")
    currentOcc = _selectOccupation("現職の大分類", "cur_cat", allCats, catOccMap, occs,
                                   _DEFAULT_CURRENT_OCC, "現職名", "cur_occ")
    targetOcc = _selectOccupation("目標の大分類", "tgt_cat", allCats, catOccMap, occs,
                                  _DEFAULT_TARGET_OCC, "目標職種名", "tgt_occ")

    # slider はスライダー。% で受け取り、/ PERCENT で 0〜1 の割合に直す。
    skillTransfer = st.sidebar.slider(
        "経験引継ぎ率（%）", *SKILL_TRANSFER_INPUT, help="0%＝完全未経験、100%＝即戦力。"
    ) / PERCENT

    st.sidebar.divider()
    st.sidebar.markdown("### 💰 投資設定")
    learningCost = st.sidebar.number_input("自己投資費用（万円）", *LEARNING_COST_INPUT[:3], step=LEARNING_COST_INPUT[3])
    # タプルの3つの値を、3つの変数に分けて受け取る。
    gdpMin, gdpMax, gdpStep = GDP_GROWTH_INPUT
    gdpGrowth = st.sidebar.slider(
        "期待GDP成長率（%）", gdpMin, gdpMax,
        float(round(macro["avg_gdp_growth_10yr"] * PERCENT, GDP_DIGITS)), step=gdpStep,
    )
    futureCpi = st.sidebar.slider("将来のCPI（物価指数）", *FUTURE_CPI_INPUT[:3], step=FUTURE_CPI_INPUT[3])
    # 名目昇給率 = GDP 成長率 + 物価の上昇分 × 反映の割合。マイナスにはしない。
    nominalRaise = max(gdpGrowth / PERCENT + (futureCpi - CPI_BASE) / PERCENT * CPI_TO_RAISE_WEIGHT, 0.0)

    st.sidebar.divider()
    st.sidebar.markdown("### 🎯 リアリティ補正")
    raiseSuppression = st.sidebar.slider(
        "転職後の昇給抑制（%）", *RAISE_SUPPRESSION_INPUT[:3], step=RAISE_SUPPRESSION_INPUT[3]
    ) / PERCENT
    careerRisk = st.sidebar.slider(
        "キャリアリスク係数（%）", *CAREER_RISK_INPUT[:3], step=CAREER_RISK_INPUT[3]
    ) / PERCENT

    # クラスの定義の順番どおりに値を並べて作る。
    return SidebarInputs(
        currentOcc, targetOcc, currentAge, currentExp, currentIncome,
        skillTransfer, learningCost, modelKey, modelLabel,
        nominalRaise, gdpGrowth, futureCpi, raiseSuppression, careerRisk,
        isSubmitted,
    )
