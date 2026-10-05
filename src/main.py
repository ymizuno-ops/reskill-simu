# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
import os
# sys: import の検索先（sys.path）を操作するために読み込む。
import sys
import json
# pickle: 保存した Python のオブジェクト（学習済みモデル）を復元する標準モジュール。
import pickle
import warnings

import pandas as pd
# Streamlit: Python だけで Web 画面を作るライブラリ。
import streamlit as st

# ライブラリの警告を表示しない。
warnings.filterwarnings("ignore")
# このファイルのフォルダ（src/）を import の検索先の先頭に加える。streamlit run で起動しても、自作のモジュールを読み込めるようにするため。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# models.pkl の復元に必要（step3_train.py を直接実行して保存した場合、クラスは __main__ 側の名前で探される）
from model_wrappers import LGBMWrapper, CatBoostWrapper, StackingEnsemble  # noqa: F401

from log_config import getLogger
from model_types import MacroParams, ModelDict
from simulation import simulate, calcRoi
from ui.sidebar import SidebarInputs, renderSidebar
from ui.charts import plotMainPlotly, plotAllModelsPlotly
from ui.guides import renderPreSimGuides, renderPostSimGuides
from ui.results import renderAnalysisResults

# このファイル用のロガー。
logger = getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリのルート
# 意味: 画面が読む職種マスタ・マクロ経済パラメータの場所。注意: step2_to_master.py の OUT_DIR と同じ場所にする。
MASTER_DIR = os.path.join(_ROOT, "data", "master")
# 意味: 学習済みモデルの場所。ここに models.pkl がないと、初回起動時に Step1〜3 を自動で実行する。
MODEL_DIR = os.path.join(_ROOT, "models")
# 意味: 転職直後の基準年収を引く、年齢別の統計 CSV。注意: step1_common.py の OUT_DIR の下にあるファイル。
AGE_ALL_PATH = os.path.join(_ROOT, "data", "processed", "age_wage_all.csv")
# 影響: 小さくすると年次詳細の表の行が増える。注意: 表の見出し「5年刻み」の文言は別に書いてあるので、変えたら合わせて直す。
TABLE_STEP_YEARS = 5   # 年次詳細の表の刻み（年）

# 意味: 全モデル比較グラフに並べるモデルの順番。
# 注意: 名前は学習済みモデルの名前なので書き換えない。並び順だけ入れ替えてよい。
_MODEL_KEY_ORDER = [
    "ridge", "elasticnet", "custom", "random_forest", "gradient_boosting",
    "lightgbm", "catboost", "xgboost", "stacking",
]
# 意味: 全モデル比較グラフの小見出し。右の表示名は自由に変えてよく、左の名前は変えない。
_MODEL_LABEL_MAP = {
    "ridge": "Ridge", "elasticnet": "ElasticNet", "custom": "Custom Ridge",
    "random_forest": "Random Forest", "gradient_boosting": "GradientBoosting",
    "lightgbm": "LightGBM", "catboost": "CatBoost", "xgboost": "XGBoost",
    "stacking": "Stacking",
}

# ページのタイトル・アイコン・レイアウトの設定。Streamlit の命令の中で最初に呼ぶ必要がある。
st.set_page_config(
    # 意味: ブラウザのタブに出るアプリ名。
    page_title="リスキリングによる年収シミュレーター",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 画面全体の見た目を調整する CSS（<style> タグ）を埋め込む。
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


# @ はデコレーター（関数に機能を足す仕組み）。cache_resource で1回目の結果を覚えておき、画面を操作するたびに読み込み直さないようにする。
@st.cache_resource(show_spinner="モデルを読み込み中...")
# 戻り値は (モデル, 職種マスタ, 年齢カーブ, マクロ経済パラメータ) の4つ組。
def loadAssets() -> tuple[ModelDict, pd.DataFrame, pd.DataFrame, MacroParams]:
    pklPath = os.path.join(MODEL_DIR, "models.pkl")
    # 学習済みのモデルがまだなければ、Step1〜3 を実行してデータとモデルを作る。
    if not os.path.exists(pklPath):
        # 処理の間、くるくる回る表示を出す。
        with st.spinner("初回起動: データ加工 & モデル訓練中（1〜2 分）"):
            # 必要なときだけ import する。as で別名を付け、3つの main を区別する。
            from step1_to_processed import main as runStep1
            from step2_to_master import main as runStep2
            from step3_train import main as runStep3
            runStep1()
            runStep2()
            runStep3()

    with open(pklPath, "rb") as f:
        # 保存しておいたモデルの辞書を復元する。
        models = pickle.load(f)

    occList = pd.read_csv(os.path.join(MASTER_DIR, "occupation_list.csv"))
    ageCurve = pd.read_csv(os.path.join(MASTER_DIR, "age_curve.csv"))
    with open(os.path.join(MASTER_DIR, "macro_params.json"), encoding="utf-8") as f:
        macro = json.load(f)

    return models, occList, ageCurve, macro


# 意味: 初回表示の注意事項ダイアログ。見出しはこの行の文字列、本文は下の関数の中の文字列を直接書き換えてよい。
@st.dialog("⚠️ ご利用にあたっての注意事項")
# 注意事項のダイアログ（画面の上に重なる小窓）の中身。
def _showDisclaimer() -> None:
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
    # 押されたら、同意したことを session_state（画面の再実行をまたいで値を覚えておく場所）に記録し、rerun で画面を描き直す。
    if st.button("上記に同意して始める", type="primary", use_container_width=True):
        st.session_state["disclaimer_accepted"] = True
        st.rerun()


# 入力値から simulate を呼ぶ共通の処理。
def _simulate(models: ModelDict, modelKey: str, p: SidebarInputs,
              ageCurve: pd.DataFrame) -> tuple[list[float], list[float]]:
    return simulate(
        models, modelKey,
        p.currentOcc, p.targetOcc,
        p.currentAge, p.currentExp, p.currentIncome,
        p.skillTransfer, p.nominalRaise, ageCurve,
        ageAllPath=AGE_ALL_PATH,
        raiseSuppression=p.raiseSuppression,
        careerRisk=p.careerRisk,
    )


# 全モデルで同じ条件のシミュレーションを行い、比較グラフ用の結果をまとめる。
def _runAllModelSimulations(
    models: ModelDict, p: SidebarInputs, ageCurve: pd.DataFrame
) -> tuple[list[str], list[list[float]], list[list[float]], list[tuple[int | None, float]]]:
    modelKeys = [k for k in _MODEL_KEY_ORDER if k in models]
    sqAll, ccAll, roiAll = [], [], []
    for key in modelKeys:
        statusQuo, careerChange = _simulate(models, key, p, ageCurve)
        sqAll.append(statusQuo)
        ccAll.append(careerChange)
        roiAll.append(calcRoi(statusQuo, careerChange, p.learningCost))
    return modelKeys, sqAll, ccAll, roiAll


# 画面全体の処理。Streamlit は画面を操作するたびに、このファイルを上から実行し直す。
def main() -> None:
    st.title("📊 リスキリングによる年収シミュレーター")

    # まだ同意していなければダイアログを出し、st.stop で以降の描画を止める。
    if not st.session_state.get("disclaimer_accepted", False):
        _showDisclaimer()
        st.stop()

    try:
        models, occList, ageCurve, macro = loadAssets()
    # 読み込みに失敗したら、ログに詳細を残し、画面にもエラーを出して止める。
    except Exception as e:
        logger.exception("起動時のデータ・モデル読み込みに失敗した")
        st.error(f"起動エラー: {e}")
        st.stop()

    # サイドバーを描画し、入力値を受け取る。
    inputs = renderSidebar(occList, models, macro)
    # 実行ボタンが押されたときだけ入力値を保存する。その後に入力を変えても、ボタンを押すまで結果は変わらない。
    if inputs.isSubmitted:
        st.session_state["sim_params"] = inputs

    # まだ一度も実行していなければ、案内とガイドだけを表示して終わる。
    if "sim_params" not in st.session_state:
        st.info("👈 サイドバーで条件を設定し、「シミュレーション実行」ボタンを押してください。")
        renderPreSimGuides(MODEL_DIR)
        return

    # 最後に実行したときの入力値を取り出す。
    p: SidebarInputs = st.session_state["sim_params"]
    statusQuo, careerChange = _simulate(models, p.modelKey, p, ageCurve)
    modelKeys, sqAll, ccAll, roiAll = _runAllModelSimulations(models, p, ageCurve)

    st.markdown("### 📋 分析結果")
    renderAnalysisResults(
        statusQuo, careerChange,
        p.currentAge, p.currentOcc, p.targetOcc,
        p.currentIncome, p.skillTransfer, p.learningCost,
    )

    st.markdown("### 📈 年収推移グラフ")
    # Plotly のグラフを表示する。use_container_width=True で横幅いっぱいに広げる。
    st.plotly_chart(
        plotMainPlotly(statusQuo, careerChange, p.currentAge, p.currentOcc, p.targetOcc, p.learningCost),
        use_container_width=True, theme="streamlit",
    )

    st.markdown("### 📉 全モデル比較グラフ")
    st.plotly_chart(
        plotAllModelsPlotly(sqAll, ccAll, p.currentAge, [r[0] for r in roiAll], [_MODEL_LABEL_MAP[k] for k in modelKeys]),
        use_container_width=True, theme="streamlit",
    )

    st.markdown("### 📊 年次詳細（選択モデル・5年刻み）")
    # 内包表記で5年ごとの行を作り、表にして表示する。
    st.dataframe(
        pd.DataFrame([
            {
                "年齢": f"{p.currentAge + i}歳",
                "現状維持（万円）": f"{statusQuo[i]:,.0f}",
                "転職後（万円）": f"{careerChange[i]:,.0f}",
                "年間差（万円）": f"{careerChange[i] - statusQuo[i]:+,.0f}",
            }
            for i in range(0, len(statusQuo), TABLE_STEP_YEARS)
        ]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    renderPostSimGuides(
        models, p.modelKey, p.targetOcc,
        p.currentAge, p.currentExp, p.currentIncome,
        AGE_ALL_PATH, MODEL_DIR,
    )

    st.markdown("---")
    # 小さな文字の注記を表示する。f文字列を並べて書くと、1つの文字列につながる。
    st.caption(
        f"📌 使用モデル: {p.modelLabel} ／ "
        f"GDP: {p.gdpGrowth:+.2f}% ／ CPI: {p.futureCpi} ／ "
        f"昇給抑制: {p.raiseSuppression * 100:.0f}% ／ キャリアリスク: {p.careerRisk * 100:.0f}% ／ "
        "本シミュレーションは厚生労働省「賃金構造基本統計調査」・GDP・CPI をもとにした統計的推計です。"
        "個人の実際の収入を保証するものではありません。"
    )


# streamlit run main.py で起動したときも __name__ は "__main__" になるため、main() が実行される。
if __name__ == "__main__":
    main()
