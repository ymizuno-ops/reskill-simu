from __future__ import annotations
import os
import json
import pandas as pd
import streamlit as st
from model_types import ModelDict
from simulation import predict, getOneStepDownIncome, MIN_FIRST_INCOME_RATIO

# 意味: モデル精度カードを横に何枚並べるか。見た目だけが変わる。
MAX_GRID_COLS = 3
PERCENT = 100
# 影響: 長くすると職種名が省略されにくくなるが、表の列名・見出しが長くなる。
MAX_OCC_NAME_IN_HEADER = 12   # 表の列名に入れる職種名の最大文字数
MAX_OCC_NAME_IN_TITLE = 15    # 見出しに入れる職種名の最大文字数

# ── モデル説明（MODEL_DESCRIPTIONS は1箇所のみ定義） ──────────────────────
# 意味: 画面の「各モデルの特徴と使い分け」に出る、モデル名と説明文。
# 影響: 文言は自由に書き換えてよい。注意: 2番目の値（linear など）は色分けの種類で、_TYPE_COLOR にある名前のどれかにする。
MODEL_DESCRIPTIONS: list[tuple[str, str, str]] = [
    (
        "🔵 Ridge Regression", "linear",
        "線形回帰にL2正則化を加えたシンプルモデル。職種・年齢・経験年数の主効果を線形に捉える。"
        "過学習しにくく安定した予測が特徴。標準的なキャリアパスの基準として使いやすい。",
    ),
    (
        "🔵 ElasticNet", "linear",
        "RidgeとLassoを融合した線形モデル。不要な特徴量の係数を自動でゼロに近づける（スパース性）。"
        "解釈性が高く、影響の強い特徴量を絞り込んで学習する。",
    ),
    (
        "🔵 Custom Ridge", "linear",
        "年齢²・年齢×経験年数の交互作用項など独自特徴量を追加した線形モデル。"
        "給与ピーク帯（35〜54歳）のフラグも組み込み、年収カーブの非線形な動きを線形モデルで近似する。",
    ),
    (
        "🟢 Random Forest", "tree",
        "複数の決定木を組み合わせたバギングモデル。職種ごとの細かい条件分岐を学習するが、"
        "外挿（訓練データ範囲外の予測）が苦手。CV精度は低めだが、特定職種の上振れ・下振れシナリオ確認に有効。",
    ),
    (
        "🟢 Gradient Boosting", "tree",
        "弱い決定木を順番に積み上げてエラーを修正する勾配ブースティング。sklearnの標準実装で追加インストール不要。"
        "XGBoostより低速だが安定性が高く、過学習に強い。",
    ),
    (
        "🟡 LightGBM", "boosting",
        "MicrosoftのLightGBMは葉ごとに成長する「Leaf-wise」戦略で高速。"
        "職種名をカテゴリ変数としてネイティブに処理でき、OHEが不要。大規模データでも実用的な速度で学習できる。",
    ),
    (
        "🟡 CatBoost", "boosting",
        "Yandexのカテゴリ特化ブースティング。Target Encodingを対称木とともに最適化する独自手法で職種名を扱う。"
        "ハイパーパラメータのデフォルト値が優秀でチューニング不要でも高精度。",
    ),
    (
        "🟡 XGBoost", "boosting",
        "勾配ブースティングの業界標準。L1/L2正則化・欠損値の自動処理・並列化など多くの最適化を備える。"
        "特徴量エンジニアリング（FE）を組み合わせることでさらに精度が向上。",
    ),
    (
        "🏆 Stacking Ensemble", "ensemble",
        "全ベースモデルのOOF（Out-of-Fold）予測をメタ特徴量としてRidgeで統合する2層アンサンブル。"
        "単一モデルでは捉えられない「モデル間の誤差の相補関係」を学習し、理論上最高精度を目指す。ただし訓練時間は最長。",
    ),
]

# 意味: モデルの種類ごとの表示色。変えると色だけが変わる。
_TYPE_COLOR: dict[str, str] = {
    "linear": "#4F8EF7",
    "tree": "#4CAF50",
    "boosting": "#FF9800",
    "ensemble": "#E040FB",
}
# 意味: モデルの種類ごとに説明の横に出すラベルの文言。
_TYPE_LABEL: dict[str, str] = {
    "linear": "線形系",
    "tree": "ツリー系",
    "boosting": "ブースティング系",
    "ensemble": "アンサンブル",
}

# 意味: 「マクロ経済シナリオガイド」の GDP の表の中身。説明用の目安で、計算には使わない。
_GDP_SCENARIOS = [
    {"期待GDP成長率": "-2.0%", "シナリオの意味": "深刻な不況。賃金水準全体が低下する悲観シナリオ。"},
    {"期待GDP成長率": "0.0%",  "シナリオの意味": "ゼロ成長。経済が停滞し、昇給は年齢・経験カーブのみに依存。"},
    {"期待GDP成長率": "1.0%",  "シナリオの意味": "緩やかな成長。過去10年間の日本経済の平均的な水準（標準）。"},
    {"期待GDP成長率": "2.0%〜", "シナリオの意味": "安定成長。インフレに連動した健全なベースアップが見込める楽観シナリオ。"},
]
# 意味: 同じガイドの CPI の表の中身。説明用の目安で、計算には使わない。
_CPI_SCENARIOS = [
    {"将来のCPI": "90",  "シナリオの意味": "デフレ進行。物価が下落し、名目賃金の上昇が強く抑えられる。"},
    {"将来のCPI": "105", "シナリオの意味": "現状維持。直近の緩やかなインフレ傾向の継続（標準）。"},
    {"将来のCPI": "120", "シナリオの意味": "マイルドインフレ。物価上昇に合わせて賃金への還元が進む。"},
    {"将来のCPI": "140", "シナリオの意味": "高インフレ。急激な物価高騰による名目賃金の上押し圧力。"},
]
# 意味: 「リアリティ補正シナリオガイド」の昇給抑制の表の中身。説明用の目安で、計算には使わない。
_RAISE_SCENARIOS = [
    {"転職後の昇給抑制": "0%",  "シナリオの意味": "抑制なし。AI予測通りの標準的な昇給を享受できる（理想的）。"},
    {"転職後の昇給抑制": "10%", "シナリオの意味": "軽微なビハインド。キャッチアップ期間として序盤の昇給がやや遅れる。"},
    {"転職後の昇給抑制": "30%", "シナリオの意味": "現実的な壁。未経験領域の評価構築に時間がかかり、昇給ペースが3割減速。"},
    {"転職後の昇給抑制": "50%", "シナリオの意味": "厳しい下積み。最初の数年間はベースアップがほぼ期待できない保守的シナリオ。"},
]
# 意味: 同じガイドのキャリアリスクの表の中身。説明用の目安で、計算には使わない。
_RISK_SCENARIOS = [
    {"キャリアリスク係数": "0%",  "シナリオの意味": "リスクなし。生涯にわたり継続的に最新スキルをキャッチアップできる前提。"},
    {"キャリアリスク係数": "10%", "シナリオの意味": "標準的な陳腐化。中堅層以降で技術変化により若干の年収頭打ちが発生。"},
    {"キャリアリスク係数": "20%", "シナリオの意味": "シニア期の停滞。プレイングマネージャーとしての給与限界に直面する。"},
    {"キャリアリスク係数": "30%", "シナリオの意味": "スキルの陳腐化。技術パラダイムシフト等で10年後以降に市場価値が目減りする。"},
]
# 意味: 「スキル引継ぎ率ガイド」の表の行。シミュレーション後は、各引継ぎ率での初年度年収も計算して表示する。
# 影響: 行を増減すると表の行も増減する。
# (引継ぎ率 %, 目安)
_SKILL_TRANSFER_LEVELS: list[tuple[int, str]] = [
    (0,   "完全未経験スタート（1段下の年齢階級相当）"),
    (20,  "前職の汎用スキルが少し評価される"),
    (40,  "ドメイン知識がある程度活かせる"),
    (60,  "近接領域・類似業務からの転職"),
    (80,  "かなりのスキルが転用できる"),
    (100, "即戦力（資格・経験が完全移行）"),
]
_SKILL_TRANSFER_STATIC = [{"引継ぎ率": f"{rate}%", "目安": meaning} for rate, meaning in _SKILL_TRANSFER_LEVELS]


def renderModelAccuracy(modelDir: str, *, isExpanded: bool) -> None:
    with st.expander("🤖 モデル精度情報（クリックで展開）", expanded=isExpanded):
        metaPath = os.path.join(modelDir, "model_meta.json")
        if os.path.exists(metaPath):
            with open(metaPath, encoding="utf-8") as f:
                metaAll = json.load(f)
            items = list(metaAll.items())
            nCols = min(MAX_GRID_COLS, len(items))
            for rowStart in range(0, len(items), nCols):
                chunk = items[rowStart: rowStart + nCols]
                for col, (_, m) in zip(st.columns(len(chunk)), chunk):
                    with col.container(border=True):
                        st.markdown(f"**{m['label']}**")
                        st.metric("R² Score", f"{m['r2_train']}")
                        st.caption(f"CV={m['r2_cv_mean']}±{m['r2_cv_std']} / MAE={m['mae_train']}万円")

        st.markdown("---")
        st.markdown("#### 📖 各モデルの特徴と使い分け")
        for modelName, modelType, desc in MODEL_DESCRIPTIONS:
            clr = _TYPE_COLOR[modelType]
            lbl = _TYPE_LABEL[modelType]
            st.markdown(
                f"<div style='border-left:3px solid {clr};padding:.4rem .8rem;margin:.35rem 0;'>"
                f"<span style='font-weight:700;color:{clr}'>{modelName}</span>"
                f"<span style='font-size:.7rem;background:{clr}22;color:{clr};"
                f"border-radius:4px;padding:1px 6px;margin-left:8px'>{lbl}</span>"
                f"<div style='font-size:.82rem;margin-top:.25rem;opacity:0.9;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def renderSkillTransferStatic(*, isExpanded: bool) -> None:
    with st.expander("📊 スキル引継ぎ率ガイド", expanded=isExpanded):
        st.markdown("転職先でどの程度スキルが評価されるかを設定します。初年度年収の計算に使用されます。")
        st.dataframe(pd.DataFrame(_SKILL_TRANSFER_STATIC), use_container_width=True, hide_index=True)
        st.caption("※ シミュレーション実行後は目標職種・年齢・年収をもとに実際の初年度年収も表示されます。")


def renderSkillTransferTable(
    models: ModelDict,
    modelKey: str,
    targetOcc: str,
    currentAge: int,
    currentExp: float,
    currentIncome: float,
    ageAllPath: str,
) -> None:
    baseIncome, baseLabel = getOneStepDownIncome(targetOcc, currentAge, ageAllPath)
    expIncome = predict(models, modelKey, targetOcc, currentAge, currentExp)

    data = []
    for rate, meaning in _SKILL_TRANSFER_LEVELS:
        first = max(baseIncome + (expIncome - baseIncome) * (rate / PERCENT), baseIncome * MIN_FIRST_INCOME_RATIO)
        diff = first - currentIncome
        diffStr = f"▲ {abs(diff):.0f}万円" if diff < 0 else f"+{diff:.0f}万円"
        data.append({
            "引継ぎ率": f"{rate}%",
            "意味": meaning,
            f"初年度（現職→{targetOcc[:MAX_OCC_NAME_IN_HEADER]}）": f"{first:.0f}万円 ({diffStr})",
        })

    with st.expander(
        f"📊 スキル引継ぎ率ガイド　※ベースライン: {targetOcc[:MAX_OCC_NAME_IN_TITLE]} の {baseLabel} 平均年収 {baseIncome:.0f}万円",
        expanded=False,
    ):
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def renderMacroGuide(*, isExpanded: bool) -> None:
    with st.expander("🌍 マクロ経済シナリオガイド（GDP・CPI）", expanded=isExpanded):
        st.markdown("将来の日本全体の名目賃金上昇率に影響を与えます。")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame(_GDP_SCENARIOS), use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(pd.DataFrame(_CPI_SCENARIOS), use_container_width=True, hide_index=True)


def renderRiskGuide(*, isExpanded: bool) -> None:
    with st.expander("🛡️ リアリティ補正シナリオガイド（昇給抑制・キャリアリスク）", expanded=isExpanded):
        st.markdown("AIモデルが算出した「理想的な予測」に対して、現実的な下方修正を加えます。")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame(_RAISE_SCENARIOS), use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(pd.DataFrame(_RISK_SCENARIOS), use_container_width=True, hide_index=True)


def renderPreSimGuides(modelDir: str) -> None:
    renderModelAccuracy(modelDir, isExpanded=True)
    renderSkillTransferStatic(isExpanded=True)
    renderMacroGuide(isExpanded=True)
    renderRiskGuide(isExpanded=True)


def renderPostSimGuides(
    models: ModelDict,
    modelKey: str,
    targetOcc: str,
    currentAge: int,
    currentExp: float,
    currentIncome: float,
    ageAllPath: str,
    modelDir: str,
) -> None:
    renderModelAccuracy(modelDir, isExpanded=False)
    renderSkillTransferTable(models, modelKey, targetOcc, currentAge, currentExp, currentIncome, ageAllPath)
    renderMacroGuide(isExpanded=False)
    renderRiskGuide(isExpanded=False)
