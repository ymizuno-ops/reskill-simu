from __future__ import annotations
import os
import json
import pandas as pd
import streamlit as st
from simulation import predict, get_one_step_down_income

# ── モデル説明（MODEL_DESCRIPTIONS は1箇所のみ定義） ──────────────────────
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

_TYPE_COLOR: dict[str, str] = {
    "linear": "#4F8EF7",
    "tree": "#4CAF50",
    "boosting": "#FF9800",
    "ensemble": "#E040FB",
}
_TYPE_LABEL: dict[str, str] = {
    "linear": "線形系",
    "tree": "ツリー系",
    "boosting": "ブースティング系",
    "ensemble": "アンサンブル",
}

_GDP_SCENARIOS = [
    {"期待GDP成長率": "-2.0%", "シナリオの意味": "深刻な不況。賃金水準全体が低下する悲観シナリオ。"},
    {"期待GDP成長率": "0.0%",  "シナリオの意味": "ゼロ成長。経済が停滞し、昇給は年齢・経験カーブのみに依存。"},
    {"期待GDP成長率": "1.0%",  "シナリオの意味": "緩やかな成長。過去10年間の日本経済の平均的な水準（標準）。"},
    {"期待GDP成長率": "2.0%〜", "シナリオの意味": "安定成長。インフレに連動した健全なベースアップが見込める楽観シナリオ。"},
]
_CPI_SCENARIOS = [
    {"将来のCPI": "90",  "シナリオの意味": "デフレ進行。物価が下落し、名目賃金の上昇が強く抑えられる。"},
    {"将来のCPI": "105", "シナリオの意味": "現状維持。直近の緩やかなインフレ傾向の継続（標準）。"},
    {"将来のCPI": "120", "シナリオの意味": "マイルドインフレ。物価上昇に合わせて賃金への還元が進む。"},
    {"将来のCPI": "140", "シナリオの意味": "高インフレ。急激な物価高騰による名目賃金の上押し圧力。"},
]
_RAISE_SCENARIOS = [
    {"転職後の昇給抑制": "0%",  "シナリオの意味": "抑制なし。AI予測通りの標準的な昇給を享受できる（理想的）。"},
    {"転職後の昇給抑制": "10%", "シナリオの意味": "軽微なビハインド。キャッチアップ期間として序盤の昇給がやや遅れる。"},
    {"転職後の昇給抑制": "30%", "シナリオの意味": "現実的な壁。未経験領域の評価構築に時間がかかり、昇給ペースが3割減速。"},
    {"転職後の昇給抑制": "50%", "シナリオの意味": "厳しい下積み。最初の数年間はベースアップがほぼ期待できない保守的シナリオ。"},
]
_RISK_SCENARIOS = [
    {"キャリアリスク係数": "0%",  "シナリオの意味": "リスクなし。生涯にわたり継続的に最新スキルをキャッチアップできる前提。"},
    {"キャリアリスク係数": "10%", "シナリオの意味": "標準的な陳腐化。中堅層以降で技術変化により若干の年収頭打ちが発生。"},
    {"キャリアリスク係数": "20%", "シナリオの意味": "シニア期の停滞。プレイングマネージャーとしての給与限界に直面する。"},
    {"キャリアリスク係数": "30%", "シナリオの意味": "スキルの陳腐化。技術パラダイムシフト等で10年後以降に市場価値が目減りする。"},
]
_SKILL_TRANSFER_STATIC = [
    {"引継ぎ率": "0%",   "目安": "完全未経験スタート（1段下の年齢階級相当）"},
    {"引継ぎ率": "20%",  "目安": "前職の汎用スキルが少し評価される"},
    {"引継ぎ率": "40%",  "目安": "ドメイン知識がある程度活かせる"},
    {"引継ぎ率": "60%",  "目安": "近接領域・類似業務からの転職"},
    {"引継ぎ率": "80%",  "目安": "かなりのスキルが転用できる"},
    {"引継ぎ率": "100%", "目安": "即戦力（資格・経験が完全移行）"},
]


def render_model_accuracy(model_dir: str, *, expanded: bool) -> None:
    with st.expander("🤖 モデル精度情報（クリックで展開）", expanded=expanded):
        meta_path = os.path.join(model_dir, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta_all = json.load(f)
            items = list(meta_all.items())
            ncols = min(3, len(items))
            for row_start in range(0, len(items), ncols):
                chunk = items[row_start: row_start + ncols]
                for col, (_, m) in zip(st.columns(len(chunk)), chunk):
                    with col.container(border=True):
                        st.markdown(f"**{m['label']}**")
                        st.metric("R² Score", f"{m['r2_train']}")
                        st.caption(f"CV={m['r2_cv_mean']}±{m['r2_cv_std']} / MAE={m['mae_train']}万円")

        st.markdown("---")
        st.markdown("#### 📖 各モデルの特徴と使い分け")
        for model_name, mtype, desc in MODEL_DESCRIPTIONS:
            clr = _TYPE_COLOR[mtype]
            lbl = _TYPE_LABEL[mtype]
            st.markdown(
                f"<div style='border-left:3px solid {clr};padding:.4rem .8rem;margin:.35rem 0;'>"
                f"<span style='font-weight:700;color:{clr}'>{model_name}</span>"
                f"<span style='font-size:.7rem;background:{clr}22;color:{clr};"
                f"border-radius:4px;padding:1px 6px;margin-left:8px'>{lbl}</span>"
                f"<div style='font-size:.82rem;margin-top:.25rem;opacity:0.9;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_skill_transfer_static(*, expanded: bool) -> None:
    with st.expander("📊 スキル引継ぎ率ガイド", expanded=expanded):
        st.markdown("転職先でどの程度スキルが評価されるかを設定します。初年度年収の計算に使用されます。")
        st.dataframe(pd.DataFrame(_SKILL_TRANSFER_STATIC), use_container_width=True, hide_index=True)
        st.caption("※ シミュレーション実行後は目標職種・年齢・年収をもとに実際の初年度年収も表示されます。")


def render_skill_transfer_table(
    models: dict,
    model_key: str,
    target_occ: str,
    current_age: int,
    current_exp: float,
    current_income: float,
    age_all_path: str,
) -> None:
    base_income, base_label = get_one_step_down_income(target_occ, current_age, age_all_path)
    exp_income = predict(models, model_key, target_occ, current_age, current_exp)

    rates = [0, 20, 40, 60, 80, 100]
    meanings = [row["目安"] for row in _SKILL_TRANSFER_STATIC]

    data = []
    for rate, meaning in zip(rates, meanings):
        first = max(base_income + (exp_income - base_income) * (rate / 100), base_income * 0.8)
        diff = first - current_income
        diff_str = f"▲ {abs(diff):.0f}万円" if diff < 0 else f"+{diff:.0f}万円"
        data.append({
            "引継ぎ率": f"{rate}%",
            "意味": meaning,
            f"初年度（現職→{target_occ[:12]}）": f"{first:.0f}万円 ({diff_str})",
        })

    with st.expander(
        f"📊 スキル引継ぎ率ガイド　※ベースライン: {target_occ[:15]} の {base_label} 平均年収 {base_income:.0f}万円",
        expanded=False,
    ):
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def render_macro_guide(*, expanded: bool) -> None:
    with st.expander("🌍 マクロ経済シナリオガイド（GDP・CPI）", expanded=expanded):
        st.markdown("将来の日本全体の名目賃金上昇率に影響を与えます。")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame(_GDP_SCENARIOS), use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(pd.DataFrame(_CPI_SCENARIOS), use_container_width=True, hide_index=True)


def render_risk_guide(*, expanded: bool) -> None:
    with st.expander("🛡️ リアリティ補正シナリオガイド（昇給抑制・キャリアリスク）", expanded=expanded):
        st.markdown("AIモデルが算出した「理想的な予測」に対して、現実的な下方修正を加えます。")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame(_RAISE_SCENARIOS), use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(pd.DataFrame(_RISK_SCENARIOS), use_container_width=True, hide_index=True)


def render_pre_sim_guides(model_dir: str) -> None:
    render_model_accuracy(model_dir, expanded=True)
    render_skill_transfer_static(expanded=True)
    render_macro_guide(expanded=True)
    render_risk_guide(expanded=True)


def render_post_sim_guides(
    models: dict,
    model_key: str,
    target_occ: str,
    current_age: int,
    current_exp: float,
    current_income: float,
    age_all_path: str,
    model_dir: str,
) -> None:
    render_model_accuracy(model_dir, expanded=False)
    render_skill_transfer_table(models, model_key, target_occ, current_age, current_exp, current_income, age_all_path)
    render_macro_guide(expanded=False)
    render_risk_guide(expanded=False)
