"""
app.py  ─  リスキリング年収シミュレーター
"""

from __future__ import annotations
import os, sys, json, pickle, warnings
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── 日本語フォント ────────────────────────────────────
for font in ["IPAexGothic", "Noto Sans CJK JP", "Hiragino Sans",
             "Yu Gothic", "MS Gothic", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = font
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# ── パス ─────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(_HERE, "data", "master")
MODEL_DIR  = os.path.join(_HERE, "models")

# ══════════════════════════════════════════════════════
# ページ設定
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="リスキリング年収シミュレーター",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.metric-card .lbl  { font-size:.78rem; color:#999; margin-bottom:4px; }
.metric-card .val  { font-size:1.55rem; font-weight:700; }
.metric-card .sub  { font-size:.72rem; color:#aaa; margin-top:3px; }
.pos { color:#4CAF50 !important; }
.neg { color:#F44336 !important; }
.neu { color:#4F8EF7 !important; }
.sec-hdr {
    font-size:1.05rem; font-weight:600;
    border-left:4px solid #4F8EF7;
    padding-left:.6rem;
    margin:1.2rem 0 .7rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# アセット読み込み
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner="モデルを読み込み中...")
def load_assets():
    # モデルが無ければ自動訓練
    pkl = os.path.join(MODEL_DIR, "models.pkl")
    if not os.path.exists(pkl):
        with st.spinner("初回起動: データ加工 & モデル訓練中（1〜2 分）"):
            from step1_to_processed import main as s1
            from step2_to_master    import main as s2
            from step3_train        import main as s3
            s1(); s2(); s3()

    with open(pkl, "rb") as f:
        models = pickle.load(f)

    occ_list   = pd.read_csv(os.path.join(MASTER_DIR, "occupation_list.csv"))
    age_curve  = pd.read_csv(os.path.join(MASTER_DIR, "age_curve.csv"))
    exp_curve  = pd.read_csv(os.path.join(MASTER_DIR, "exp_curve.csv"))
    with open(os.path.join(MASTER_DIR, "macro_params.json"), encoding="utf-8") as f:
        macro = json.load(f)

    return models, occ_list, age_curve, exp_curve, macro


# ══════════════════════════════════════════════════════
# 予測・シミュレーション
# ══════════════════════════════════════════════════════
def predict(models: dict, model_key: str,
            occupation: str, age: float, experience: float) -> float:
    m = models[model_key]
    X = pd.DataFrame([{"occupation": occupation, "age": age, "experience_years": experience}])
    if model_key == "custom":
        X["age_sq"]         = X["age"] ** 2 / 1000
        X["age_x_exp"]      = X["age"] * X["experience_years"] / 100
        X["exp_ratio"]      = X["experience_years"] / X["age"].clip(lower=1)
        X["prime_age_flag"] = ((X["age"] >= 35) & (X["age"] <= 54)).astype(float)
    return float(m["pipeline"].predict(X)[0])


def simulate(
    models, model_key,
    current_occ, target_occ,
    current_age, current_exp, current_income,
    skill_transfer, macro, age_curve,
    years=50,
):
    """
    50 年間の年収推移を返す。

    現状維持シナリオ:
      - AI 予測値に「実績年収との乖離補正」を適用
      - マクロ賃金上昇率を毎年加算
    転職シナリオ:
      - スキル引継ぎ率で未経験/経験者の初年度年収を線形補間（Step1 記載ロジック）
      - 2 年目以降は転職先職種の昇給カーブに乗る
    """
    nominal_raise = macro["forecast_nominal_raise"]
    raise_by_age  = dict(zip(age_curve["age_mid"], age_curve["raise_rate"]))

    def age_raise(age):
        nearest = min(raise_by_age, key=lambda a: abs(a - age))
        return raise_by_age[nearest]

    # ── 現状維持 ──
    base_pred   = predict(models, model_key, current_occ, current_age, current_exp)
    correction  = current_income / max(base_pred, 1)

    status_quo = []
    income = current_income
    for i in range(years):
        age = current_age + i
        if age >= 65:
            income *= 0.97          # 定年後は緩やかに逓減
        else:
            pred    = predict(models, model_key, current_occ, age, current_exp + i)
            income  = pred * correction
            income *= (1 + nominal_raise)
        status_quo.append(max(income, 0))

    # ── 転職シナリオ ──
    # Step1: 未経験者の初年度（経験0年扱い）
    inexperienced = predict(models, model_key, target_occ, current_age, 0)
    # Step2: 同年齢・同勤続の場合
    experienced   = predict(models, model_key, target_occ, current_age, current_exp)
    # Step3: スキル引継ぎ率で線形補間
    first_income  = inexperienced * (1 - skill_transfer) + experienced * skill_transfer

    # 補正比率（初年度ベース）
    first_pred    = predict(models, model_key, target_occ, current_age, 0)
    corr2         = first_income / max(first_pred, 1)

    career_change = []
    for i in range(years):
        age    = current_age + i
        new_exp = i
        if age >= 65:
            career_change.append(max(career_change[-1] * 0.97, 0))
        else:
            pred   = predict(models, model_key, target_occ, age, new_exp)
            income = pred * corr2 * (1 + nominal_raise * i * 0.05)
            career_change.append(max(income, 0))

    return status_quo, career_change


def calc_roi(sq, cc, cost):
    """回収月数・生涯差益を計算"""
    cumulative, breakeven_month = 0, None
    for i, (s, c) in enumerate(zip(sq, cc)):
        annual_diff   = c - s
        cumulative   += annual_diff
        monthly_diff  = annual_diff / 12
        if monthly_diff > 0 and breakeven_month is None:
            months_to_break = (-cumulative + annual_diff + cost) / monthly_diff
            if months_to_break <= 12:
                breakeven_month = i * 12 + int(months_to_break)
    lifetime = sum(c - s for s, c in zip(sq, cc))
    return breakeven_month, lifetime


# ══════════════════════════════════════════════════════
# グラフ描画
# ══════════════════════════════════════════════════════
def plot_comparison(sq_all, cc_all, current_age, breakevens, model_labels):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="#1A1A2E")
    fig.suptitle("モデル別 年収推移シミュレーション（50 年）",
                 color="white", fontsize=13, y=1.01)

    palette_sq = ["#5B8CFF", "#63D297", "#FF8C61"]
    palette_cc = ["#FF5B5B", "#C45BFF", "#FFD15B"]
    ages = [current_age + i for i in range(len(sq_all[0]))]

    for ax, sq, cc, label, csq, ccc, be in zip(
        axes, sq_all, cc_all, model_labels, palette_sq, palette_cc, breakevens
    ):
        ax.set_facecolor("#1A1A2E")
        ax.plot(ages, sq, color=csq, lw=2,   label="現状維持", alpha=0.9)
        ax.fill_between(ages, sq, alpha=0.08, color=csq)
        ax.plot(ages, cc, color=ccc, lw=2,   label="転職後",   alpha=0.9)
        ax.fill_between(ages, cc, alpha=0.08, color=ccc)

        if be:
            be_age  = current_age + be / 12
            be_val  = float(np.interp(be_age, ages, cc))
            ax.axvline(be_age, color="yellow", ls="--", lw=1.2, alpha=0.7)
            ax.annotate(f"回収\n{be//12}年{be%12}か月",
                        xy=(be_age, be_val),
                        xytext=(be_age + 1.5, be_val * 1.06),
                        color="yellow", fontsize=7.5,
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=0.8))

        ax.set_title(label, color="white", fontsize=9.5, pad=6)
        ax.set_xlabel("年齢", color="#aaa", fontsize=8)
        ax.set_ylabel("年収（万円）", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=7, labelcolor="white", framealpha=0.2)
        ax.grid(axis="y", color="#333", lw=0.5)
        ax.set_xlim(current_age, current_age + 50)

    plt.tight_layout()
    return fig


def plot_cumulative(sq, cc, current_age, cost):
    ages  = [current_age + i for i in range(len(sq))]
    cumul = np.cumsum([c - s for s, c in zip(sq, cc)])

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    ax.fill_between(ages, np.where(cumul >= 0, cumul, 0),  alpha=0.25, color="#4CAF50", label="転職が有利")
    ax.fill_between(ages, np.where(cumul <  0, cumul, 0),  alpha=0.25, color="#F44336", label="転職が不利")
    ax.plot(ages, cumul, color="white", lw=2)
    ax.axhline(0, color="#666", lw=1)
    if cost > 0:
        ax.axhline(-cost, color="orange", ls="--", lw=1.2, alpha=0.8)
        ax.annotate(f"学習コスト ▲{cost:,}万円",
                    xy=(current_age + 1, -cost), color="orange", fontsize=8)

    ax.set_xlabel("年齢", color="#aaa")
    ax.set_ylabel("累積差益（万円）", color="#aaa")
    ax.set_title("累積収益差（転職後 − 現状維持）", color="white", fontsize=11)
    ax.tick_params(colors="#aaa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):+,}"))
    ax.legend(fontsize=8, labelcolor="white", framealpha=0.2)
    ax.grid(axis="y", color="#333", lw=0.5)
    ax.set_xlim(current_age, current_age + 50)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════
def render_sidebar(occ_list):
    st.sidebar.header("⚙️ シミュレーション設定")
    occs = sorted(occ_list["occupation"].tolist())

    st.sidebar.markdown("### 👤 現在の状況")
    current_occ    = st.sidebar.selectbox("現在の職種", occs, index=occs.index("販売店員") if "販売店員" in occs else 0)
    current_age    = st.sidebar.slider("現在の年齢", 20, 60, 30)
    current_exp    = st.sidebar.slider("現職の勤続年数", 0, 30, 3)
    current_income = st.sidebar.number_input("昨年の実績年収（万円）", 100, 3000, 400, step=10)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 転職先")
    target_occ = st.sidebar.selectbox("転職先の職種", occs,
                                       index=occs.index("システムコンサルタント・設計者")
                                             if "システムコンサルタント・設計者" in occs else min(1, len(occs)-1))
    skill_transfer = st.sidebar.slider(
        "スキル引継ぎ率（%）", 0, 100, 20,
        help="0%＝完全未経験、100%＝即戦力。前職の経験がどれだけ評価されるか。"
    ) / 100

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💸 投資コスト")
    learning_cost = st.sidebar.number_input("学習・リスキリング費用（万円）", 0, 500, 50, step=5)

    return current_occ, target_occ, current_age, current_exp, current_income, skill_transfer, learning_cost


def metric_card(label, value, sub="", cls="neu"):
    return (f"<div class='metric-card'>"
            f"<div class='lbl'>{label}</div>"
            f"<div class='val {cls}'>{value}</div>"
            f"<div class='sub'>{sub}</div>"
            f"</div>")


def main():
    st.title("📊 リスキリング年収シミュレーター")
    st.caption("賃金構造基本統計調査 × 機械学習で「私の年齢・私の実績」に基づいた50年収支を可視化")

    try:
        models, occ_list, age_curve, exp_curve, macro = load_assets()
    except Exception as e:
        st.error(f"起動エラー: {e}\n\npython src/step1_to_processed.py → step2 → step3 を順に実行してください。")
        st.stop()

    # ── サイドバー ──
    current_occ, target_occ, current_age, current_exp, current_income, skill_transfer, learning_cost = \
        render_sidebar(occ_list)

    # ── モデル精度 ──
    with st.expander("🤖 モデル精度情報（クリックで展開）"):
        meta_path = os.path.join(MODEL_DIR, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta_all = json.load(f)
            cols = st.columns(3)
            for col, (key, m) in zip(cols, meta_all.items()):
                best = max(v["r2_train"] for v in meta_all.values())
                clr  = "#4F8EF7" if m["r2_train"] == best else "#888"
                col.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='lbl'>{m['label']}</div>"
                    f"<div class='val' style='color:{clr}'>R²={m['r2_train']}</div>"
                    f"<div class='sub'>CV={m['r2_cv_mean']}±{m['r2_cv_std']} / MAE={m['mae_train']}万円</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── シミュレーション実行 ──
    MODEL_KEYS   = ["ridge", "random_forest", "custom"]
    MODEL_LABELS = ["Ridge Regression\n（安定型）", "Random Forest\n（変動型）", "Custom Ridge\n（特徴量強化型）"]

    sq_all, cc_all, roi_all = [], [], []
    for key in MODEL_KEYS:
        sq, cc = simulate(
            models, key,
            current_occ, target_occ,
            current_age, current_exp, current_income,
            skill_transfer, macro, age_curve,
        )
        be, benefit = calc_roi(sq, cc, learning_cost)
        sq_all.append(sq)
        cc_all.append(cc)
        roi_all.append((be, benefit))

    sq0, cc0    = sq_all[0], cc_all[0]
    be0, ben0   = roi_all[0]
    delta_y1    = cc0[0] - sq0[0]
    delta_pct   = delta_y1 / max(sq0[0], 1) * 100

    # ── KPI カード ──
    st.markdown('<div class="sec-hdr">📈 シミュレーション結果（Ridge モデル基準）</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(metric_card(
        "転職直後の年収変化",
        f"{delta_y1:+.0f}万円",
        f"({delta_pct:+.1f}%)",
        "pos" if delta_y1 >= 0 else "neg",
    ), unsafe_allow_html=True)

    if be0:
        c2.markdown(metric_card(
            "学習コスト回収期間",
            f"{be0//12}年{be0%12}か月",
            f"費用 {learning_cost:,}万円",
            "pos" if be0 < 36 else "neu",
        ), unsafe_allow_html=True)
    else:
        c2.markdown(metric_card("学習コスト回収期間", "回収困難", "転職後も収入低下傾向", "neg"),
                    unsafe_allow_html=True)

    c3.markdown(metric_card(
        "生涯収入差（50年累積）",
        f"{ben0:+,.0f}万円",
        "転職後 − 現状維持",
        "pos" if ben0 >= 0 else "neg",
    ), unsafe_allow_html=True)

    c4.markdown(metric_card(
        "スキル引継ぎ率",
        f"{int(skill_transfer*100)}%",
        "0%=未経験 / 100%=即戦力",
        "neu",
    ), unsafe_allow_html=True)

    # ── グラフ ──
    st.markdown('<div class="sec-hdr">📉 3モデル別 年収推移グラフ</div>', unsafe_allow_html=True)
    fig1 = plot_comparison(sq_all, cc_all, current_age,
                           [r[0] for r in roi_all], MODEL_LABELS)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.markdown('<div class="sec-hdr">📊 累積収益差（Ridgeモデル）</div>', unsafe_allow_html=True)
    fig2 = plot_cumulative(sq0, cc0, current_age, learning_cost)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ── 年次詳細テーブル ──
    st.markdown('<div class="sec-hdr">📋 年次詳細（Ridgeモデル・5年刻み）</div>', unsafe_allow_html=True)
    rows = []
    for i in range(0, 50, 5):
        age   = current_age + i
        diff  = cc0[i] - sq0[i]
        cum   = sum(cc0[j] - sq0[j] for j in range(i + 1))
        rows.append({
            "年齢": f"{age}歳",
            "現状維持（万円）": f"{sq0[i]:,.0f}",
            "転職後（万円）":   f"{cc0[i]:,.0f}",
            "年間差（万円）":   f"{diff:+,.0f}",
            "累積差（万円）":   f"{cum:+,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── フッター ──
    st.markdown("---")
    st.caption(
        "📌 本シミュレーションは厚生労働省「賃金構造基本統計調査」・GDP・CPI をもとにした統計的推計です。"
        "個人の実際の収入を保証するものではありません。"
    )


if __name__ == "__main__":
    main()
