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

# ── 日本語フォント（OS問わず自動検出）────────────────────
def _set_jp_font():
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    # 優先順位: Windows → macOS → Linux → フォールバック
    candidates = [
        "Yu Gothic", "YuGothic", "Meiryo", "MS Gothic", "MS PGothic",  # Windows
        "Hiragino Sans", "Hiragino Kaku Gothic Pro",                     # macOS
        "Noto Sans CJK JP", "IPAexGothic", "IPAGothic",                 # Linux
        "TakaoPGothic", "VL Gothic",
    ]
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            return font
    return None

_jp_font = _set_jp_font()
matplotlib.rcParams["axes.unicode_minus"] = False

# japanize-matplotlib が入っていれば自動適用（フォントが見つからなかった場合）
if _jp_font is None:
    try:
        import japanize_matplotlib  # noqa: F401
    except ImportError:
        pass

# ── パス ──────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(_HERE, "data", "master")
MODEL_DIR  = os.path.join(_HERE, "models")

# ══════════════════════════════════════════════════════
# ページ設定
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="リスキリングによる年収シミュレーター",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── 分析結果ブロック ── */
.result-section {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 1rem 1.3rem 0.8rem;
    margin-bottom: 0.8rem;
}
.result-section h4 {
    font-size: 0.95rem;
    font-weight: 600;
    color: #ccc;
    margin: 0 0 0.5rem;
}
.result-section ul {
    margin: 0;
    padding-left: 1.2rem;
    list-style: disc;
}
.result-section li {
    font-size: 0.88rem;
    color: #ddd;
    margin-bottom: 0.25rem;
    line-height: 1.5;
}
.result-section li span.val {
    font-weight: 700;
    color: #fff;
}
.result-section li span.pos { color: #4CAF50; font-weight: 700; }
.result-section li span.neg { color: #F44336; font-weight: 700; }
/* ── 生涯差額ハイライト ── */
.lifetime-highlight {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.3rem 0 0;
}
/* ── KPIカード ── */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.metric-card .lbl { font-size:.78rem; color:#999; margin-bottom:4px; }
.metric-card .val { font-size:1.45rem; font-weight:700; }
.metric-card .sub { font-size:.72rem; color:#aaa; margin-top:3px; }
.pos { color:#4CAF50 !important; }
.neg { color:#F44336 !important; }
.neu { color:#4F8EF7 !important; }
/* ── セクションヘッダー ── */
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
    pkl = os.path.join(MODEL_DIR, "models.pkl")
    if not os.path.exists(pkl):
        with st.spinner("初回起動: データ加工 & モデル訓練中（1〜2 分）"):
            from step1_to_processed import main as s1
            from step2_to_master    import main as s2
            from step3_train        import main as s3
            s1(); s2(); s3()

    with open(pkl, "rb") as f:
        models = pickle.load(f)

    occ_list  = pd.read_csv(os.path.join(MASTER_DIR, "occupation_list.csv"))
    age_curve = pd.read_csv(os.path.join(MASTER_DIR, "age_curve.csv"))
    with open(os.path.join(MASTER_DIR, "macro_params.json"), encoding="utf-8") as f:
        macro = json.load(f)

    return models, occ_list, age_curve, macro


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
    skill_transfer, nominal_raise, age_curve,
    years=50,
):
    """
    nominal_raise: ユーザーが指定した名目賃金上昇率（GDP成長率+CPI補正済み）
    """
    raise_by_age = dict(zip(age_curve["age_mid"], age_curve["raise_rate"]))

    def age_raise(age):
        nearest = min(raise_by_age, key=lambda a: abs(a - age))
        return raise_by_age[nearest]

    # ── 現状維持 ──
    base_pred  = predict(models, model_key, current_occ, current_age, current_exp)
    correction = current_income / max(base_pred, 1)

    status_quo = []
    income = current_income
    for i in range(years):
        age = current_age + i
        if age >= 65:
            income *= 0.97
        else:
            pred   = predict(models, model_key, current_occ, age, current_exp + i)
            income = pred * correction * (1 + nominal_raise)
        status_quo.append(max(income, 0))

    # ── 転職シナリオ ──
    inexperienced = predict(models, model_key, target_occ, current_age, 0)
    experienced   = predict(models, model_key, target_occ, current_age, current_exp)
    first_income  = inexperienced * (1 - skill_transfer) + experienced * skill_transfer
    first_pred    = predict(models, model_key, target_occ, current_age, 0)
    corr2         = first_income / max(first_pred, 1)

    career_change = []
    for i in range(years):
        age = current_age + i
        if age >= 65:
            career_change.append(max(career_change[-1] * 0.97, 0))
        else:
            pred   = predict(models, model_key, target_occ, age, i)
            income = pred * corr2 * (1 + nominal_raise * i * 0.05)
            career_change.append(max(income, 0))

    return status_quo, career_change


def calc_roi(sq, cc, cost):
    cumulative, breakeven_month = 0, None
    for i, (s, c) in enumerate(zip(sq, cc)):
        annual_diff  = c - s
        cumulative  += annual_diff
        monthly_diff = annual_diff / 12
        if monthly_diff > 0 and breakeven_month is None:
            months_to_break = (-cumulative + annual_diff + cost) / monthly_diff
            if months_to_break <= 12:
                breakeven_month = i * 12 + int(months_to_break)
    lifetime = sum(c - s for s, c in zip(sq, cc))
    return breakeven_month, lifetime


# ══════════════════════════════════════════════════════
# グラフ描画
# ══════════════════════════════════════════════════════
def plot_main(sq, cc, current_age, current_occ, target_occ, cost):
    """メイングラフ: 年収推移 + 累積収支差額（2軸）"""
    ages  = [current_age + i for i in range(len(sq))]
    cumul = np.cumsum([c - s for s, c in zip(sq, cc)])

    fig, ax1 = plt.subplots(figsize=(14, 5), facecolor="#1A1A2E")
    ax2 = ax1.twinx()
    ax1.set_facecolor("#1A1A2E")

    # 年収線
    ax1.plot(ages, sq, color="#4F8EF7", lw=2.5, label="現状", zorder=3)
    ax1.plot(ages, cc, color="#FF5B5B", lw=2.5, label="転職後", zorder=3)

    # 累積差額（緑点線）
    ax2.plot(ages, cumul, color="#4CAF50", lw=1.8, ls="--", label="累積収支差額", zorder=2)
    ax2.fill_between(ages, 0, cumul, where=(np.array(cumul) >= 0),
                     alpha=0.08, color="#4CAF50")
    ax2.fill_between(ages, 0, cumul, where=(np.array(cumul) < 0),
                     alpha=0.08, color="#F44336")
    ax2.axhline(0, color="#555", lw=0.8)

    # 学習コスト線
    if cost > 0:
        ax2.axhline(-cost, color="orange", ls=":", lw=1.2, alpha=0.7)

    ax1.set_title(f"年収推移グラフ ({current_occ} vs {target_occ})",
                  color="white", fontsize=12, pad=10)
    ax1.set_xlabel("年齢", color="#aaa", fontsize=9)
    ax1.set_ylabel("年収（万円）", color="#aaa", fontsize=9)
    ax2.set_ylabel("累積収支差額（万円）", color="#4CAF50", fontsize=9)

    for ax in [ax1, ax2]:
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#444")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):+,}"))
    ax2.tick_params(axis="y", colors="#4CAF50")

    # 凡例統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, labelcolor="white", framealpha=0.2,
               loc="upper left")

    ax1.grid(axis="y", color="#2a2a3e", lw=0.7)
    ax1.set_xlim(current_age, current_age + 50)
    plt.tight_layout()
    return fig


def plot_3models(sq_all, cc_all, current_age, breakevens, model_labels):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="#1A1A2E")
    fig.suptitle("モデル別 年収推移シミュレーション（50年）",
                 color="white", fontsize=13, y=1.01)
    # メイングラフと同じ色系統: 現状=青, 転職後=赤（モデルごとに明度を変える）
    palette_sq = ["#4F8EF7", "#5BA3FF", "#3D7AE8"]   # 青系（現状維持）
    palette_cc = ["#FF5B5B", "#FF7B7B", "#E84040"]   # 赤系（転職後）
    ages = [current_age + i for i in range(len(sq_all[0]))]

    for ax, sq, cc, label, csq, ccc, be in zip(
        axes, sq_all, cc_all, model_labels, palette_sq, palette_cc, breakevens
    ):
        ax.set_facecolor("#1A1A2E")
        ax.plot(ages, sq, color=csq, lw=2, label="現状維持", alpha=0.9)
        ax.fill_between(ages, sq, alpha=0.07, color=csq)
        ax.plot(ages, cc, color=ccc, lw=2, label="転職後", alpha=0.9)
        ax.fill_between(ages, cc, alpha=0.07, color=ccc)
        if be:
            be_age = current_age + be / 12
            be_val = float(np.interp(be_age, ages, cc))
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



# ══════════════════════════════════════════════════════
# 職種大分類マップ（日本標準職業分類に準拠）
# ══════════════════════════════════════════════════════
OCCUPATION_CATEGORIES = {
    "管理職": [
        "管理的職業従事者", "男管理的職業従事者", "女管理的職業従事者",
    ],
    "専門職・技術職（IT・理工系）": [
        "研究者", "システムコンサルタント・設計者", "ソフトウェア作成者",
        "電気・電子・電気通信技術者（通信ネットワーク技術者を除く）",
        "機械技術者", "輸送用機器技術者", "金属技術者", "化学技術者",
        "建築技術者", "土木技術者", "測量技術者", "他に分類されない技術者",
        "その他の情報処理・通信技術者",
    ],
    "専門職・技術職（医療・福祉）": [
        "医師", "歯科医師", "獣医師", "薬剤師", "保健師", "助産師", "看護師",
        "准看護師", "看護助手", "男看護助手", "女看護助手",
        "理学療法士，作業療法士，言語聴覚士，視能訓練士",
        "臨床検査技師", "診療放射線技師", "歯科衛生士", "歯科技工士",
        "栄養士", "介護職員（医療・福祉施設等）", "訪問介護従事者",
        "介護支援専門員（ケアマネージャー）",
        "その他の保健医療従事者", "その他の保健医療サービス職業従事者",
        "その他の社会福祉専門職業従事者",
    ],
    "専門職・技術職（教育・文化）": [
        "大学教授（高専含む）", "大学准教授（高専含む）", "大学講師・助教（高専含む）",
        "高等学校教員", "小・中学校教員", "幼稚園教員，保育教諭",
        "保育士", "個人教師", "その他の教員",
        "著述家，記者，編集者", "美術家，写真家，映像撮影者",
        "音楽家，舞台芸術家", "デザイナー", "宗教家",
        "他に分類されない専門的職業従事者",
    ],
    "専門職・技術職（法務・経営・金融）": [
        "公認会計士，税理士", "法務従事者",
        "その他の経営・金融・保険専門職業従事者",
    ],
    "事務職": [
        "総合事務員", "庶務・人事事務員", "企画事務員", "会計事務従事者",
        "営業・販売事務従事者", "生産関連事務従事者", "外勤事務従事者",
        "運輸・郵便事務従事者", "事務用機器操作員", "秘書",
        "受付・案内事務員", "電話応接事務員",
        "その他の一般事務従事者",
    ],
    "営業・販売職": [
        "販売店員", "販売類似職業従事者",
        "機械器具・通信・システム営業職業従事者（自動車を除く）",
        "自動車営業職業従事者", "保険営業職業従事者",
        "金融営業職業従事者", "その他の営業職業従事者",
        "その他の商品販売従事者",
    ],
    "サービス職": [
        "飲食物調理従事者", "飲食物給仕従事者", "娯楽場等接客員",
        "理容・美容師", "美容サービス・浴場従事者（美容師を除く）",
        "航空機客室乗務員", "身の回り世話従事者",
        "居住施設・ビル等管理人", "ビル・建物清掃員",
        "清掃員（ビル・建物を除く），廃棄物処理従事者",
        "その他のサービス職業従事者",
    ],
    "保安職": [
        "警備員", "その他の保安職業従事者",
    ],
    "農林漁業": [
        "農林漁業従事者",
    ],
    "生産・製造職（機械・金属）": [
        "機械検査従事者", "自動車整備・修理従事者", "自動車組立従事者",
        "はん用・生産用・業務用機械器具組立従事者",
        "はん用・生産用・業務用機械器具・電気機械器具整備・修理従事者",
        "電気機械器具組立従事者", "金属工作機械作業従事者",
        "金属プレス従事者", "金属溶接・溶断従事者", "鉄工，製缶従事者",
        "板金従事者", "鋳物製造・鍛造従事者", "金属彫刻・表面処理従事者",
        "製銑・製鋼・非鉄金属製錬従事者",
        "製品検査従事者（金属製品）", "その他の機械組立従事者",
        "その他の機械整備・修理従事者",
    ],
    "生産・製造職（その他）": [
        "食料品・飲料・たばこ製造従事者", "化学製品製造従事者",
        "ゴム・プラスチック製品製造従事者", "窯業・土石製品製造従事者",
        "木・紙製品製造従事者", "紡織・衣服・繊維製品製造従事者",
        "印刷・製本従事者", "製品検査従事者（金属製品を除く）",
        "製図その他生産関連・生産類似作業従事者", "包装従事者",
        "その他の製品製造・加工処理従事者（金属製品）",
        "その他の製品製造・加工処理従事者（金属製品を除く）",
    ],
    "建設・土木職": [
        "大工", "建設躯体工事従事者", "土木従事者，鉄道線路工事従事者",
        "建設・さく井機械運転従事者", "ダム・トンネル掘削従事者，採掘従事者",
        "配管従事者", "電気工事従事者", "画工，塗装・看板制作従事者",
        "その他の建設従事者",
    ],
    "輸送・機械運転職": [
        "鉄道運転従事者", "バス運転者", "タクシー運転者",
        "乗用自動車運転者（タクシー運転者を除く）",
        "営業用大型貨物自動車運転者",
        "営業用貨物自動車運転者（大型車を除く）",
        "自家用貨物自動車運転者", "航空機操縦士", "車掌",
        "クレーン・ウインチ運転従事者",
        "その他の定置・建設機械運転従事者",
        "その他の自動車運転従事者",
        "他に分類されない輸送従事者",
    ],
    "運搬・清掃・その他": [
        "船内・沿岸荷役従事者", "その他の運搬従事者",
        "クリーニング職，洗張職", "発電員，変電員",
        "他に分類されない運搬・清掃・包装等従事者",
    ],
}

# 逆引き: 職種名 → 大分類
_OCC_TO_CATEGORY = {
    occ: cat
    for cat, occs in OCCUPATION_CATEGORIES.items()
    for occ in occs
}

def get_category(occ: str) -> str:
    return _OCC_TO_CATEGORY.get(occ, "その他")

# ══════════════════════════════════════════════════════
# サイドバー
# ══════════════════════════════════════════════════════
def render_sidebar(occ_list, models, macro):
    occs = sorted(occ_list["occupation"].tolist())

    # ── プロフィール設定 ──
    st.sidebar.markdown("### 👤 プロフィール設定")
    current_age    = st.sidebar.number_input("現在の年齢", 20, 65, 30, step=1)
    current_exp    = st.sidebar.number_input("現在の勤続年数", 0, 40, 5, step=1)
    current_income = st.sidebar.number_input("現在の年収（万円）", 100, 3000, 450, step=10)

    # ── 予測モデル選択 ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 予測モデル選択")
    model_options = {
        "Ridge（安定型）":        "ridge",
        "Random Forest（変動型）": "random_forest",
        "Custom Ridge（高精度）":  "custom",
    }
    model_label = st.sidebar.selectbox("使用するAIモデル", list(model_options.keys()), index=2)
    model_key   = model_options[model_label]

    # ── キャリア選択 ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💼 キャリア選択")

    all_cats = ["（すべて）"] + sorted(OCCUPATION_CATEGORIES.keys())

    # 現職
    cur_cat = st.sidebar.selectbox("現職の大分類", all_cats, index=0, key="cur_cat")
    if cur_cat == "（すべて）":
        cur_occs = occs
    else:
        cur_occs = sorted([o for o in occs if get_category(o) == cur_cat]) or occs
    default_current = cur_occs.index("販売店員") if "販売店員" in cur_occs else 0
    current_occ = st.sidebar.selectbox("現職名", cur_occs, index=default_current, key="cur_occ")

    # 目標職種
    tgt_cat = st.sidebar.selectbox("目標の大分類", all_cats, index=0, key="tgt_cat")
    if tgt_cat == "（すべて）":
        tgt_occs = occs
    else:
        tgt_occs = sorted([o for o in occs if get_category(o) == tgt_cat]) or occs
    default_target = tgt_occs.index("システムコンサルタント・設計者") \
                     if "システムコンサルタント・設計者" in tgt_occs else 0
    target_occ = st.sidebar.selectbox("目標職種名", tgt_occs, index=default_target, key="tgt_occ")

    skill_transfer = st.sidebar.slider(
        "経験引継ぎ率（%）", 0, 100, 20,
        help="0%＝完全未経験、100%＝即戦力。"
    ) / 100

    # ── 投資設定 ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 投資設定")
    learning_cost = st.sidebar.number_input("自己投資費用（万円）", 0, 500, 50, step=5)

    gdp_growth = st.sidebar.slider(
        "期待GDP成長率（%）", -3.0, 3.0,
        float(round(macro["avg_gdp_growth_10yr"] * 100, 2)),
        step=0.05,
        help="将来の名目賃金上昇率の基準として使用されます。"
    )
    future_cpi = st.sidebar.slider(
        "将来のCPI（物価指数）", 80, 150,
        105, step=1,
        help="2020年=100基準。高いほどインフレを想定した推計になります。"
    )

    # GDP成長率とCPIから名目賃金上昇率を推定
    nominal_raise = max(gdp_growth / 100 + (future_cpi - 100) / 100 * 0.3, 0.0)

    return (current_occ, target_occ, current_age, current_exp, current_income,
            skill_transfer, learning_cost, model_key, model_label, nominal_raise,
            gdp_growth, future_cpi)


# ══════════════════════════════════════════════════════
# 分析結果セクション描画
# ══════════════════════════════════════════════════════
def render_analysis_results(
    sq, cc, current_age, current_occ, target_occ,
    current_income, skill_transfer, learning_cost
):
    """画像1のレイアウト: 5ブロックを2カラムで表示"""

    ages_to_65 = max(0, 65 - current_age)

    # ── 各ポイントの値計算 ──
    # 1. 現状
    current_monthly = current_income / 14  # 賞与込みで14か月分想定
    # 2. 転職直後
    first_monthly   = cc[0] / 14
    # 3. 転職5年後
    idx5            = min(5, len(sq) - 1)
    sq5_annual      = sq[idx5]
    cc5_annual      = cc[idx5]
    sq5_monthly     = sq5_annual / 14
    cc5_monthly     = cc5_annual / 14
    # 4. 65歳時点の生涯年収
    sq_lifetime     = sum(sq[:ages_to_65])
    cc_lifetime     = sum(cc[:ages_to_65])
    net_benefit     = cc_lifetime - sq_lifetime - learning_cost
    # 5. 費用対効果
    breakeven_month, _ = calc_roi(sq, cc, learning_cost)
    roi_pct = (net_benefit / max(learning_cost, 1)) * 100 if learning_cost > 0 else float("inf")

    def v(val, fmt=".1f", unit="万円", cls="val"):
        return f"<span class='{cls}'>{val:{fmt}}{unit}</span>"

    def diff_span(val, unit="万円", fmt=".1f"):
        cls = "pos" if val >= 0 else "neg"
        sign = "+" if val >= 0 else ""
        return f"<span class='{cls}'>{sign}{val:{fmt}}{unit}</span>"

    col_l, col_r = st.columns(2)

    # ── 左カラム ──
    with col_l:
        # 1. 現状
        st.markdown(
            f"<div class='result-section'>"
            f"<h4>1. 現状（現在年齢）</h4><ul>"
            f"<li>月収: {v(current_monthly)}</li>"
            f"<li>年収: {v(current_income)}</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )

        # 2. 転職直後
        st.markdown(
            f"<div class='result-section'>"
            f"<h4>2. 転職直後（初年度）</h4><ul>"
            f"<li>月収: {v(first_monthly)}</li>"
            f"<li>年収: {v(cc[0])}</li>"
            f"<li style='color:#888;font-size:.78rem'>※スキル引継ぎ率 {int(skill_transfer*100)}% を適用済み</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )

        # 4. 65歳時点の生涯年収
        net_cls = "pos" if net_benefit >= 0 else "neg"
        st.markdown(
            f"<div class='result-section'>"
            f"<h4>4. 65歳時点での生涯年収</h4><ul>"
            f"<li>転職しなかった場合: {v(sq_lifetime, fmt=',.0f')}</li>"
            f"<li>転職した場合: {v(cc_lifetime, fmt=',.0f')}</li>"
            f"</ul>"
            f"<div style='font-size:.78rem;color:#888;margin-top:.4rem'>生涯差額（投資コスト控除後）</div>"
            f"<div class='lifetime-highlight {net_cls}'>{net_benefit:+,.0f} 万円</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── 右カラム ──
    with col_r:
        # 3. 転職5年後
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

        # 5. 費用対効果
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

    return net_benefit


# ══════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════
def main():
    st.title("📊 リスキリングによる年収シミュレーター")

    try:
        models, occ_list, age_curve, macro = load_assets()
    except Exception as e:
        st.error(f"起動エラー: {e}\n\npython src/step1_to_processed.py → step2 → step3 を順に実行してください。")
        st.stop()

    # ── サイドバー ──
    (current_occ, target_occ, current_age, current_exp, current_income,
     skill_transfer, learning_cost, model_key, model_label, nominal_raise,
     gdp_growth, future_cpi) = render_sidebar(occ_list, models, macro)

    # ── シミュレーション実行ボタン ──
    run = st.button("🚀 シミュレーション実行", type="primary")

    # セッション状態で結果を保持（ボタン押下後も再描画で消えないように）
    if run:
        st.session_state["sim_done"] = True
        st.session_state["sim_params"] = dict(
            current_occ=current_occ, target_occ=target_occ,
            current_age=current_age, current_exp=current_exp,
            current_income=current_income, skill_transfer=skill_transfer,
            learning_cost=learning_cost, model_key=model_key,
            model_label=model_label, nominal_raise=nominal_raise,
        )

    if not st.session_state.get("sim_done"):
        st.info("👈 サイドバーで条件を設定し、「シミュレーション実行」ボタンを押してください。")
        return

    # ── 結果描画 ──
    p = st.session_state["sim_params"]

    # 選択モデルでシミュレーション
    sq, cc = simulate(
        models, p["model_key"],
        p["current_occ"], p["target_occ"],
        p["current_age"], p["current_exp"], p["current_income"],
        p["skill_transfer"], p["nominal_raise"], age_curve,
    )

    # 3モデル全て計算（グラフ比較用）
    MODEL_KEYS   = ["ridge", "random_forest", "custom"]
    MODEL_LABELS = ["Ridge（安定型）", "Random Forest（変動型）", "Custom（高精度）"]
    sq_all, cc_all, roi_all = [], [], []
    for key in MODEL_KEYS:
        s, c = simulate(
            models, key,
            p["current_occ"], p["target_occ"],
            p["current_age"], p["current_exp"], p["current_income"],
            p["skill_transfer"], p["nominal_raise"], age_curve,
        )
        be, benefit = calc_roi(s, c, p["learning_cost"])
        sq_all.append(s)
        cc_all.append(c)
        roi_all.append((be, benefit))

    # ── 分析結果セクション ──
    st.markdown("---")
    st.markdown("### 📋 分析結果")
    render_analysis_results(
        sq, cc,
        p["current_age"], p["current_occ"], p["target_occ"],
        p["current_income"], p["skill_transfer"], p["learning_cost"],
    )

    # ── メイングラフ ──
    st.markdown('<div class="sec-hdr">年収推移グラフ</div>', unsafe_allow_html=True)
    fig_main = plot_main(sq, cc, p["current_age"],
                         p["current_occ"], p["target_occ"], p["learning_cost"])
    st.pyplot(fig_main, use_container_width=True)
    plt.close(fig_main)

    # ── モデル精度情報 ──
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

    # ── 3モデル比較グラフ ──
    st.markdown('<div class="sec-hdr">📉 3モデル比較グラフ</div>', unsafe_allow_html=True)
    fig3 = plot_3models(sq_all, cc_all, p["current_age"],
                        [r[0] for r in roi_all], MODEL_LABELS)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # ── 年次詳細テーブル ──
    st.markdown('<div class="sec-hdr">📊 年次詳細（選択モデル・5年刻み）</div>', unsafe_allow_html=True)
    rows = []
    for i in range(0, 50, 5):
        age  = p["current_age"] + i
        diff = cc[i] - sq[i]
        cum  = sum(cc[j] - sq[j] for j in range(i + 1))
        rows.append({
            "年齢":           f"{age}歳",
            "現状維持（万円）": f"{sq[i]:,.0f}",
            "転職後（万円）":   f"{cc[i]:,.0f}",
            "年間差（万円）":   f"{diff:+,.0f}",
            "累積差（万円）":   f"{cum:+,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── フッター ──
    st.markdown("---")
    st.caption(
        f"📌 使用モデル: {p['model_label']} ／ "
        f"GDP成長率: {gdp_growth:+.2f}% ／ CPI: {future_cpi} ／ "
        "本シミュレーションは厚生労働省「賃金構造基本統計調査」・GDP・CPI をもとにした統計的推計です。"
        "個人の実際の収入を保証するものではありません。"
    )


if __name__ == "__main__":
    main()
