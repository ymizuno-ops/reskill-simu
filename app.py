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

# pickle で保存したモデルの復元に必要なクラスを事前インポート
# （step3_train.py のモジュールレベルクラスを app.py 側でも認識させる）
try:
    from step3_train import LGBMWrapper, CatBoostWrapper, StackingEnsemble  # noqa: F401
except ImportError:
    pass  # ライブラリ未インストール時はスキップ

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
_HERE        = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR   = os.path.join(_HERE, "data", "master")
MODEL_DIR    = os.path.join(_HERE, "models")
AGE_ALL_PATH = os.path.join(_HERE, "data", "processed", "age_wage_all.csv")

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
def _add_features(X: pd.DataFrame) -> pd.DataFrame:
    """特徴量エンジニアリング（Custom Ridge・XGBoost で使用）"""
    Xc = X.copy()
    Xc["age_sq"]         = Xc["age"] ** 2 / 1000
    Xc["age_x_exp"]      = Xc["age"] * Xc["experience_years"] / 100
    Xc["exp_ratio"]      = Xc["experience_years"] / Xc["age"].clip(lower=1)
    Xc["prime_age_flag"] = ((Xc["age"] >= 35) & (Xc["age"] <= 54)).astype(float)
    return Xc


# 特徴量エンジニアリングが必要なモデルキーのセット
_FE_MODELS = {"custom", "xgboost", "elasticnet", "gradient_boosting"}


def predict(models: dict, model_key: str,
            occupation: str, age: float, experience: float) -> float:
    """
    モデルの種類に応じて前処理を適用して予測する。
    - sklearn Pipeline系      : X をそのまま渡す（パイプライン内で処理）
    - LightGBM/CatBoost系     : Wrapper クラスがエンコーディングを内包
    - Custom/XGBoost/ElasticNet/GradientBoosting : FEを追加してから渡す
    - StackingEnsemble        : FE処理は内部で実施、X のまま渡す
    """
    m = models[model_key]
    X = pd.DataFrame([{
        "occupation":       occupation,
        "age":              float(age),
        "experience_years": float(experience),
    }])
    # StackingEnsemble は内部でFEを処理するため追加不要
    if model_key != "stacking" and model_key in _FE_MODELS:
        X = _add_features(X)
    return float(m["pipeline"].predict(X)[0])


# 年齢階級のmid値リスト（賃金構造基本統計調査の区分に対応）
_AGE_MIDS = [18.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 67.0]
_AGE_LABELS = ["〜19歳", "20〜24歳", "25〜29歳", "30〜34歳", "35〜39歳",
               "40〜44歳", "45〜49歳", "50〜54歳", "55〜59歳", "60〜64歳", "65〜69歳"]


def _get_one_step_down_income(occ_name: str, current_age: float,
                               age_all_path: str, year: int = 2024) -> tuple[float, str]:
    """
    現在の年齢より「1段階下の年齢階級」の転職先職種の平均年収を返す。
    これが「スキル引継ぎ率0%（純粋な未経験）」の基準年収となる。

    例: 30歳転職 → 25〜29歳階級（mid=27）の年収がスタートライン
        45歳転職 → 40〜44歳階級（mid=42）の年収がスタートライン

    Returns: (基準年収, ラベル文字列)
    """
    try:
        # 現在の年齢が属する階級のインデックスを特定
        current_mid = min(_AGE_MIDS, key=lambda m: abs(m - current_age))
        idx = _AGE_MIDS.index(current_mid)
        # 1段階下（最低でも0インデックス）
        lower_idx   = max(0, idx - 1)
        lower_mid   = _AGE_MIDS[lower_idx]
        lower_label = _AGE_LABELS[lower_idx]

        age_all = pd.read_csv(age_all_path)
        latest  = age_all[(age_all["year"] == year) & (age_all["occupation"] == occ_name)]
        row     = latest[latest["age_mid"] == lower_mid]
        if len(row) > 0:
            return float(row["annual_income"].mean()), lower_label

        # 職種データがなければ全職種の同階級平均
        all_row = age_all[(age_all["year"] == year) & (age_all["age_mid"] == lower_mid)]
        if len(all_row) > 0:
            return float(all_row["annual_income"].mean()), lower_label

    except Exception:
        pass

    # 最終フォールバック
    return 300.0, "〜"


def simulate(
    models, model_key,
    current_occ, target_occ,
    current_age, current_exp, current_income,
    skill_transfer, nominal_raise, age_curve,
    years=50,
    age_all_path: str = "",
    # ── リアリティ補正パラメータ ──
    raise_suppression: float = 0.0,   # 転職後昇給抑制率（0〜0.5）
    career_risk: float = 0.0,         # キャリアリスク係数（0〜0.3）
):
    """
    転職初年度年収の設計:
    ─────────────────────────────────────────
    ベース    = 転職先職種の「現年齢より1段階下の年齢階級」の平均年収
              → 30歳転職なら25〜29歳ベース、45歳転職なら40〜44歳ベース
              → どの年齢でも年齢相応の下方修正が自動でかかる

    上乗せ    = (同職種・現年齢・現勤続の経験者年収 − 1段下ベース) × スキル引継ぎ率
              → 前職の汎用スキル・ドメイン知識が評価される分を加算

    つまり:
      skill_transfer=0%   → 1段下ベース（完全未経験スタート）
      skill_transfer=50%  → ベース + 経験者との差の半分
      skill_transfer=100% → 同年齢・同勤続の経験者と同等（即戦力）

    raise_suppression: 転職後の序盤（〜10年）の昇給を抑制
    career_risk      : 10年以降から蓄積する技術陳腐化・頭打ちリスク
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
    # Step1: 現年齢より1段階下の年齢階級の転職先職種年収をベースとする
    base_income, base_label = _get_one_step_down_income(
        target_occ, current_age, age_all_path
    )

    # Step2: 同職種・同年齢・同勤続年数の経験者年収（即戦力上限）
    experienced_income = predict(models, model_key, target_occ, current_age, current_exp)

    # Step3: スキル引継ぎ率で「1段下ベース〜即戦力」を線形補間
    #   0%   → 1段下の年齢階級年収（完全未経験スタート）
    #   100% → 同年齢・同勤続の経験者と同等（即戦力）
    first_income = base_income + (experienced_income - base_income) * skill_transfer
    first_income = max(first_income, base_income * 0.8)   # 下限: ベースの80%

    # Step4: 昇給カーブの補正比率（ベース年収から昇給カーブを適用）
    # 1段下の年齢階級midをMLモデルの基準点として使用
    current_mid = min(_AGE_MIDS, key=lambda m: abs(m - current_age))
    idx = _AGE_MIDS.index(current_mid)
    lower_mid = _AGE_MIDS[max(0, idx - 1)]
    base_pred_at_lower = predict(models, model_key, target_occ, lower_mid, 0)
    corr2 = first_income / max(base_pred_at_lower, 1)

    career_change = []
    for i in range(years):
        age = current_age + i
        if age >= 65:
            career_change.append(max(career_change[-1] * 0.97, 0))
        else:
            # 転職後の経験年数ベースで予測（22歳スタートと同じ昇給カーブを辿る）
            effective_exp = i   # 転職後の実務経験年数
            pred = predict(models, model_key, target_occ, age, effective_exp)

            # 昇給抑制: 転職後序盤（〜10年）はモデルより昇給が遅い
            suppression_factor = 1.0 - raise_suppression * max(0, (10 - i) / 10)

            # キャリアリスク: 10年以降から蓄積
            risk_decay = 1.0 - career_risk * max(0, (i - 10) / 40)

            income = pred * corr2 * suppression_factor * risk_decay * (1 + nominal_raise * i * 0.05)
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


def plot_all_models(sq_all, cc_all, current_age, breakevens, model_labels):
    """モデル数に応じて動的にサブプロット列数を決定"""
    n     = len(sq_all)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig_w = ncols * 6
    fig_h = 5 * nrows

    fig, axes_raw = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), facecolor="#1A1A2E")
    fig.suptitle("モデル別 年収推移シミュレーション（50年）",
                 color="white", fontsize=13, y=1.01 if nrows == 1 else 1.0)

    # axes を常にフラットリストとして扱う
    if n == 1:
        axes = [axes_raw]
    elif nrows == 1:
        axes = list(axes_raw)
    else:
        axes = [ax for row in axes_raw for ax in row]

    # 余ったサブプロットを非表示
    for ax in axes[n:]:
        ax.set_visible(False)

    palette_sq = ["#4F8EF7","#5BA3FF","#3D7AE8","#4FC3F7","#29B6F6","#0288D1"]
    palette_cc = ["#FF5B5B","#FF7B7B","#E84040","#FF8A65","#FF7043","#E64A19"]
    ages = [current_age + i for i in range(len(sq_all[0]))]

    for ax, sq, cc, label, csq, ccc, be in zip(
        axes[:n], sq_all, cc_all, model_labels, palette_sq, palette_cc, breakevens
    ):
        ax.set_facecolor("#1A1A2E")
        ax.plot(ages, sq, color=csq, lw=2, label="現状維持", alpha=0.9)
        ax.fill_between(ages, sq, alpha=0.07, color=csq)
        ax.plot(ages, cc, color=ccc, lw=2, label="転職後",   alpha=0.9)
        ax.fill_between(ages, cc, alpha=0.07, color=ccc)
        if be:
            be_age = current_age + be / 12
            be_val = float(np.interp(be_age, ages, cc))
            ax.axvline(be_age, color="yellow", ls="--", lw=1.2, alpha=0.7)
            ax.annotate(f"回収\n{be//12}年{be%12}か月",
                        xy=(be_age, be_val), xytext=(be_age + 1.5, be_val * 1.06),
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
    # 利用可能モデルを動的に構築（pkl に存在するモデルのみ表示）
    _all_model_options = {
        "Ridge（安定型）":                      "ridge",
        "ElasticNet（L1+L2正則化）":            "elasticnet",
        "Custom Ridge（特徴量強化型）":         "custom",
        "Random Forest（変動型）":              "random_forest",
        "Gradient Boosting（sklearn標準）":     "gradient_boosting",
        "LightGBM（高速ブースティング）":       "lightgbm",
        "CatBoost（カテゴリ変数特化）":         "catboost",
        "XGBoost（勾配ブースティング）":        "xgboost",
        "Stacking Ensemble（全モデル統合）":   "stacking",
    }
    # pkl に含まれているモデルのみ選択肢に出す
    available_models = {
        label: key
        for label, key in _all_model_options.items()
        if key in models
    }
    default_model = "Custom Ridge（特徴量強化型）" if "Custom Ridge（特徴量強化型）" in available_models else list(available_models.keys())[-1]
    model_label = st.sidebar.selectbox("使用するAIモデル", list(available_models.keys()),
                                        index=list(available_models.keys()).index(default_model))
    model_key   = available_models[model_label]

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

    # ── リアリティ補正 ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 リアリティ補正")
    st.sidebar.caption("統計平均に対して、より現実的な下方修正を加えます。")

    raise_suppression = st.sidebar.slider(
        "転職後の昇給抑制（%）", 0, 50, 0, step=5,
        help="0%＝モデル通り / 30%＝昇給が30%遅くなる / 50%＝かなり保守的"
    ) / 100

    career_risk = st.sidebar.slider(
        "キャリアリスク係数（%）", 0, 30, 0, step=5,
        help="0%＝リスクなし / 15%＝現実的 / 30%＝厳しめ（10年後から徐々に効いてきます）"
    ) / 100

    return (current_occ, target_occ, current_age, current_exp, current_income,
            skill_transfer, learning_cost, model_key, model_label, nominal_raise,
            gdp_growth, future_cpi, raise_suppression, career_risk)


# ══════════════════════════════════════════════════════
# 引継ぎ率説明表
# ══════════════════════════════════════════════════════
def _render_skill_transfer_table(
    models, model_key, target_occ, current_age, current_exp,
    current_income, age_all_path
):
    """
    スキル引継ぎ率ごとの初年度年収を表で表示する。
    ユーザーが引継ぎ率を直感的に選べるよう補助する。
    """
    base_income, base_label = _get_one_step_down_income(
        target_occ, current_age, age_all_path
    )
    exp_income = predict(models, model_key, target_occ, current_age, current_exp)

    rates   = [0, 20, 40, 60, 80, 100]
    meanings = [
        "完全未経験スタート（1段下の年齢階級相当）",
        "前職の汎用スキルが少し評価される",
        "ドメイン知識がある程度活かせる",
        "近接領域・類似業務からの転職",
        "かなりのスキルが転用できる",
        "即戦力（資格・経験が完全移行）",
    ]

    rows = []
    for rate, meaning in zip(rates, meanings):
        first = base_income + (exp_income - base_income) * (rate / 100)
        first = max(first, base_income * 0.8)
        diff  = first - current_income
        diff_str = f"+{diff:.0f}万円" if diff >= 0 else f"{diff:.0f}万円"
        diff_color = "color:#4CAF50" if diff >= 0 else "color:#F44336"
        rows.append((rate, meaning, first, diff_str, diff_color))

    # テーブルHTML生成
    header = (
        "<table style='width:100%;border-collapse:collapse;font-size:.85rem'>"
        "<thead><tr style='border-bottom:1px solid #444;color:#aaa'>"
        "<th style='padding:.4rem .6rem;text-align:left'>引継ぎ率</th>"
        "<th style='padding:.4rem .6rem;text-align:left'>意味</th>"
        f"<th style='padding:.4rem .6rem;text-align:right'>初年度（現職→{target_occ[:12]}）</th>"
        "</tr></thead><tbody>"
    )
    body = ""
    for rate, meaning, first, diff_str, diff_color in rows:
        body += (
            "<tr style='border-bottom:1px solid #2a2a3e'>"
            f"<td style='padding:.4rem .6rem;font-weight:700;color:#4F8EF7'>{rate}%</td>"
            f"<td style='padding:.4rem .6rem;color:#ccc'>{meaning}</td>"
            f"<td style='padding:.4rem .6rem;text-align:right'>"
            f"<span style='color:#fff;font-weight:600'>{first:.0f}万円</span> "
            f"<span style='{diff_color};font-size:.8rem'>（{diff_str}）</span>"
            f"</td></tr>"
        )
    footer = "</tbody></table>"

    with st.expander(
        f"📊 スキル引継ぎ率ガイド　※ベースライン: {target_occ[:15]} の {base_label} 平均年収 {base_income:.0f}万円",
        expanded=False,
    ):
        st.markdown(header + body + footer, unsafe_allow_html=True)


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
     gdp_growth, future_cpi,
     raise_suppression, career_risk) = render_sidebar(occ_list, models, macro)

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
            raise_suppression=raise_suppression,
            career_risk=career_risk,
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
        age_all_path=AGE_ALL_PATH,
        raise_suppression=p.get("raise_suppression", 0.0),
        career_risk=p.get("career_risk", 0.0),
    )

    # 全モデル計算（グラフ比較用）
    MODEL_KEYS   = [k for k in ["ridge", "elasticnet", "custom",
                                 "random_forest", "gradient_boosting",
                                 "lightgbm", "catboost", "xgboost",
                                 "stacking"] if k in models]
    MODEL_LABELS_MAP = {
        "ridge":             "Ridge",
        "elasticnet":        "ElasticNet",
        "custom":            "Custom Ridge",
        "random_forest":     "Random Forest",
        "gradient_boosting": "GradientBoosting",
        "lightgbm":          "LightGBM",
        "catboost":          "CatBoost",
        "xgboost":           "XGBoost",
        "stacking":          "Stacking",
    }
    MODEL_LABELS = [MODEL_LABELS_MAP[k] for k in MODEL_KEYS]
    sq_all, cc_all, roi_all = [], [], []
    for key in MODEL_KEYS:
        s, c = simulate(
            models, key,
            p["current_occ"], p["target_occ"],
            p["current_age"], p["current_exp"], p["current_income"],
            p["skill_transfer"], p["nominal_raise"], age_curve,
            age_all_path=AGE_ALL_PATH,
            raise_suppression=p.get("raise_suppression", 0.0),
            career_risk=p.get("career_risk", 0.0),
        )
        be, benefit = calc_roi(s, c, p["learning_cost"])
        sq_all.append(s)
        cc_all.append(c)
        roi_all.append((be, benefit))

    # ── 分析結果セクション ──
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
            # モデル数に応じて列数を調整（最大3列×複数行）
            n = len(meta_all)
            ncols = min(3, n)
            items = list(meta_all.items())
            best  = max(v["r2_train"] for v in meta_all.values())
            for row_start in range(0, n, ncols):
                row_items = items[row_start:row_start + ncols]
                cols = st.columns(len(row_items))
                for col, (key, m) in zip(cols, row_items):
                    clr = "#4F8EF7" if m["r2_train"] == best else "#888"
                    col.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='lbl'>{m['label']}</div>"
                        f"<div class='val' style='color:{clr}'>R²={m['r2_train']}</div>"
                        f"<div class='sub'>CV={m['r2_cv_mean']}±{m['r2_cv_std']} / MAE={m['mae_train']}万円</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── 各モデルの特徴説明 ──
        st.markdown("---")
        st.markdown("#### 📖 各モデルの特徴と使い分け")
        MODEL_DESCRIPTIONS = [
            ("🔵 Ridge Regression",      "linear",   "線形回帰にL2正則化を加えたシンプルモデル。職種・年齢・経験年数の主効果を線形に捉える。過学習しにくく安定した予測が特徴。標準的なキャリアパスの基準として使いやすい。"),
            ("🔵 ElasticNet",            "linear",   "RidgeとLassoを融合した線形モデル。不要な特徴量の係数を自動でゼロに近づける（スパース性）。解釈性が高く、影響の強い特徴量を絞り込んで学習する。"),
            ("🔵 Custom Ridge",          "linear",   "年齢²・年齢×経験年数の交互作用項など独自特徴量を追加した線形モデル。給与ピーク帯（35〜54歳）のフラグも組み込み、年収カーブの非線形な動きを線形モデルで近似する。"),
            ("🟢 Random Forest",         "tree",     "複数の決定木を組み合わせたバギングモデル。職種ごとの細かい条件分岐を学習するが、外挿（訓練データ範囲外の予測）が苦手。CV精度は低めだが、特定職種の上振れ・下振れシナリオ確認に有効。"),
            ("🟢 Gradient Boosting",     "tree",     "弱い決定木を順番に積み上げてエラーを修正する勾配ブースティング。sklearnの標準実装で追加インストール不要。XGBoostより低速だが安定性が高く、過学習に強い。"),
            ("🟡 LightGBM",              "boosting", "MicrosoftのLightGBMは葉ごとに成長する「Leaf-wise」戦略で高速。職種名をカテゴリ変数としてネイティブに処理でき、OHEが不要。大規模データでも実用的な速度で学習できる。"),
            ("🟡 CatBoost",              "boosting", "Yandexのカテゴリ特化ブースティング。Target Encodingを対称木とともに最適化する独自手法で職種名を扱う。ハイパーパラメータのデフォルト値が優秀でチューニング不要でも高精度。"),
            ("🟡 XGBoost",               "boosting", "勾配ブースティングの業界標準。L1/L2正則化・欠損値の自動処理・並列化など多くの最適化を備える。特徴量エンジニアリング（FE）を組み合わせることでさらに精度が向上。"),
            ("🏆 Stacking Ensemble",     "ensemble", "全ベースモデルのOOF（Out-of-Fold）予測をメタ特徴量としてRidgeで統合する2層アンサンブル。単一モデルでは捉えられない「モデル間の誤差の相補関係」を学習し、理論上最高精度を目指す。ただし訓練時間は最長。"),
        ]
        type_color = {"linear": "#4F8EF7", "tree": "#4CAF50", "boosting": "#FF9800", "ensemble": "#E040FB"}
        type_label = {"linear": "線形系", "tree": "ツリー系", "boosting": "ブースティング系", "ensemble": "アンサンブル"}

        for model_name, mtype, desc in MODEL_DESCRIPTIONS:
            clr = type_color[mtype]
            lbl = type_label[mtype]
            st.markdown(
                f"<div style='border-left:3px solid {clr};padding:.4rem .8rem;margin:.35rem 0;"
                f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0'>"
                f"<span style='font-weight:700;color:{clr}'>{model_name}</span>"
                f"<span style='font-size:.7rem;background:{clr}22;color:{clr};"
                f"border-radius:4px;padding:1px 6px;margin-left:8px'>{lbl}</span>"
                f"<div style='font-size:.82rem;color:#ccc;margin-top:.25rem'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── スキル引継ぎ率ガイド（デフォルト閉じた状態・精度情報の下に移動）──
    _render_skill_transfer_table(
        models, p["model_key"],
        p["target_occ"], p["current_age"], p["current_exp"],
        p["current_income"], AGE_ALL_PATH,
    )

    # ── 全モデル比較グラフ ──
    st.markdown(f'<div class="sec-hdr">📉 全{len(MODEL_KEYS)}モデル比較グラフ</div>', unsafe_allow_html=True)
    fig3 = plot_all_models(sq_all, cc_all, p["current_age"],
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
    od  = p.get("offer_discount", 0.0)
    rs  = p.get("raise_suppression", 0.0)
    cr  = p.get("career_risk", 0.0)
    st.caption(
        f"📌 使用モデル: {p['model_label']} ／ "
        f"GDP: {gdp_growth:+.2f}% ／ CPI: {future_cpi} ／ "
        f"オファー割引: {od*100:.0f}% ／ 昇給抑制: {rs*100:.0f}% ／ キャリアリスク: {cr*100:.0f}% ／ "
        "本シミュレーションは厚生労働省「賃金構造基本統計調査」・GDP・CPI をもとにした統計的推計です。"
        "個人の実際の収入を保証するものではありません。"
    )


if __name__ == "__main__":
    main()
