# tasks.md
# リスキリング年収シミュレーター 実装タスク一覧

AI駆動開発（Cursor / GitHub Copilot 等）で本アプリを再現するための
タスク分解リストです。各タスクは独立して実装・テスト可能な粒度で定義しています。

---

## フェーズ概要

```
Phase 1  環境セットアップ              （タスク 1〜2）
Phase 2  データパイプライン構築         （タスク 3〜9）
Phase 3  MLデータセット・モデル訓練     （タスク 10〜18）
Phase 4  Streamlit アプリ基礎          （タスク 19〜24）
Phase 5  シミュレーションロジック       （タスク 25〜29）
Phase 6  UI・グラフ整備                （タスク 30〜37）
Phase 7  品質・仕上げ                  （タスク 38〜42）
```

---

## Phase 1: 環境セットアップ

### Task 1: プロジェクト構造の作成
**目的**: 統一されたディレクトリ構造を用意する

```
作成するディレクトリ:
  reskill-simu/
    data/raw/経験年数階級別きまって支給する現金給与額/
    data/raw/国民経済計算_GDP統計/
    data/raw/消費者物価指数/
    data/raw/職種別きまって支給する現金給与額/
    data/raw/年齢階級別きまって支給する現金給与額/
    data/raw/毎月勤労統計調査_結果確報/
    data/processed/
    data/master/
    src/
    models/
    tests/
```

**完了条件**: `find . -type d | sort` で全ディレクトリが存在すること

---

### Task 2: requirements.txt の作成・インストール

**ファイル内容**:
```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
openpyxl>=3.1.0
japanize-matplotlib>=1.1.3
protobuf>=4.25.0
blinker>=1.7.0
lightgbm>=4.0.0
catboost>=1.2.0
xgboost>=2.0.0
```

**完了条件**: `pip install -r requirements.txt` が正常完了すること

---

## Phase 2: データパイプライン構築（step1_to_processed.py）

### Task 3: 共通ユーティリティ関数の実装

**ファイル**: `src/step1_to_processed.py`

**実装する関数**:
```python
def safe_num(v) -> float:
    # 文字列・ハイフン・記号を NaN に変換
    # "-", "－", "…", "***", "nan", "" → np.nan

def extract_year(fname: str) -> int:
    # ファイル名から4桁の年を抽出
    # "2024年_職種別..." → 2024

def list_xlsx(key: str) -> list[str]:
    # サブディレクトリ内の .xlsx ファイル一覧を返す
```

**完了条件**: 単体テストで各関数が期待通りに動作すること

---

### Task 4: 職種別給与データの解析（旧形式・新形式対応）

**対象ファイル**: `data/raw/職種別きまって支給する現金給与額/*.xlsx`

**列レイアウト**（旧新共通）:
```
col1 = 職種名
col7 = きまって支給する現金給与額（千円）
col9 = 年間賞与その他特別給与額（千円）
```

**フォーマット差異**:
```
旧形式（〜2019）: データ開始行が異なる。職種名に性別(男)が含まれる
新形式（2020〜）: col1に「男女計\n職種名」形式で記載
```

**実装する関数**:
```python
def process_occupation_wage() -> pd.DataFrame:
    # 出力: occupation_wage_all.csv
    # カラム: year, occupation, monthly_wage, annual_bonus, annual_income
```

**完了条件**:
- 2006〜2024年の19ファイルが処理され、occupation_wage_all.csv が生成されること
- `df['occupation'].nunique()` が 100 以上であること

---

### Task 5: 年齢階級別給与データの解析

**対象ファイル**: `data/raw/年齢階級別きまって支給する現金給与額/*.xlsx`

**旧形式（〜2019）の列レイアウト**:
```
col0 = 職種名(男) or 年齢階級
col5 = 給与（千円）
col7 = 賞与（千円）
```

**新形式（2020〜）の列レイアウト**:
```
col1 = 職種名 or 年齢階級（インデントあり）
col7 = 給与（千円）
col9 = 賞与（千円）
```

**年齢階級の検出**:
```python
# 年齢階級行の判定
age_m  = re.search(r"(\d+)\s*[～~]\s*(\d+)", name)
is_u19 = "１９歳" in name or "19歳" in name
# mid値: (a1 + a2) / 2, 〜19歳は 18.0
```

**実装する関数**:
```python
def _parse_age_new(df, year) -> list  # 2020〜
def _parse_age_old(df, year) -> list  # 〜2019
def process_age_wage() -> pd.DataFrame:
    # 出力: age_wage_all.csv
    # カラム: year, occupation, age_label, age_mid, monthly_wage, annual_bonus, annual_income
```

**完了条件**: age_wage_all.csv が生成され、age_mid に 18〜67 の値が存在すること

---

### Task 6: 経験年数別給与データの解析

**対象ファイル**: `data/raw/経験年数階級別きまって支給する現金給与額/*.xlsx`

**経験年数バンドの列マッピング**（ヘッダー行から動的検出）:
```python
patterns = [
    ("total", ["経験年数計"]),
    (0.0,   ["０年", "0年"]),
    (2.5,   ["１～４年", "1～4年"]),
    (7.0,   ["５～９年", "5～9年"]),
    (12.0,  ["１０～１４年", "10～14年"]),
    (17.0,  ["１５～１９年", "15～19年", "１５年以上", "15年以上"]),
    (22.0,  ["２０年以上", "20年以上"]),
]
```

**実装する関数**:
```python
def _get_exp_col_map(df) -> dict   # ヘッダーを走査して列マップを返す
def _parse_exp(df, year, col_map, new_fmt) -> list
def process_experience_wage() -> pd.DataFrame:
    # 出力: experience_wage_all.csv
    # カラム: year, occupation, experience_years, monthly_wage, annual_bonus, annual_income
```

**完了条件**: experience_years に [0.0, 2.5, 7.0, 12.0, 17.0] が存在すること（22.0は旧形式のみ）

---

### Task 7: 毎月勤労統計データの解析

**対象ファイル**: `data/raw/毎月勤労統計調査_結果確報/*.xlsx`

**抽出対象**: 「調査産業計」行
```
col4 = きまって支給する給与（円）
col5 = 前年比（%）
```

**実装する関数**:
```python
def process_monthly_labor() -> pd.DataFrame:
    # 出力: monthly_labor_all.csv
    # カラム: year, scheduled_wage_yen, yoy_pct, yoy_rate
```

**完了条件**: 2008〜2024年の17年分が存在すること

---

### Task 8: GDP・CPI データの解析

**GDP（2024_年次GDP成長率_実質.csv）**:
```
エンコーディング: shift_jis
col0 = 年度（"YYYY/4-3."形式）→ YYYY に変換
col1 = 実質GDP成長率（%）
```

**CPI（2025_消費者物価指数_中分類指数_全国__年平均.xlsx）**:
```
skiprows=14
col8  = 年（"YYYY年"形式）
col12 = 総合指数（2020年=100）
```

**実装する関数**:
```python
def process_gdp() -> pd.DataFrame:
    # 出力: gdp_annual.csv
    # カラム: year, gdp_real_growth_pct, gdp_real_growth

def process_cpi() -> pd.DataFrame:
    # 出力: cpi_annual.csv
    # カラム: year, cpi, cpi_yoy
```

**完了条件**: gdp_annual.csv に1995〜2024年、cpi_annual.csv に1970〜2024年のデータがあること

---

### Task 9: step1 メイン関数の実装

```python
def main():
    process_occupation_wage()
    process_age_wage()
    process_experience_wage()
    process_monthly_labor()
    process_gdp()
    process_cpi()
```

**完了条件**: `python src/step1_to_processed.py` が正常終了し、processed/ に6ファイルが生成されること

---

## Phase 3: MLデータセット・モデル訓練

### Task 10: 職種マスタの構築（step2）

**ファイル**: `src/step2_to_master.py`

```python
def build_occupation_list() -> pd.DataFrame:
    # 2024年最新データから148職種を抽出
    # フィルタ: 年収100〜3000万円、歳・才を含む行を除外
    # 出力: occupation_list.csv
    # カラム: occupation, monthly_wage, annual_bonus, annual_income
```

---

### Task 11: 年齢カーブ・経験年数カーブの構築

```python
def build_age_curve() -> pd.DataFrame:
    # 2024年全職種平均の年齢別年収と昇給率
    # raise_rate = 隣接階級の年収変化率（-5〜12%クリップ）
    # 出力: age_curve.csv

def build_exp_curve() -> pd.DataFrame:
    # 2024年全職種平均の経験年数別年収
    # 出力: exp_curve.csv
```

---

### Task 12: MLデータセットの構築

```python
def build_ml_dataset(occ_list) -> pd.DataFrame:
    # 職種 × 年齢 × 経験年数 の組み合わせで合成年収を生成
    # 合成式: base × sqrt(age_ratio × exp_ratio) × Normal(1.0, 0.05)
    # 不可能な組み合わせ除外: exp_yr > age_mid - 18
    # 出力: ml_dataset.csv  shape: (5578, 4)
    # カラム: occupation, age, experience_years, annual_income
```

---

### Task 13: マクロ経済パラメータの算出

```python
def build_macro_params() -> dict:
    # 直近10年の平均から将来推計を算出
    # forecast_nominal_raise = avg_wage_growth.clip(0, 0.03)
    # 出力: macro_params.json
```

---

### Task 14: 共通前処理・特徴量エンジニアリングの実装

**ファイル**: `src/step3_train.py`

```python
def make_ohe_preprocessor(num_features) -> ColumnTransformer:
    # occupation → OneHotEncoder
    # num_features → StandardScaler

def add_features(X) -> pd.DataFrame:
    # age_sq         = age ** 2 / 1000
    # age_x_exp      = age * experience_years / 100
    # exp_ratio      = experience_years / age.clip(lower=1)
    # prime_age_flag = (35 <= age <= 54).astype(float)
```

---

### Task 15: sklearn モデル（Ridge / ElasticNet / Custom Ridge / RF / GBM）の実装

```python
def train_ridge(X, y) -> tuple[Pipeline, dict]
def train_elasticnet(X, y) -> tuple[Pipeline, dict]
    # alpha=0.001, l1_ratio=0.7（チューニング済み）
def train_custom_ridge(X, y) -> tuple[Pipeline, dict]
def train_random_forest(X, y) -> tuple[Pipeline, dict]
def train_gradient_boosting(X, y) -> tuple[Pipeline, dict]
```

**共通ヘルパー**:
```python
def _cv_and_fit(pipe, X, y, label) -> tuple:
    # 5-fold CV → fit → R²/MAE 計算 → 経過時間表示
```

---

### Task 16: 勾配ブースティング系 Wrapper クラスの実装

> **注意**: pickle 保存のためクラスはモジュールレベルで定義すること

```python
class LGBMWrapper(BaseEstimator, RegressorMixin):
    # fit: LabelEncoding → LGBMRegressor.fit(categorical_feature=[0])
    # predict: 未知カテゴリを最頻値で補完

class CatBoostWrapper(BaseEstimator, RegressorMixin):
    # fit: CatBoostRegressor.fit(cat_features=["occupation"])
    # CV用(200iter)と最終訓練用(500iter)でパラメータを変える
```

```python
def train_lightgbm(X, y) -> tuple[LGBMWrapper, dict]
def train_catboost(X, y) -> tuple[CatBoostWrapper, dict]
    # CV: iterations=200, learning_rate=0.1, depth=6（高速化）
    # 最終: iterations=500, learning_rate=0.05, depth=8（高精度）
def train_xgboost(X, y) -> tuple[Pipeline, dict]
    # FEを適用した Xc を Pipeline に渡す
```

---

### Task 17: StackingEnsemble クラスの実装

```python
class StackingEnsemble(BaseEstimator, RegressorMixin):
    FE_KEYS = frozenset({"custom", "xgboost", "elasticnet", "gradient_boosting"})

    def __init__(self, base_models, n_splits=5, meta_alpha=1.0)

    def _prepare_X(self, X, key) -> pd.DataFrame:
        # key in FE_KEYS なら add_features を適用

    def _make_oof_matrix(self, X, y) -> np.ndarray:
        # KFold で各ベースモデルの OOF 予測行列を生成
        # shape: (n_samples, n_base_models)

    def _make_meta_X(self, oof_or_pred, X) -> np.ndarray:
        # [OOF予測行列, age, experience_years] を結合

    def fit(self, X, y):
        # 1. OOF予測行列の生成
        # 2. 全データでベースモデルを再訓練
        # 3. メタモデル（Ridge）の訓練

    def predict(self, X) -> np.ndarray:
        # 各ベースモデルで予測 → メタモデルで最終予測
```

---

### Task 18: step3 メイン関数の実装・モデル保存

```python
def main():
    # 1. sklearn 5モデルを訓練
    # 2. 利用可能な場合 LightGBM/CatBoost/XGBoost を訓練
    # 3. Stacking Ensemble を訓練（全ベースモデルが揃ってから）
    # 4. models.pkl / model_meta.json に保存
    # 5. 精度ランキングを表示（CV R² 降順）
```

**モデル辞書スキーマ**:
```python
models["key"] = {
    "pipeline": fitted_model,
    "meta": {"r2_train", "r2_cv_mean", "r2_cv_std", "mae_train", "features"},
    "label": str,
    "desc":  str,
    "uses_fe": bool,
}
```

**完了条件**: `python src/step3_train.py` が正常終了し、models.pkl に最大9モデルが保存されること

---

## Phase 4: Streamlit アプリ基礎（app.py）

### Task 19: ページ設定・CSS・フォント設定

```python
st.set_page_config(
    page_title="リスキリングによる年収シミュレーター",
    layout="wide"
)
# CSSの定義: .result-section / .metric-card / .sec-hdr / .lifetime-highlight

def _set_jp_font():
    # 優先順位: Yu Gothic/Meiryo(Win) → Hiragino(Mac) → Noto/IPAGothic(Linux)
    # font_manager.ttflist で存在確認してから設定

# japanize-matplotlib フォールバック
if _jp_font is None:
    try: import japanize_matplotlib
    except: pass
```

---

### Task 20: アセット読み込み関数の実装

```python
@st.cache_resource
def load_assets():
    # models.pkl が存在しなければ step1〜step3 を自動実行
    # 返り値: models, occ_list, age_curve, macro
```

---

### Task 21: Wrapper クラスのインポート（pickle復元用）

```python
# app.py の先頭付近
try:
    from step3_train import LGBMWrapper, CatBoostWrapper, StackingEnsemble
except ImportError:
    pass
```

> **理由**: pickle.load 時に step3_train.LGBMWrapper の参照が必要なため

---

### Task 22: predict() 関数の実装

```python
_FE_MODELS = {"custom", "xgboost", "elasticnet", "gradient_boosting"}

def _add_features(X) -> pd.DataFrame:
    # Task 14 と同じ特徴量エンジニアリング

def predict(models, model_key, occupation, age, experience) -> float:
    X = DataFrame([{"occupation": occupation, "age": age, "experience_years": experience}])
    if model_key != "stacking" and model_key in _FE_MODELS:
        X = _add_features(X)
    return float(models[model_key]["pipeline"].predict(X)[0])
```

---

### Task 23: サイドバーの実装

```python
def render_sidebar(occ_list, models, macro):
    # ── プロフィール設定 ──
    # current_age (number_input 20〜65)
    # current_exp (number_input 0〜40)
    # current_income (number_input 100〜3000)

    # ── 予測モデル選択 ──
    # models.pkl に存在するモデルのみ動的に選択肢を構築

    # ── キャリア選択 ──
    # cur_cat → cur_occs → current_occ（大分類フィルタ付きselectbox）
    # tgt_cat → tgt_occs → target_occ
    # skill_transfer (slider 0〜100%)

    # ── 投資設定 ──
    # learning_cost / gdp_growth / future_cpi
    # nominal_raise = gdp_growth/100 + (future_cpi-100)/100 * 0.3

    # ── リアリティ補正 ──
    # raise_suppression (slider 0〜50%)
    # career_risk (slider 0〜30%)

    return (current_occ, target_occ, current_age, current_exp, current_income,
            skill_transfer, learning_cost, model_key, model_label, nominal_raise,
            gdp_growth, future_cpi, raise_suppression, career_risk)
```

**職種大分類（15カテゴリ）の定義**: design.md の第6章を参照

---

### Task 24: セッション状態管理・実行ボタンの実装

```python
run = st.button("🚀 シミュレーション実行", type="primary")
if run:
    st.session_state["sim_done"]   = True
    st.session_state["sim_params"] = dict(...)  # 全パラメータを保存

if not st.session_state.get("sim_done"):
    st.info("サイドバーで条件を設定し、ボタンを押してください。")
    return
```

---

## Phase 5: シミュレーションロジック

### Task 25: 1段下の年齢階級年収の取得

```python
_AGE_MIDS   = [18.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 67.0]
_AGE_LABELS = ["〜19歳", "20〜24歳", ..., "65〜69歳"]

def _get_one_step_down_income(occ_name, current_age, age_all_path, year=2024) -> tuple[float, str]:
    # 1. 現在の年齢が属する階級のインデックスを特定
    # 2. インデックス - 1 の年齢階級の年収を返す
    # 3. 職種データがなければ全職種平均を使用
    # 返り値: (基準年収, 年齢階級ラベル)
```

---

### Task 26: simulate() 関数の実装

```python
def simulate(models, model_key,
             current_occ, target_occ,
             current_age, current_exp, current_income,
             skill_transfer, nominal_raise, age_curve,
             years=50, age_all_path="",
             raise_suppression=0.0, career_risk=0.0) -> tuple[list, list]:

    # ── 現状維持シナリオ ──
    # base_pred = predict(current_occ, current_age, current_exp)
    # correction = current_income / base_pred  # 実績年収との乖離補正
    # 毎年: income = predict(occ, age, exp+i) * correction * (1 + nominal_raise)
    # 65歳以降: income *= 0.97

    # ── 転職シナリオ ──
    # base_income = _get_one_step_down_income(target_occ, current_age)
    # exp_income  = predict(target_occ, current_age, current_exp)
    # first_income = base_income + (exp_income - base_income) * skill_transfer
    # 毎年:
    #   suppression_factor = 1 - raise_suppression * max(0, (10-i)/10)
    #   risk_decay         = 1 - career_risk * max(0, (i-10)/40)
    #   income = predict(target_occ, age, i) * corr2 * sf * rd * (1 + nominal_raise*i*0.05)

    return status_quo, career_change  # 各50要素のリスト
```

---

### Task 27: ROI 計算の実装

```python
def calc_roi(sq, cc, cost) -> tuple[int | None, float]:
    # breakeven_month: コスト回収が達成された月数（Noneなら回収不可）
    # lifetime: 生涯差益（転職後 - 現状維持の累積）
    # 回収判定: 累積差益 >= cost になった時点
```

---

### Task 28: スキル引継ぎ率ガイド表の実装

```python
def _render_skill_transfer_table(models, model_key, target_occ,
                                  current_age, current_exp,
                                  current_income, age_all_path):
    # 引継ぎ率: [0, 20, 40, 60, 80, 100] の各初年度年収を算出
    # HTMLテーブルで表示
    # st.expander で展開（expanded=False がデフォルト）
```

---

### Task 29: 分析結果5ブロックの実装

```python
def render_analysis_results(sq, cc, current_age, current_occ, target_occ,
                             current_income, skill_transfer, learning_cost):
    # 2カラムレイアウト
    # 左: 1.現状 / 2.転職直後 / 4.65歳時点の生涯年収（ハイライト表示）
    # 右: 3.転職5年後 / 5.費用対効果（回収月数・ROI）
```

---

## Phase 6: UI・グラフ整備

### Task 30: 年収推移グラフ（2軸）の実装

```python
def plot_main(sq, cc, current_age, current_occ, target_occ, cost) -> Figure:
    # ax1: 年収推移（現状=青, 転職後=赤）
    # ax2: 累積収支差額（緑点線、twinx）
    # 学習コスト水平線: orange 破線
    # 背景色: #1A1A2E（ダークモード）
```

---

### Task 31: 全モデル比較グラフの実装

```python
def plot_all_models(sq_all, cc_all, current_age, breakevens, model_labels) -> Figure:
    # モデル数に応じて動的にサブプロット列数を決定
    # ncols = min(3, n),  nrows = ceil(n / ncols)
    # 現状=青系, 転職後=赤系（モデルごとに明度を変える）
    # 回収時点に黄色破線・アノテーション
    # 余ったサブプロットは非表示
```

---

### Task 32: モデル精度情報エキスパンダーの実装

```python
with st.expander("🤖 モデル精度情報"):
    # 精度カード（3列×複数行・動的）
    # 各モデルの特徴説明（線形系/ツリー系/ブースティング系/アンサンブル）
    # 色分け: 線形=青 / ツリー=緑 / ブースティング=橙 / アンサンブル=紫
```

---

### Task 33: 年次詳細テーブルの実装

```python
# 5年刻みで選択モデルの年収推移を表示
rows = []
for i in range(0, 50, 5):
    rows.append({
        "年齢": f"{current_age + i}歳",
        "現状維持（万円）": ...,
        "転職後（万円）": ...,
        "年間差（万円）": ...,
        "累積差（万円）": ...,
    })
st.dataframe(rows, hide_index=True)
```

---

### Task 34: フッターの実装

```python
st.caption(
    f"使用モデル: {model_label} ／ GDP: {gdp:+.2f}% ／ CPI: {cpi} ／ "
    f"昇給抑制: {rs}% ／ キャリアリスク: {cr}% ／ "
    "本シミュレーションは統計的推計です。..."
)
```

---

### Task 35: CSSスタイルの整備

```css
/* ダークモード対応の主要スタイル */
.result-section    /* 分析結果ブロック */
.metric-card       /* 精度カード */
.sec-hdr           /* セクションヘッダー（左ボーダー） */
.lifetime-highlight /* 生涯差額の大きな数字 */
.pos / .neg / .neu  /* 色クラス（緑/赤/青） */
```

---

### Task 36: main() の組み立て

```python
def main():
    # 1. アセット読み込み
    # 2. サイドバー描画・パラメータ取得
    # 3. シミュレーション実行ボタン
    # 4. 選択モデルでシミュレーション
    # 5. 全モデルでシミュレーション（比較グラフ用）
    # 6. スキル引継ぎ率ガイド表
    # 7. 分析結果5ブロック
    # 8. 年収推移グラフ
    # 9. モデル精度情報エキスパンダー（精度カード + 特徴説明）
    # 10. 全モデル比較グラフ
    # 11. 年次詳細テーブル
    # 12. フッター
```

---

## Phase 7: 品質・仕上げ

### Task 37: 日本語フォント対応の確認

```
確認環境: Windows / macOS / Linux
確認事項:
  - グラフタイトルが文字化けしないこと
  - 軸ラベルが文字化けしないこと
  - 凡例が文字化けしないこと
```

### Task 38: README.txt の作成

```
記載内容:
  - 概要・ターゲットユーザー
  - ディレクトリ構成
  - セットアップ手順（4ステップ）
  - 搭載モデル一覧
  - シミュレーションロジック説明
  - データソース一覧
  - 注意事項
  - 更新履歴
```

### Task 39: requirements.md / design.md / tasks.md の作成

本ドキュメント群の作成。

### Task 40: テストコードの実装（tests/）

```python
# tests/test_simulate.py
def test_simulate_status_quo_decreases_after_65()
def test_first_income_zero_skill_transfer()
def test_first_income_full_skill_transfer()
def test_calc_roi_breakeven()

# tests/test_step1.py
def test_safe_num()
def test_extract_year()

# tests/test_step3.py
def test_stacking_pickle_serializable()
def test_lgbm_wrapper_unknown_category()
```

### Task 41: 動作確認チェックリスト

```
□ python src/step1_to_processed.py が正常終了すること
□ python src/step2_to_master.py が正常終了すること
□ python src/step3_train.py が正常終了すること（全9モデル）
□ streamlit run app.py でブラウザが開くこと
□ シミュレーション実行ボタンが動作すること
□ 全モデル比較グラフに9モデルが表示されること
□ スキル引継ぎ率ガイドが正しい値を表示すること
□ Windows/macOS/Linux でグラフが文字化けしないこと
```

### Task 42: 将来の拡張ポイント（メモ）

```
- Optuna によるハイパーパラメータ自動チューニング
- 手取り計算（所得税・住民税・社会保険料の控除）
- Streamlit Cloud へのデプロイ
- 毎年の e-stat データ更新の自動化スクリプト
- より詳細な職種分類（129種類の職種コード対応）
- 性別・企業規模別の予測オプション
```

---

## タスク依存関係

```
Task 1,2 → Task 3〜9（Phase 2 全体）
         → Task 10〜13（Phase 3 前半）
Task 3〜9 → Task 10〜13
Task 10〜13 → Task 14〜18（モデル訓練）
Task 14〜18 → Task 19〜24（アプリ基礎）
Task 19〜24 → Task 25〜29（シミュレーション）
Task 25〜29 → Task 30〜36（UI・グラフ）
Task 30〜36 → Task 37〜42（品質・仕上げ）
```
