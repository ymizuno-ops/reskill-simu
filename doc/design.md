# design.md
# リスキリング年収シミュレーター 設計書

---

## 1. システム構成

```
reskill-simu/
├── app.py                      # フロントエンド + シミュレーションロジック
├── src/
│   ├── step1_to_processed.py   # データ変換（raw → processed）
│   ├── step2_to_master.py      # データ構築（processed → master）
│   └── step3_train.py          # モデル訓練
├── data/
│   ├── raw/                    # e-stat 元データ（xlsx/csv）
│   ├── processed/              # 整形済みCSV（6ファイル）
│   └── master/                 # ML用CSV・JSON（5ファイル）
└── models/
    ├── models.pkl              # 訓練済みモデル辞書
    └── model_meta.json         # 精度メタ情報
```

---

## 2. データフロー

```
e-stat (xlsx/csv)
    │
    ▼  src/step1_to_processed.py
data/processed/
    ├── occupation_wage_all.csv     # 職種別給与（年×職種）
    ├── age_wage_all.csv            # 年齢階級×職種×年別給与
    ├── experience_wage_all.csv     # 経験年数×職種×年別給与
    ├── monthly_labor_all.csv       # 毎月勤労統計（年別前年比）
    ├── gdp_annual.csv              # 実質GDP成長率（年次）
    └── cpi_annual.csv              # CPI総合指数（年平均）
    │
    ▼  src/step2_to_master.py
data/master/
    ├── occupation_list.csv         # 職種マスタ（148職種・2024年基準）
    ├── age_curve.csv               # 年齢別平均年収・昇給率（11階級）
    ├── exp_curve.csv               # 経験年数別平均年収（5〜6階級）
    ├── ml_dataset.csv              # ML訓練用データ（5,578サンプル）
    └── macro_params.json           # マクロ経済パラメータ（10年平均）
    │
    ▼  src/step3_train.py
models/
    ├── models.pkl                  # 訓練済みモデル辞書（最大9モデル）
    └── model_meta.json             # R²・CV・MAE
    │
    ▼  app.py
Streamlit UI（ブラウザ）
```

---

## 3. モジュール設計

### 3.1 step1_to_processed.py

| 関数 | 入力 | 出力 | 説明 |
|---|---|---|---|
| `process_occupation_wage()` | raw/職種別/*.xlsx | occupation_wage_all.csv | 旧形式(〜2019)/新形式(2020〜)を自動判定して解析 |
| `process_age_wage()` | raw/年齢階級別/*.xlsx | age_wage_all.csv | 職種×年齢階級×年の給与データ |
| `process_experience_wage()` | raw/経験年数別/*.xlsx | experience_wage_all.csv | 経験年数バンドの列インデックスを動的検出 |
| `process_monthly_labor()` | raw/毎月勤労統計/*.xlsx | monthly_labor_all.csv | 「調査産業計」行から前年比を抽出 |
| `process_gdp()` | raw/国民経済計算/*.csv | gdp_annual.csv | shift_jis, 年度→暦年変換 |
| `process_cpi()` | raw/消費者物価指数/*.xlsx | cpi_annual.csv | skiprows=14, col8=年, col12=総合指数 |

**フォーマット差異対応（職種別・年齢別・経験年数別）**

```
旧形式（〜2019年）
  - 列0: 職種名(男)  列5: 給与  列7: 賞与
  - 性別が職種名に含まれる（例: "システムコンサルタント(男)"）

新形式（2020年〜）
  - 列1: 職種名  列7: 給与  列9: 賞与
  - 男女計が別行として存在
  - ファイル名に「産業計」が含まれる
```

---

### 3.2 step2_to_master.py

| 関数 | 入力 | 出力 | 説明 |
|---|---|---|---|
| `build_occupation_list()` | occupation_wage_all.csv | occupation_list.csv | 2024年最新・重複除去・年収フィルタ（100〜3000万） |
| `build_age_curve()` | age_wage_all.csv | age_curve.csv | 2024年全職種平均・昇給率（隣接階級比・-5〜12%クリップ） |
| `build_exp_curve()` | experience_wage_all.csv | exp_curve.csv | 2024年全職種平均・経験年数別 |
| `build_ml_dataset()` | occupation_list + age_wage + exp_wage | ml_dataset.csv | 職種×年齢×経験の組み合わせで合成年収を生成 |
| `build_macro_params()` | monthly_labor + gdp + cpi | macro_params.json | 直近10年平均・名目/実質賃金成長率 |

**MLデータセット生成ロジック**

```python
# 3要素の相乗平均で合成年収を算出
age_ratio  = 年齢別年収 / 年齢別平均
exp_ratio  = 経験年数別年収 / 経験年数別平均
synthetic  = base_income * sqrt(age_ratio * exp_ratio)
final      = synthetic * Normal(1.0, 0.05)  # ±5%ノイズ

# 不可能な組み合わせを除外
if exp_yr > age_mid - 18: continue  # 経験年数が年齢的に不可能
```

---

### 3.3 step3_train.py

#### クラス設計

```
BaseEstimator, RegressorMixin  ← sklearn 互換の基底クラス
    │
    ├── LGBMWrapper          # LightGBM ラッパー（LabelEncoding内包）
    ├── CatBoostWrapper      # CatBoost ラッパー（cat_features指定）
    └── StackingEnsemble     # 2層アンサンブル（OOF + Ridgeメタモデル）
```

> **重要**: pickle で保存するクラスは必ずモジュールレベルで定義すること。
> 関数内のローカルクラスは `pickle.dump` 時に AttributeError になる。

#### モデル辞書のスキーマ

```python
models = {
    "model_key": {
        "pipeline": fitted_model,   # predict() を持つオブジェクト
        "meta": {
            "r2_train":   float,    # 訓練データ R²
            "r2_cv_mean": float,    # CV R² 平均
            "r2_cv_std":  float,    # CV R² 標準偏差
            "mae_train":  float,    # 訓練データ MAE（万円）
            "features":   list,     # 入力特徴量リスト
        },
        "label": str,               # UI表示用ラベル
        "desc":  str,               # UI表示用説明文
        "uses_fe": bool,            # 特徴量エンジニアリング要否
    }
}
```

#### 特徴量エンジニアリング（FEあり モデル共通）

```python
age_sq         = age ** 2 / 1000        # 昇給の逓減・ピーク後低下
age_x_exp      = age * experience / 100 # シニア×ベテランの相乗効果
exp_ratio      = experience / age       # 年齢に占める経験年数割合
prime_age_flag = 1 if 35 <= age <= 54   # 給与ピーク帯フラグ
```

#### FE適用モデル一覧

| モデルキー | FE | 前処理 |
|---|---|---|
| ridge | ❌ | OHE + StandardScaler |
| elasticnet | ✅ | OHE + StandardScaler |
| custom | ✅ | OHE + StandardScaler |
| random_forest | ❌ | OHE + StandardScaler |
| gradient_boosting | ✅ | OHE + StandardScaler |
| lightgbm | ❌ | LabelEncoding（内部） |
| catboost | ❌ | なし（ネイティブ） |
| xgboost | ✅ | OHE + StandardScaler |
| stacking | ❌ | 内部で各モデルが処理 |

#### StackingEnsemble の動作フロー

```
fit(X, y):
  1. KFold(n=5) で OOF 予測行列を生成
     各 fold: ベースモデルを fold 内の訓練データで再訓練 → 検証データを予測
  2. 全データでベースモデルを再訓練（最終予測用）
  3. メタ特徴量 = OOF予測行列 + [age, experience_years]
  4. Ridge(alpha=1.0) でメタモデルを訓練

predict(X):
  1. 各ベースモデルで予測 → 予測行列を作成
  2. メタ特徴量 = 予測行列 + [age, experience_years]
  3. メタモデルで最終予測
```

---

### 3.4 app.py

#### 主要関数一覧

| 関数 | 役割 |
|---|---|
| `load_assets()` | models.pkl / CSV / JSON を読み込む（@cache_resource） |
| `_set_jp_font()` | OS問わず日本語フォントを自動検出・設定 |
| `_add_features(X)` | 特徴量エンジニアリングを適用 |
| `predict(models, key, occ, age, exp)` | モデル種別に応じてFE適用・予測 |
| `_get_one_step_down_income(occ, age, path)` | 1段下の年齢階級の年収を返す |
| `simulate(...)` | 50年間の現状維持・転職後シミュレーション |
| `calc_roi(sq, cc, cost)` | 回収月数・生涯差益を計算 |
| `render_sidebar(...)` | サイドバーUI・全パラメータを返す |
| `_render_skill_transfer_table(...)` | スキル引継ぎ率ガイド表を描画 |
| `render_analysis_results(...)` | 分析結果5ブロックを描画 |
| `plot_main(...)` | 年収推移グラフ（2軸）を描画 |
| `plot_all_models(...)` | 全モデル比較グラフ（動的サブプロット）を描画 |

#### セッション状態管理

```python
st.session_state["sim_done"]   = bool    # シミュレーション実行済みフラグ
st.session_state["sim_params"] = dict    # 実行時のパラメータを保持
# ボタン押下後にサイドバーが変更されても結果が消えないよう保持
```

#### predict() の振り分けロジック

```python
_FE_MODELS = {"custom", "xgboost", "elasticnet", "gradient_boosting"}

def predict(models, key, occ, age, exp):
    X = DataFrame([{"occupation": occ, "age": age, "experience_years": exp}])
    if key != "stacking" and key in _FE_MODELS:
        X = _add_features(X)   # FEあり
    return float(models[key]["pipeline"].predict(X)[0])
    # stacking は内部で各ベースモデルのFEを処理するため除外
```

---

## 4. データモデル

### 4.1 processed CSV スキーマ

**occupation_wage_all.csv**
```
year           : int    - 調査年（2006〜2024）
occupation     : str    - 職種名
monthly_wage   : float  - 月額給与（万円）
annual_bonus   : float  - 年間賞与（万円）
annual_income  : float  - 年収 = monthly_wage×12 + annual_bonus（万円）
```

**age_wage_all.csv**
```
year           : int    - 調査年
occupation     : str    - 職種名
age_label      : str    - 年齢階級ラベル（例: "30〜34歳"）
age_mid        : float  - 年齢中央値（例: 32.0）
monthly_wage   : float  - 月額給与（万円）
annual_bonus   : float  - 年間賞与（万円）
annual_income  : float  - 年収（万円）
```

**experience_wage_all.csv**
```
year             : int    - 調査年
occupation       : str    - 職種名
experience_years : float  - 経験年数バンドの代表値（0.0/2.5/7.0/12.0/17.0/22.0）
monthly_wage     : float  - 月額給与（万円）
annual_bonus     : float  - 年間賞与（万円）
annual_income    : float  - 年収（万円）
```

### 4.2 master JSON スキーマ

**macro_params.json**
```json
{
  "latest_year":            2024,
  "avg_wage_growth_10yr":   0.0055,  // 名目賃金上昇率（10年平均）
  "avg_gdp_growth_10yr":    0.0056,  // 実質GDP成長率（10年平均）
  "avg_cpi_10yr":           0.0123,  // CPI変化率（10年平均）
  "real_wage_growth_10yr":  -0.0068, // 実質賃金成長率
  "forecast_nominal_raise": 0.0055,  // 将来推計（名目・0〜3%クリップ）
  "forecast_real_raise":    0.0      // 将来推計（実質・-1〜2%クリップ）
}
```

---

## 5. 年齢階級定義

| インデックス | ラベル | mid値 |
|---|---|---|
| 0 | 〜19歳 | 18.0 |
| 1 | 20〜24歳 | 22.0 |
| 2 | 25〜29歳 | 27.0 |
| 3 | 30〜34歳 | 32.0 |
| 4 | 35〜39歳 | 37.0 |
| 5 | 40〜44歳 | 42.0 |
| 6 | 45〜49歳 | 47.0 |
| 7 | 50〜54歳 | 52.0 |
| 8 | 55〜59歳 | 57.0 |
| 9 | 60〜64歳 | 62.0 |
| 10 | 65〜69歳 | 67.0 |

「1段下の年齢階級」= 現在のインデックス - 1（最小は0）

---

## 6. 職種大分類（15カテゴリ）

| カテゴリ | 代表的な職種 |
|---|---|
| 管理職 | 管理的職業従事者 |
| 専門職・技術職（IT・理工系） | システムコンサルタント・設計者、ソフトウェア作成者、研究者 |
| 専門職・技術職（医療・福祉） | 医師、看護師、薬剤師、理学療法士 |
| 専門職・技術職（教育・文化） | 大学教授、小・中学校教員、保育士 |
| 専門職・技術職（法務・経営・金融） | 公認会計士・税理士、法務従事者 |
| 事務職 | 総合事務員、企画事務員、会計事務従事者 |
| 営業・販売職 | 販売店員、機械器具営業、保険営業 |
| サービス職 | 飲食物調理従事者、理容・美容師、警備員 |
| 保安職 | 警備員 |
| 農林漁業 | 農林漁業従事者 |
| 生産・製造職（機械・金属） | 自動車組立、金属溶接、機械検査 |
| 生産・製造職（その他） | 食料品製造、化学製品製造、印刷 |
| 建設・土木職 | 大工、土木従事者、電気工事 |
| 輸送・機械運転職 | バス運転者、タクシー運転者、航空機操縦士 |
| 運搬・清掃・その他 | 船内荷役従事者、クリーニング職 |

---

## 7. 依存ライブラリ

| ライブラリ | バージョン | 用途 |
|---|---|---|
| streamlit | >=1.32.0 | WebアプリUIフレームワーク |
| pandas | >=2.0.0 | データ処理 |
| numpy | >=1.24.0 | 数値計算 |
| scikit-learn | >=1.3.0 | ML前処理・モデル |
| matplotlib | >=3.7.0 | グラフ描画 |
| openpyxl | >=3.1.0 | xlsx読み込み |
| japanize-matplotlib | >=1.1.3 | matplotlib日本語フォント |
| protobuf | >=4.25.0 | Streamlit依存 |
| blinker | >=1.7.0 | Streamlit依存 |
| lightgbm | >=4.0.0 | LightGBMモデル（任意） |
| catboost | >=1.2.0 | CatBoostモデル（任意） |
| xgboost | >=2.0.0 | XGBoostモデル（任意） |
