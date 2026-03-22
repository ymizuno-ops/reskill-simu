"""
step2_to_master.py
==================
data/processed/ の各CSVを結合・整形し、
機械学習に使用する data/master/ のCSVを生成する。

出力ファイル一覧:
  master/
    ml_dataset.csv        # 訓練用メインデータ
                          #   occupation, age, experience_years, annual_income
    occupation_list.csv   # 職種マスタ（UI用）
                          #   occupation, latest_annual_income, latest_monthly_wage
    age_curve.csv         # 年齢別平均年収カーブ（全職種平均）
    exp_curve.csv         # 経験年数別平均年収カーブ（全職種平均）
    macro_params.json     # マクロ経済パラメータ（将来推計用）
"""

from __future__ import annotations
import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

_HERE    = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(_HERE, "..", "data", "processed")
OUT_DIR  = os.path.join(_HERE, "..", "data", "master")
os.makedirs(OUT_DIR, exist_ok=True)

LATEST_YEAR = 2024   # 基準年


# ──────────────────────────────────────────────────────
# 補助: 年収のCPI実質化（全年を2024年基準に統一）
# ──────────────────────────────────────────────────────
def load_cpi_deflator() -> pd.Series:
    """year → deflator（2024年=1.0）のSeriesを返す"""
    cpi = pd.read_csv(os.path.join(PROC_DIR, "cpi_annual.csv"))
    base = cpi.loc[cpi["year"] == LATEST_YEAR, "cpi"].values[0]
    cpi["deflator"] = cpi["cpi"] / base
    return cpi.set_index("year")["deflator"]


# ──────────────────────────────────────────────────────
# 1. 職種マスタ（UI用職種リスト）
# ──────────────────────────────────────────────────────
def build_occupation_list() -> pd.DataFrame:
    occ_all = pd.read_csv(os.path.join(PROC_DIR, "occupation_wage_all.csv"))

    # 年齢行など不正な職種を除去
    occ_all = occ_all[~occ_all["occupation"].str.contains(r"歳|才", na=False)]

    # 最新年のデータを基準とする
    latest = occ_all[occ_all["year"] == LATEST_YEAR].copy()

    # 重複がある場合は月収の高いものを優先
    latest = (
        latest.sort_values("monthly_wage", ascending=False)
        .drop_duplicates(subset="occupation")
        .reset_index(drop=True)
    )

    # 年収が現実的な範囲のみ残す（100〜3000万円）
    latest = latest[
        (latest["annual_income"] >= 100) & (latest["annual_income"] <= 3000)
    ]

    result = latest[["occupation", "monthly_wage", "annual_bonus", "annual_income"]].copy()
    result = result.sort_values("occupation").reset_index(drop=True)

    result.to_csv(os.path.join(OUT_DIR, "occupation_list.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ occupation_list.csv: {len(result)} 職種")
    return result


# ──────────────────────────────────────────────────────
# 2. 年齢カーブ（全職種平均・2024年基準）
# ──────────────────────────────────────────────────────
def build_age_curve() -> pd.DataFrame:
    age_all = pd.read_csv(os.path.join(PROC_DIR, "age_wage_all.csv"))
    age_all = age_all[~age_all["occupation"].str.contains(r"歳|才", na=False)]

    # 最新年のみ使用、全職種平均
    latest = age_all[age_all["year"] == LATEST_YEAR]
    curve = (
        latest.groupby(["age_label", "age_mid"])
        .agg(monthly_wage=("monthly_wage", "mean"), annual_income=("annual_income", "mean"))
        .reset_index()
        .sort_values("age_mid")
    )

    # 年齢別の昇給率（隣接年齢階級との比率）
    curve["raise_rate"] = curve["annual_income"].pct_change().fillna(0.03)
    curve["raise_rate"] = curve["raise_rate"].clip(-0.05, 0.12)

    curve.to_csv(os.path.join(OUT_DIR, "age_curve.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ age_curve.csv: {len(curve)} 年齢階級")
    print(curve[["age_label", "age_mid", "monthly_wage", "raise_rate"]].to_string(index=False))
    return curve


# ──────────────────────────────────────────────────────
# 3. 経験年数カーブ（全職種平均・2024年基準）
# ──────────────────────────────────────────────────────
def build_exp_curve() -> pd.DataFrame:
    exp_all = pd.read_csv(os.path.join(PROC_DIR, "experience_wage_all.csv"))
    exp_all = exp_all[~exp_all["occupation"].str.contains(r"歳|才", na=False)]

    latest = exp_all[exp_all["year"] == LATEST_YEAR]
    curve = (
        latest.groupby("experience_years")
        .agg(monthly_wage=("monthly_wage", "mean"), annual_income=("annual_income", "mean"))
        .reset_index()
        .sort_values("experience_years")
    )

    curve.to_csv(os.path.join(OUT_DIR, "exp_curve.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ exp_curve.csv: {len(curve)} 経験年数階級")
    print(curve.to_string(index=False))
    return curve


# ──────────────────────────────────────────────────────
# 4. 機械学習用メインデータセット
# ──────────────────────────────────────────────────────
def build_ml_dataset(occ_list: pd.DataFrame) -> pd.DataFrame:
    """
    年齢階級×経験年数×職種 の組み合わせで年収を計算したデータセットを作る。

    設計方針:
      - 職種別ベース年収（latest年）を基準とする
      - 年齢による倍率（age_curve基準）を乗算
      - 経験年数による倍率（exp_curve基準）を乗算
      - 全体の平均値でスケーリングして自然なレンジに調整
      - 年度推移（2006〜2024）の賃金上昇率を反映
    """
    age_all = pd.read_csv(os.path.join(PROC_DIR, "age_wage_all.csv"))
    exp_all = pd.read_csv(os.path.join(PROC_DIR, "experience_wage_all.csv"))

    # 不正職種除去
    for df in [age_all, exp_all]:
        mask = ~df["occupation"].str.contains(r"歳|才", na=False)
        df.drop(df[~mask].index, inplace=True)

    # 職種×年齢の2024年データ
    age_2024 = age_all[age_all["year"] == LATEST_YEAR].copy()
    exp_2024 = exp_all[exp_all["year"] == LATEST_YEAR].copy()

    valid_occs = set(occ_list["occupation"])

    records = []
    for _, occ_row in occ_list.iterrows():
        occ = occ_row["occupation"]
        base_income = occ_row["annual_income"]

        # この職種の年齢別データ
        occ_age = age_2024[age_2024["occupation"] == occ]
        occ_exp = exp_2024[exp_2024["occupation"] == occ]

        # 年齢データがなければ全職種平均カーブを使う
        if len(occ_age) == 0:
            occ_age = age_2024.groupby(["age_label", "age_mid"]).agg(
                annual_income=("annual_income", "mean")
            ).reset_index()

        # 経験年数データがなければ全職種平均を使う
        if len(occ_exp) == 0:
            occ_exp = exp_2024.groupby("experience_years").agg(
                annual_income=("annual_income", "mean")
            ).reset_index()

        # 全職種の平均年収（スケーリング基準）
        age_mean = occ_age["annual_income"].mean()
        exp_mean = occ_exp["annual_income"].mean() if len(occ_exp) > 0 else base_income

        for _, age_row in occ_age.iterrows():
            age_mid = age_row["age_mid"]
            age_income = age_row["annual_income"]
            age_ratio = age_income / age_mean if age_mean > 0 else 1.0

            for _, exp_row in occ_exp.iterrows():
                exp_yr = exp_row["experience_years"]
                exp_income = exp_row["annual_income"]

                # 経験年数が年齢的に不可能なケースを除外
                # （最低でも18歳から働いたとして、経験年数 <= 年齢 - 18）
                if exp_yr > age_mid - 18:
                    continue

                exp_ratio = exp_income / exp_mean if exp_mean > 0 else 1.0

                # 合成年収 = ベース × 年齢倍率 × 経験年数倍率（相乗平均で合成）
                synthetic_income = base_income * np.sqrt(age_ratio * exp_ratio)

                # ノイズを加えて汎化性能向上（±5%）
                noise = np.random.normal(1.0, 0.05)
                final_income = max(synthetic_income * noise, 50)

                records.append({
                    "occupation": occ,
                    "age": age_mid,
                    "experience_years": exp_yr,
                    "annual_income": round(final_income, 1),
                })

    df = pd.DataFrame(records)
    print(f"\n  ML dataset: {len(df):,} サンプル, {df['occupation'].nunique()} 職種")
    print(f"  年収レンジ: {df['annual_income'].min():.0f}〜{df['annual_income'].max():.0f} 万円")
    print(f"  年収中央値: {df['annual_income'].median():.0f} 万円")

    df.to_csv(os.path.join(OUT_DIR, "ml_dataset.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ ml_dataset.csv: {len(df):,} レコード")
    return df


# ──────────────────────────────────────────────────────
# 5. マクロ経済パラメータ
# ──────────────────────────────────────────────────────
def build_macro_params() -> dict:
    labor = pd.read_csv(os.path.join(PROC_DIR, "monthly_labor_all.csv"))
    gdp   = pd.read_csv(os.path.join(PROC_DIR, "gdp_annual.csv"))
    cpi   = pd.read_csv(os.path.join(PROC_DIR, "cpi_annual.csv"))

    # 直近10年の平均（より現実的な推計値）
    recent_labor = labor[labor["year"] >= LATEST_YEAR - 10]
    recent_gdp   = gdp[gdp["year"]   >= LATEST_YEAR - 10]
    recent_cpi   = cpi[cpi["year"]   >= LATEST_YEAR - 10]

    avg_wage_growth = recent_labor["yoy_rate"].dropna().mean()
    avg_gdp_growth  = recent_gdp["gdp_real_growth"].dropna().mean()
    avg_cpi_yoy     = recent_cpi["cpi_yoy"].dropna().mean()

    # 実質賃金成長率 = 名目賃金成長 - インフレ率
    real_wage_growth = avg_wage_growth - avg_cpi_yoy

    # 将来推計: 直近実績を重視、最低0%・最大3%でクリップ
    forecast_nominal = float(np.clip(avg_wage_growth, 0.0, 0.03))
    forecast_real    = float(np.clip(real_wage_growth, -0.01, 0.02))

    params = {
        "latest_year":            LATEST_YEAR,
        "avg_wage_growth_10yr":   float(avg_wage_growth),
        "avg_gdp_growth_10yr":    float(avg_gdp_growth),
        "avg_cpi_10yr":           float(avg_cpi_yoy),
        "real_wage_growth_10yr":  float(real_wage_growth),
        "forecast_nominal_raise": forecast_nominal,
        "forecast_real_raise":    forecast_real,
    }

    out_path = os.path.join(OUT_DIR, "macro_params.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print(f"  ✅ macro_params.json")
    print(f"     名目賃金上昇（10年平均）: {avg_wage_growth*100:.2f}%")
    print(f"     実質GDP成長（10年平均）:  {avg_gdp_growth*100:.2f}%")
    print(f"     CPI（10年平均）:          {avg_cpi_yoy*100:.2f}%")
    print(f"     将来推計（名目）:          {forecast_nominal*100:.2f}%")
    return params


# ──────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("  Step2: processed → master  データセット構築")
    print("=" * 60 + "\n")

    print("[職種マスタ]")
    occ_list = build_occupation_list()
    print()

    print("[年齢カーブ]")
    age_curve = build_age_curve()
    print()

    print("[経験年数カーブ]")
    exp_curve = build_exp_curve()
    print()

    print("[MLデータセット構築]")
    ml_df = build_ml_dataset(occ_list)
    print()

    print("[マクロ経済パラメータ]")
    build_macro_params()

    print("\n" + "=" * 60)
    print("  Step2 完了 → data/master/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
