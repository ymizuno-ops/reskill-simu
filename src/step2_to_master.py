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

from log_config import getLogger
from model_types import MacroParams

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

_HERE    = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(_HERE, "..", "data", "processed")
OUT_DIR  = os.path.join(_HERE, "..", "data", "master")
os.makedirs(OUT_DIR, exist_ok=True)

LATEST_YEAR  = 2024   # 基準年
RANDOM_SEED  = 42
CSV_ENCODING = "utf-8-sig"
INVALID_OCC_PATTERN = r"歳|才"   # 年齢行など職種でない行

# 職種マスタ: 年収が現実的な範囲（万円）
MIN_ANNUAL_INCOME = 100
MAX_ANNUAL_INCOME = 3000

# 年齢カーブ: 昇給率の初期値と上下限
DEFAULT_RAISE_RATE = 0.03
MIN_RAISE_RATE     = -0.05
MAX_RAISE_RATE     = 0.12

# ML データセット
WORK_START_AGE = 18     # 経験年数 <= 年齢 - 18 を満たす組み合わせだけ使う
NOISE_STD      = 0.05   # 汎化性能向上のためのノイズ（±5%）
MIN_INCOME     = 50     # 合成年収の下限（万円）

# マクロ経済パラメータ
RECENT_YEARS = 10
MIN_FORECAST_NOMINAL, MAX_FORECAST_NOMINAL = 0.0, 0.03
MIN_FORECAST_REAL,    MAX_FORECAST_REAL    = -0.01, 0.02
PERCENT = 100

SECTION_RULE_WIDTH = 60


def _readProcessed(fileName: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROC_DIR, fileName))


def _dropInvalidOcc(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["occupation"].str.contains(INVALID_OCC_PATTERN, na=False)]


def _saveCsv(df: pd.DataFrame, fileName: str) -> None:
    df.to_csv(os.path.join(OUT_DIR, fileName), index=False, encoding=CSV_ENCODING)


# ──────────────────────────────────────────────────────
# 1. 職種マスタ（UI用職種リスト）
# ──────────────────────────────────────────────────────
def buildOccupationList() -> pd.DataFrame:
    occAll = _dropInvalidOcc(_readProcessed("occupation_wage_all.csv"))

    # 最新年のデータを基準とし、重複がある場合は月収の高いものを優先
    latest = (
        occAll[occAll["year"] == LATEST_YEAR]
        .sort_values("monthly_wage", ascending=False)
        .drop_duplicates(subset="occupation")
        .reset_index(drop=True)
    )
    latest = latest[
        (latest["annual_income"] >= MIN_ANNUAL_INCOME) & (latest["annual_income"] <= MAX_ANNUAL_INCOME)
    ]

    result = latest[["occupation", "monthly_wage", "annual_bonus", "annual_income"]].copy()
    result = result.sort_values("occupation").reset_index(drop=True)

    _saveCsv(result, "occupation_list.csv")
    logger.info("  ✅ occupation_list.csv: %d 職種", len(result))
    return result


# ──────────────────────────────────────────────────────
# 2. 年齢カーブ（全職種平均・2024年基準）
# ──────────────────────────────────────────────────────
def buildAgeCurve() -> pd.DataFrame:
    ageAll = _dropInvalidOcc(_readProcessed("age_wage_all.csv"))

    # 最新年のみ使用、全職種平均
    latest = ageAll[ageAll["year"] == LATEST_YEAR]
    curve = (
        latest.groupby(["age_label", "age_mid"])
        .agg(monthly_wage=("monthly_wage", "mean"), annual_income=("annual_income", "mean"))
        .reset_index()
        .sort_values("age_mid")
    )

    # 年齢別の昇給率（隣接年齢階級との比率）
    curve["raise_rate"] = curve["annual_income"].pct_change().fillna(DEFAULT_RAISE_RATE)
    curve["raise_rate"] = curve["raise_rate"].clip(MIN_RAISE_RATE, MAX_RAISE_RATE)

    _saveCsv(curve, "age_curve.csv")
    logger.info("  ✅ age_curve.csv: %d 年齢階級", len(curve))
    logger.info("%s", curve[["age_label", "age_mid", "monthly_wage", "raise_rate"]].to_string(index=False))
    return curve


# ──────────────────────────────────────────────────────
# 3. 経験年数カーブ（全職種平均・2024年基準）
# ──────────────────────────────────────────────────────
def buildExpCurve() -> pd.DataFrame:
    expAll = _dropInvalidOcc(_readProcessed("experience_wage_all.csv"))

    latest = expAll[expAll["year"] == LATEST_YEAR]
    curve = (
        latest.groupby("experience_years")
        .agg(monthly_wage=("monthly_wage", "mean"), annual_income=("annual_income", "mean"))
        .reset_index()
        .sort_values("experience_years")
    )

    _saveCsv(curve, "exp_curve.csv")
    logger.info("  ✅ exp_curve.csv: %d 経験年数階級", len(curve))
    logger.info("%s", curve.to_string(index=False))
    return curve


# ──────────────────────────────────────────────────────
# 4. 機械学習用メインデータセット
# ──────────────────────────────────────────────────────
def _occupationRecords(occ: str, baseIncome: float,
                       occAge: pd.DataFrame, occExp: pd.DataFrame) -> list[dict]:
    """1 職種について、年齢階級 × 経験年数の合成年収レコードを作る"""
    records: list[dict] = []
    # 全組み合わせの平均年収（スケーリング基準）
    ageMean = occAge["annual_income"].mean()
    expMean = occExp["annual_income"].mean() if len(occExp) > 0 else baseIncome

    for _, ageRow in occAge.iterrows():
        ageMid = ageRow["age_mid"]
        ageRatio = ageRow["annual_income"] / ageMean if ageMean > 0 else 1.0

        for _, expRow in occExp.iterrows():
            expYears = expRow["experience_years"]
            # 経験年数が年齢的に不可能なケースを除外
            if expYears > ageMid - WORK_START_AGE:
                continue

            expRatio = expRow["annual_income"] / expMean if expMean > 0 else 1.0
            # 合成年収 = ベース × 年齢倍率 × 経験年数倍率（相乗平均で合成）
            syntheticIncome = baseIncome * np.sqrt(ageRatio * expRatio)
            noise = np.random.normal(1.0, NOISE_STD)

            records.append({
                "occupation": occ,
                "age": ageMid,
                "experience_years": expYears,
                "annual_income": round(max(syntheticIncome * noise, MIN_INCOME), 1),
            })
    return records


def buildMlDataset(occList: pd.DataFrame) -> pd.DataFrame:
    """
    年齢階級×経験年数×職種 の組み合わせで年収を計算したデータセットを作る。

    設計方針:
      - 職種別ベース年収（latest年）を基準とする
      - 年齢による倍率（age_curve基準）を乗算
      - 経験年数による倍率（exp_curve基準）を乗算
      - 全体の平均値でスケーリングして自然なレンジに調整
      - 年度推移（2006〜2024）の賃金上昇率を反映
    """
    ageAll = _dropInvalidOcc(_readProcessed("age_wage_all.csv"))
    expAll = _dropInvalidOcc(_readProcessed("experience_wage_all.csv"))
    ageLatest = ageAll[ageAll["year"] == LATEST_YEAR].copy()
    expLatest = expAll[expAll["year"] == LATEST_YEAR].copy()

    # 職種別データがない場合に使う全職種平均カーブ
    ageAverage = ageLatest.groupby(["age_label", "age_mid"]).agg(
        annual_income=("annual_income", "mean")
    ).reset_index()
    expAverage = expLatest.groupby("experience_years").agg(
        annual_income=("annual_income", "mean")
    ).reset_index()

    records: list[dict] = []
    for _, occRow in occList.iterrows():
        occ = occRow["occupation"]
        occAge = ageLatest[ageLatest["occupation"] == occ]
        occExp = expLatest[expLatest["occupation"] == occ]
        records.extend(_occupationRecords(
            occ, occRow["annual_income"],
            occAge if len(occAge) > 0 else ageAverage,
            occExp if len(occExp) > 0 else expAverage,
        ))

    df = pd.DataFrame(records)
    logger.info("\n  ML dataset: %s サンプル, %d 職種", f"{len(df):,}", df["occupation"].nunique())
    logger.info("  年収レンジ: %.0f〜%.0f 万円", df["annual_income"].min(), df["annual_income"].max())
    logger.info("  年収中央値: %.0f 万円", df["annual_income"].median())

    _saveCsv(df, "ml_dataset.csv")
    logger.info("  ✅ ml_dataset.csv: %s レコード", f"{len(df):,}")
    return df


# ──────────────────────────────────────────────────────
# 5. マクロ経済パラメータ
# ──────────────────────────────────────────────────────
def _recentMean(df: pd.DataFrame, column: str) -> float:
    """直近 RECENT_YEARS 年の平均（より現実的な推計値）"""
    return float(df[df["year"] >= LATEST_YEAR - RECENT_YEARS][column].dropna().mean())


def buildMacroParams() -> MacroParams:
    avgWageGrowth = _recentMean(_readProcessed("monthly_labor_all.csv"), "yoy_rate")
    avgGdpGrowth  = _recentMean(_readProcessed("gdp_annual.csv"), "gdp_real_growth")
    avgCpiYoy     = _recentMean(_readProcessed("cpi_annual.csv"), "cpi_yoy")

    # 実質賃金成長率 = 名目賃金成長 - インフレ率
    realWageGrowth = avgWageGrowth - avgCpiYoy

    # 将来推計: 直近実績を重視し、上下限でクリップ
    forecastNominal = float(np.clip(avgWageGrowth, MIN_FORECAST_NOMINAL, MAX_FORECAST_NOMINAL))
    forecastReal    = float(np.clip(realWageGrowth, MIN_FORECAST_REAL, MAX_FORECAST_REAL))

    params: MacroParams = {
        "latest_year":            LATEST_YEAR,
        "avg_wage_growth_10yr":   avgWageGrowth,
        "avg_gdp_growth_10yr":    avgGdpGrowth,
        "avg_cpi_10yr":           avgCpiYoy,
        "real_wage_growth_10yr":  realWageGrowth,
        "forecast_nominal_raise": forecastNominal,
        "forecast_real_raise":    forecastReal,
    }

    with open(os.path.join(OUT_DIR, "macro_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    logger.info("  ✅ macro_params.json")
    logger.info("     名目賃金上昇（10年平均）: %.2f%%", avgWageGrowth * PERCENT)
    logger.info("     実質GDP成長（10年平均）:  %.2f%%", avgGdpGrowth * PERCENT)
    logger.info("     CPI（10年平均）:          %.2f%%", avgCpiYoy * PERCENT)
    logger.info("     将来推計（名目）:          %.2f%%", forecastNominal * PERCENT)
    return params


# ──────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────
def main() -> None:
    np.random.seed(RANDOM_SEED)

    rule = "=" * SECTION_RULE_WIDTH
    logger.info("\n%s\n  Step2: processed → master  データセット構築\n%s\n", rule, rule)

    logger.info("[職種マスタ]")
    occList = buildOccupationList()
    logger.info("\n[年齢カーブ]")
    buildAgeCurve()
    logger.info("\n[経験年数カーブ]")
    buildExpCurve()
    logger.info("\n[MLデータセット構築]")
    buildMlDataset(occList)
    logger.info("\n[マクロ経済パラメータ]")
    buildMacroParams()

    logger.info("\n%s\n  Step2 完了 → data/master/\n%s\n", rule, rule)


if __name__ == "__main__":
    main()
