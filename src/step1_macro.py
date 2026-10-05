"""
step1_macro.py
==============
マクロ経済系データ（毎月勤労統計・GDP・CPI）の変換（step1_to_processed.py から分離）。
"""

from __future__ import annotations
import os, re
import pandas as pd
import numpy as np

from log_config import getLogger
from step1_common import subdir, listXlsx, safeNum, extractYear, saveCsv

logger = getLogger(__name__)

PERCENT = 100

# 毎月勤労統計: 「調査産業計」行の列位置
LABOR_TOTAL_ROW_KEY = "調査産業計"
LABOR_WAGE_COL      = 4   # きまって支給する給与（円）
LABOR_YOY_COL       = 5   # 前年比（%）

# GDP: 年次の実質成長率 CSV
GDP_FILE_NAME = "2024_年次GDP成長率_実質.csv"
GDP_ENCODING  = "shift_jis"
GDP_YEAR_COL  = 0         # 年度「YYYY/4-3.」
GDP_RATE_COL  = 1         # 実質GDP成長率（%）

# CPI: 中分類指数の年平均
CPI_FILE_NAME = "2025_消費者物価指数_中分類指数_全国__年平均.xlsx"
CPI_SKIP_ROWS = 14
CPI_YEAR_COL  = 8         # 年（例: '2020年'）
CPI_INDEX_COL = 12        # 総合指数（2020年=100）


def _logYearRange(fileName: str, df: pd.DataFrame) -> None:
    logger.info("  ✅ %s: %d 年分 (%s〜%s年)", fileName, len(df), df["year"].min(), df["year"].max())


# ══════════════════════════════════════════════════════
# 4. 毎月勤労統計調査
# ══════════════════════════════════════════════════════
def _findLaborTotal(df: pd.DataFrame, year: int) -> dict[str, float] | None:
    """「調査産業計」行の給与と前年比を返す。行がない・給与が空なら None"""
    for _, row in df.iterrows():
        cell = str(row.iloc[0]).replace("　", "").replace(" ", "")
        if LABOR_TOTAL_ROW_KEY not in cell:
            continue
        wage = safeNum(row.iloc[LABOR_WAGE_COL])
        yoy  = safeNum(row.iloc[LABOR_YOY_COL])
        if np.isnan(wage):
            return None
        return {
            "year": year,
            "scheduled_wage_yen": wage,
            "yoy_pct": yoy,
            "yoy_rate": yoy / PERCENT if not np.isnan(yoy) else np.nan,
        }
    return None


def processMonthlyLabor() -> pd.DataFrame:
    logger.info("[毎月勤労統計]")
    files = listXlsx("labor")
    logger.info("  対象: %d ファイル", len(files))
    records = []

    for path in files:
        year = extractYear(os.path.basename(path))
        try:
            record = _findLaborTotal(pd.read_excel(path, header=None, dtype=str), year)
        except Exception as e:
            logger.warning("  ⚠ %s: %s", os.path.basename(path), e)
            continue
        if record is not None:
            records.append(record)

    dfOut = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    saveCsv(dfOut, "monthly_labor_all.csv")
    logger.info("  ✅ monthly_labor_all.csv: %d 年分", len(dfOut))
    logger.info("%s", dfOut[["year", "scheduled_wage_yen", "yoy_pct"]].to_string(index=False))
    return dfOut


# ══════════════════════════════════════════════════════
# 5. GDP 実質成長率（年次）
# ══════════════════════════════════════════════════════
def processGdp() -> pd.DataFrame:
    logger.info("[GDP成長率]")
    path = os.path.join(subdir("gdp"), GDP_FILE_NAME)
    records = []
    try:
        df = pd.read_csv(path, encoding=GDP_ENCODING, header=None, dtype=str)
        for _, row in df.iterrows():
            match = re.match(r"(\d{4})/", str(row.iloc[GDP_YEAR_COL]).strip())
            if not match:
                continue
            rate = safeNum(row.iloc[GDP_RATE_COL])
            if np.isnan(rate):
                continue
            records.append({
                "year": int(match.group(1)),
                "gdp_real_growth_pct": rate,
                "gdp_real_growth": rate / PERCENT,
            })
    except Exception as e:
        logger.warning("  ⚠ GDP: %s", e)

    dfOut = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    saveCsv(dfOut, "gdp_annual.csv")
    _logYearRange("gdp_annual.csv", dfOut)
    return dfOut


# ══════════════════════════════════════════════════════
# 6. CPI 総合指数（年平均）
# ══════════════════════════════════════════════════════
def processCpi() -> pd.DataFrame:
    logger.info("[CPI]")
    path = os.path.join(subdir("cpi"), CPI_FILE_NAME)
    records = []
    try:
        df = pd.read_excel(path, header=None, skiprows=CPI_SKIP_ROWS, dtype=str)
        for _, row in df.iterrows():
            match = re.match(r"(\d{4})", str(row.iloc[CPI_YEAR_COL]).strip())
            if not match:
                continue
            cpi = safeNum(row.iloc[CPI_INDEX_COL])
            if not np.isnan(cpi) and cpi > 0:
                records.append({"year": int(match.group(1)), "cpi": cpi})
    except Exception as e:
        logger.warning("  ⚠ CPI: %s", e)

    dfOut = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    dfOut["cpi_yoy"] = dfOut["cpi"].pct_change()
    saveCsv(dfOut, "cpi_annual.csv")
    _logYearRange("cpi_annual.csv", dfOut)
    return dfOut
