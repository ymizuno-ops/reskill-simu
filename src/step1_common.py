"""
step1_common.py
===============
step1 の各処理が共有するパス設定と共通ユーティリティ（step1_to_processed.py から分離）。
"""

from __future__ import annotations
import os, re
import pandas as pd
import numpy as np

# ── パス設定 ──────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_HERE, "..", "data", "raw")
OUT_DIR = os.path.join(_HERE, "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

_SUB = {
    "occ":   "職種別きまって支給する現金給与額",
    "age":   "年齢階級別きまって支給する現金給与額",
    "exp":   "経験年数階級別きまって支給する現金給与額",
    "labor": "毎月勤労統計調査_結果確報",
    "gdp":   "国民経済計算_GDP統計",
    "cpi":   "消費者物価指数",
}

# 統計表で「値なし」を表す記号
_MISSING_MARKS = {"-", "－", "…", "***", "nan", "", "x", "X", "−"}
DEFAULT_DATA_START_KEYWORD = "管理的"

# ── 単位・集計の共通定数 ──────────────────────────────
THOUSAND_YEN_PER_MAN_YEN = 10       # 千円 → 万円
MONTHS_PER_YEAR          = 12
DEFAULT_BONUS_MONTHS     = 2        # 賞与が欠損のとき月収の何か月分とみなすか
NEW_FORMAT_FROM_YEAR     = 2020     # この年以降は新形式の表
MIN_VALID_WAGE           = 10       # これ以下（千円）の給与は集計行・異常値とみなす
MIN_NAME_LENGTH          = 2
UNDER19_AGE_MID          = 18.0
UNDER19_AGE_LABEL        = "〜19歳"
DEEP_INDENT              = "　　　"  # 新形式で内訳行を表すインデント
CSV_ENCODING             = "utf-8-sig"


def subdir(key: str) -> str:
    return os.path.join(RAW_DIR, _SUB[key])


def listXlsx(key: str) -> list[str]:
    dirPath = subdir(key)
    return sorted(os.path.join(dirPath, f) for f in os.listdir(dirPath) if f.endswith(".xlsx"))


# ── 共通ユーティリティ ────────────────────────────────
def safeNum(value: object) -> float:
    """統計表のセル値を数値にする。値なしの記号や数値でない文字列は NaN を返す"""
    text = str(value).replace(",", "").replace("，", "").strip()
    if text in _MISSING_MARKS:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def extractYear(fileName: str) -> int:
    match = re.search(r"(\d{4})", fileName)
    return int(match.group(1)) if match else 0


def cleanName(text: str) -> str:
    name = str(text).replace("　", "").replace("\n", "").replace("\r", "").strip()
    return re.sub(r"^(男女計|　男女計|男\s|女\s)", "", name).strip()


def findDataStart(df: pd.DataFrame, keyword: str = DEFAULT_DATA_START_KEYWORD) -> int | None:
    for i, row in df.iterrows():
        for value in row.values:
            if keyword in str(value):
                return i
    return None


def toManYen(thousandYen: float) -> float:
    """千円 → 万円。NaN はそのまま NaN を返す"""
    return thousandYen / THOUSAND_YEN_PER_MAN_YEN if not np.isnan(thousandYen) else np.nan


def addAnnualIncome(df: pd.DataFrame, bonusMonths: float = DEFAULT_BONUS_MONTHS) -> pd.DataFrame:
    """年収 = 月収 × 12 + 賞与（賞与が欠損なら月収 × bonusMonths）"""
    df["annual_income"] = (
        df["monthly_wage"] * MONTHS_PER_YEAR
        + df["annual_bonus"].fillna(df["monthly_wage"] * bonusMonths)
    )
    return df


def saveCsv(df: pd.DataFrame, fileName: str) -> None:
    df.to_csv(os.path.join(OUT_DIR, fileName), index=False, encoding=CSV_ENCODING)
