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

def subdir(key: str) -> str:
    return os.path.join(RAW_DIR, _SUB[key])

def list_xlsx(key: str) -> list:
    d = subdir(key)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".xlsx"))

# ── 共通ユーティリティ ────────────────────────────────
def safe_num(v) -> float:
    try:
        s = str(v).replace(",", "").replace("，", "").strip()
        if s in {"-", "－", "…", "***", "nan", "", "x", "X", "−"}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def extract_year(fname: str) -> int:
    m = re.search(r"(\d{4})", fname)
    return int(m.group(1)) if m else 0

def clean_name(s: str) -> str:
    s = str(s).replace("\u3000", "").replace("\n", "").replace("\r", "").strip()
    s = re.sub(r"^(男女計|　男女計|男\s|女\s)", "", s).strip()
    return s

def find_data_start(df: pd.DataFrame, keyword="管理的") -> int | None:
    for i, row in df.iterrows():
        for v in row.values:
            if keyword in str(v):
                return i
    return None
