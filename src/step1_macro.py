"""
step1_macro.py
==============
マクロ経済系データ（毎月勤労統計・GDP・CPI）の変換（step1_to_processed.py から分離）。
"""

from __future__ import annotations
import os, re
import pandas as pd
import numpy as np

from step1_common import OUT_DIR, subdir, list_xlsx, safe_num, extract_year


# ══════════════════════════════════════════════════════
# 4. 毎月勤労統計調査
# ══════════════════════════════════════════════════════
def process_monthly_labor() -> pd.DataFrame:
    """
    「調査産業計」行から:
      col4 = きまって支給する給与（円）
      col5 = 前年比（%）
    """
    print("[毎月勤労統計]")
    files = list_xlsx("labor")
    print(f"  対象: {len(files)} ファイル")
    records = []

    for path in files:
        year = extract_year(os.path.basename(path))
        try:
            df = pd.read_excel(path, header=None, dtype=str)
            for _, row in df.iterrows():
                cell = str(row.iloc[0]).replace("\u3000", "").replace(" ", "")
                if "調査産業計" in cell:
                    wage = safe_num(row.iloc[4])
                    yoy  = safe_num(row.iloc[5])
                    if not np.isnan(wage):
                        records.append({
                            "year": year,
                            "scheduled_wage_yen": wage,
                            "yoy_pct": yoy,
                            "yoy_rate": yoy / 100 if not np.isnan(yoy) else np.nan,
                        })
                    break
        except Exception as e:
            print(f"  ⚠ {os.path.basename(path)}: {e}")

    df_out = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    df_out.to_csv(os.path.join(OUT_DIR, "monthly_labor_all.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ monthly_labor_all.csv: {len(df_out)} 年分")
    print(df_out[["year", "scheduled_wage_yen", "yoy_pct"]].to_string(index=False))
    return df_out


# ══════════════════════════════════════════════════════
# 5. GDP 実質成長率（年次）
# ══════════════════════════════════════════════════════
def process_gdp() -> pd.DataFrame:
    """
    2024_年次GDP成長率_実質.csv (shift_jis)
    col0 = 年度「YYYY/4-3.」, col1 = 実質GDP成長率（%）
    """
    print("[GDP成長率]")
    path = os.path.join(subdir("gdp"), "2024_年次GDP成長率_実質.csv")
    records = []
    try:
        df = pd.read_csv(path, encoding="shift_jis", header=None, dtype=str)
        for _, row in df.iterrows():
            fy = str(row.iloc[0]).strip()
            m  = re.match(r"(\d{4})/", fy)
            if not m:
                continue
            rate = safe_num(row.iloc[1])
            if not np.isnan(rate):
                records.append({
                    "year": int(m.group(1)),
                    "gdp_real_growth_pct": rate,
                    "gdp_real_growth": rate / 100,
                })
    except Exception as e:
        print(f"  ⚠ GDP: {e}")

    df_out = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    df_out.to_csv(os.path.join(OUT_DIR, "gdp_annual.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ gdp_annual.csv: {len(df_out)} 年分 ({df_out['year'].min()}〜{df_out['year'].max()}年)")
    return df_out


# ══════════════════════════════════════════════════════
# 6. CPI 総合指数（年平均）
# ══════════════════════════════════════════════════════
def process_cpi() -> pd.DataFrame:
    """
    2025_消費者物価指数_中分類指数_全国__年平均.xlsx
    skiprows=14 後:
      col8  = 年（例: '2020年'）
      col12 = 総合指数（2020年=100）
    """
    print("[CPI]")
    path = os.path.join(subdir("cpi"), "2025_消費者物価指数_中分類指数_全国__年平均.xlsx")
    records = []
    try:
        df = pd.read_excel(path, header=None, skiprows=14, dtype=str)
        for _, row in df.iterrows():
            m = re.match(r"(\d{4})", str(row.iloc[8]).strip())
            if not m:
                continue
            cpi = safe_num(row.iloc[12])
            if not np.isnan(cpi) and cpi > 0:
                records.append({"year": int(m.group(1)), "cpi": cpi})
    except Exception as e:
        print(f"  ⚠ CPI: {e}")

    df_out = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    df_out["cpi_yoy"] = df_out["cpi"].pct_change()
    df_out.to_csv(os.path.join(OUT_DIR, "cpi_annual.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ cpi_annual.csv: {len(df_out)} 年分 ({df_out['year'].min()}〜{df_out['year'].max()}年)")
    return df_out
