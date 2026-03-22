"""
step1_to_processed.py
=====================
data/raw/ 以下の各サブディレクトリにある xlsx を読み込み、
data/processed/ に整形済み CSV として出力する。

旧形式（〜2019年）と新形式（2020年〜）でヘッダー行・列位置が異なるため
それぞれ個別のパーサーを実装している。

出力ファイル一覧:
  processed/
    occupation_wage_all.csv    # 職種別給与（年×職種）
    age_wage_all.csv           # 年齢階級×職種×年別給与
    experience_wage_all.csv    # 経験年数×職種×年別給与
    monthly_labor_all.csv      # 毎月勤労統計（年別きまって支給前年比）
    gdp_annual.csv             # 実質GDP成長率（年次）
    cpi_annual.csv             # CPI総合指数（年平均）
"""

from __future__ import annotations
import os, re, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

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


# ══════════════════════════════════════════════════════
# 1. 職種別給与（産業計）
# ══════════════════════════════════════════════════════
# 列レイアウト（旧新共通）:
#   col1 = 職種名
#   col7 = きまって支給する現金給与額（千円）
#   col9 = 年間賞与その他特別給与額（千円）
def process_occupation_wage() -> pd.DataFrame:
    print("[職種別給与]")
    files = list_xlsx("occ")
    print(f"  対象: {len(files)} ファイル")
    records = []

    for path in files:
        year = extract_year(os.path.basename(path))
        try:
            df = pd.read_excel(path, header=None, dtype=str)
            data_start = find_data_start(df)
            if data_start is None:
                continue

            for i in range(data_start, len(df)):
                row  = df.iloc[i]
                raw  = str(row.iloc[1])
                name = clean_name(raw)

                if (not name or name == "nan" or len(name) < 2
                        or re.search(r"\d+\s*[～~]\s*\d+", name)
                        or "１９歳" in name or "19歳" in name):
                    continue

                wage  = safe_num(row.iloc[7])
                bonus = safe_num(row.iloc[9])
                if np.isnan(wage) or wage <= 10:
                    continue

                records.append({
                    "year": year, "occupation": name,
                    "monthly_wage": wage / 10,
                    "annual_bonus": bonus / 10 if not np.isnan(bonus) else np.nan,
                })
        except Exception as e:
            print(f"  ⚠ {os.path.basename(path)}: {e}")

    df_out = pd.DataFrame(records)
    df_out["annual_income"] = (
        df_out["monthly_wage"] * 12
        + df_out["annual_bonus"].fillna(df_out["monthly_wage"] * 2)
    )
    df_out.to_csv(os.path.join(OUT_DIR, "occupation_wage_all.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ occupation_wage_all.csv: {len(df_out):,} レコード "
          f"({df_out['year'].min()}〜{df_out['year'].max()}年, {df_out['occupation'].nunique()} 職種)")
    return df_out


# ══════════════════════════════════════════════════════
# 2. 年齢階級別給与
# ══════════════════════════════════════════════════════
def _parse_age_new(df: pd.DataFrame, year: int) -> list:
    """2020〜: col1=職種名/年齢, col7=給与, col9=賞与"""
    records, current_occ = [], None
    data_start = find_data_start(df)
    if data_start is None:
        return records

    for i in range(data_start, len(df)):
        row  = df.iloc[i]
        raw  = str(row.iloc[1]).replace("\n", "")
        name = raw.replace("\u3000", "").strip()

        age_m  = re.search(r"(\d+)\s*[～~]\s*(\d+)", name)
        is_u19 = "１９歳" in name or "19歳" in name

        if age_m or is_u19:
            if current_occ is None:
                continue
            wage  = safe_num(row.iloc[7])
            bonus = safe_num(row.iloc[9])
            if np.isnan(wage) or wage <= 0:
                continue
            if is_u19:
                age_mid, age_label = 18.0, "〜19歳"
            else:
                a1, a2 = int(age_m.group(1)), int(age_m.group(2))
                age_mid, age_label = (a1 + a2) / 2, f"{a1}〜{a2}歳"
            records.append({
                "year": year, "occupation": current_occ,
                "age_label": age_label, "age_mid": age_mid,
                "monthly_wage": wage / 10,
                "annual_bonus": bonus / 10 if not np.isnan(bonus) else np.nan,
            })
        elif name and name != "nan" and len(name) > 1:
            if not raw.startswith("\u3000\u3000\u3000"):
                current_occ = clean_name(name)
    return records


def _parse_age_old(df: pd.DataFrame, year: int) -> list:
    """
    〜2019: col0=職種名(男)/年齢階級, col5=給与, col7=賞与
    ※ 旧形式は産業計ではなく性別ファイルのため、男女計のデータはない。
      「職種名(男)」行の総計値（年齢小計）を使用する。
    """
    records, current_occ = [], None
    data_start = None
    for i, row in df.iterrows():
        v = str(row.iloc[0]).replace("\u3000", "").strip()
        if (len(v) > 2 and "区" not in v and "nan" not in v
                and not v.startswith("第") and not re.match(r"^\d", v)
                and "歳" not in v and "〜" not in v and "～" not in v):
            data_start = i
            break
    if data_start is None:
        return records

    for i in range(data_start, len(df)):
        row  = df.iloc[i]
        raw0 = str(row.iloc[0]).replace("\u3000", "").strip()

        age_m  = re.search(r"(\d+)\s*[～~\s]+\s*(\d+)", raw0)
        is_u19 = "17歳" in raw0 or "19歳" in raw0 or "18　～　19" in raw0

        if age_m or is_u19:
            if current_occ is None:
                continue
            wage  = safe_num(row.iloc[5])
            bonus = safe_num(row.iloc[7])
            if np.isnan(wage) or wage <= 0:
                continue
            if is_u19:
                age_mid, age_label = 18.0, "〜19歳"
            else:
                a1, a2 = int(age_m.group(1)), int(age_m.group(2))
                age_mid, age_label = (a1 + a2) / 2, f"{a1}〜{a2}歳"
            records.append({
                "year": year, "occupation": current_occ,
                "age_label": age_label, "age_mid": age_mid,
                "monthly_wage": wage / 10,
                "annual_bonus": bonus / 10 if not np.isnan(bonus) else np.nan,
            })
        elif raw0 and raw0 != "nan" and len(raw0) > 2:
            occ = re.sub(r"\s*\(.*?\)\s*$", "", raw0).strip()
            if occ and not re.match(r"^\d", occ):
                current_occ = occ
    return records


def process_age_wage() -> pd.DataFrame:
    print("[年齢階級別給与]")
    files = list_xlsx("age")
    print(f"  対象: {len(files)} ファイル")
    records = []

    for path in files:
        year = extract_year(os.path.basename(path))
        try:
            df = pd.read_excel(path, header=None, dtype=str)
            records.extend(_parse_age_new(df, year) if year >= 2020 else _parse_age_old(df, year))
        except Exception as e:
            print(f"  ⚠ {os.path.basename(path)}: {e}")

    df_out = pd.DataFrame(records)
    df_out["annual_income"] = (
        df_out["monthly_wage"] * 12
        + df_out["annual_bonus"].fillna(df_out["monthly_wage"] * 2)
    )
    df_out.to_csv(os.path.join(OUT_DIR, "age_wage_all.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ age_wage_all.csv: {len(df_out):,} レコード "
          f"({df_out['year'].min()}〜{df_out['year'].max()}年)")
    return df_out


# ══════════════════════════════════════════════════════
# 3. 経験年数階級別給与
# ══════════════════════════════════════════════════════
def _get_exp_col_map(df: pd.DataFrame) -> dict:
    """
    ヘッダー行を走査して 経験年数バンド→列インデックス のマップを返す。
    新形式(2020〜): 0年/1〜4年/5〜9年/10〜14年/15年以上  (5バンド)
    旧形式(〜2019): 0年/1〜4年/5〜9年/10〜14年/15〜19年/20年以上 (6バンド)
    """
    patterns = [
        ("total", ["経験年数計"]),
        (0.0,  ["０年", "0年"]),
        (2.5,  ["１～４年", "1～4年", "1 ～ 4 年"]),
        (7.0,  ["５～９年", "5～9年", "5 ～ 9 年"]),
        (12.0, ["１０～１４年", "10～14年"]),
        (17.0, ["１５～１９年", "15～19年", "１５年以上", "15年以上"]),
        (22.0, ["２０年以上", "20年以上"]),
    ]
    col_map = {}
    for _, row in df.head(10).iterrows():
        for col_idx, val in enumerate(row):
            v = str(val).replace("\u3000", "").replace(" ", "").strip()
            for key, labels in patterns:
                if key not in col_map:
                    if any(lbl.replace(" ", "") in v for lbl in labels):
                        col_map[key] = col_idx
    return col_map


def _parse_exp(df: pd.DataFrame, year: int, col_map: dict, new_fmt: bool) -> list:
    records = []
    total_col  = col_map.get("total", 3 if new_fmt else 1)
    name_col   = 1 if new_fmt else 0
    data_start = find_data_start(df) if new_fmt else None

    if not new_fmt:
        for i, row in df.iterrows():
            v = str(row.iloc[0]).replace("\u3000", "").strip()
            if (len(v) > 2 and "区" not in v and "nan" not in v
                    and not v.startswith("第") and not re.match(r"^\d", v)
                    and "歳" not in v):
                data_start = i
                break

    if data_start is None:
        return records

    for i in range(data_start, len(df)):
        row = df.iloc[i]
        raw  = str(row.iloc[name_col]).replace("\n", "")
        name = raw.replace("\u3000", "").strip()

        # 年齢行スキップ
        if (re.search(r"\d+\s*[～~\s]+\s*\d+", name)
                or "17歳" in name or "19歳" in name or "１９歳" in name):
            continue
        # 深いインデントスキップ（新形式）
        if new_fmt and raw.startswith("\u3000\u3000\u3000"):
            continue

        if not name or name == "nan" or len(name) < 2:
            continue

        # 旧形式: 職種名末尾の(男)除去
        if not new_fmt:
            name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
        occ = clean_name(name)

        total_w = safe_num(row.iloc[total_col])
        if np.isnan(total_w) or total_w <= 10:
            continue

        for exp_yr, col in col_map.items():
            if exp_yr == "total":
                continue
            if col >= len(row):
                continue
            w = safe_num(row.iloc[col])
            b = safe_num(row.iloc[col + 1]) if col + 1 < len(row) else np.nan
            if not np.isnan(w) and w > 0:
                records.append({
                    "year": year, "occupation": occ,
                    "experience_years": float(exp_yr),
                    "monthly_wage": w / 10,
                    "annual_bonus": b / 10 if not np.isnan(b) else np.nan,
                })
    return records


def process_experience_wage() -> pd.DataFrame:
    print("[経験年数別給与]")
    files = list_xlsx("exp")
    print(f"  対象: {len(files)} ファイル")
    records = []

    for path in files:
        year = extract_year(os.path.basename(path))
        try:
            df      = pd.read_excel(path, header=None, dtype=str)
            col_map = _get_exp_col_map(df)
            if not col_map:
                print(f"  ⚠ 列マップ取得失敗: {os.path.basename(path)}")
                continue
            records.extend(_parse_exp(df, year, col_map, new_fmt=(year >= 2020)))
        except Exception as e:
            print(f"  ⚠ {os.path.basename(path)}: {e}")

    df_out = pd.DataFrame(records)
    df_out["annual_income"] = (
        df_out["monthly_wage"] * 12
        + df_out["annual_bonus"].fillna(df_out["monthly_wage"] * 1.5)
    )
    df_out.to_csv(os.path.join(OUT_DIR, "experience_wage_all.csv"), index=False, encoding="utf-8-sig")
    print(f"  ✅ experience_wage_all.csv: {len(df_out):,} レコード "
          f"({df_out['year'].min()}〜{df_out['year'].max()}年)")
    return df_out


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


# ══════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  Step1: data/raw → data/processed  変換開始")
    print("=" * 60 + "\n")

    process_occupation_wage();  print()
    process_age_wage();         print()
    process_experience_wage();  print()
    process_monthly_labor();    print()
    process_gdp();              print()
    process_cpi()

    print("\n" + "=" * 60)
    print("  Step1 完了 → data/processed/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
