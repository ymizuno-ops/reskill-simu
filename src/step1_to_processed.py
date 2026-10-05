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
from collections.abc import Callable
import pandas as pd
import numpy as np

from log_config import getLogger
from step1_common import (
    listXlsx, safeNum, extractYear, cleanName, findDataStart, toManYen,
    addAnnualIncome, saveCsv, MIN_VALID_WAGE, MIN_NAME_LENGTH, NEW_FORMAT_FROM_YEAR,
    UNDER19_AGE_MID, UNDER19_AGE_LABEL, DEEP_INDENT,
)
from step1_macro import processMonthlyLabor, processGdp, processCpi

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

# 職種別給与（旧新共通）の列位置
# 意味: 職種別の表で、職種名・月給・賞与が入っている列の番号（左端の列が 0）。
# 注意: 表の様式が変わったら Excel で列を数え直して合わせる。
OCC_NAME_COL  = 1
OCC_WAGE_COL  = 7   # きまって支給する現金給与額（千円）
OCC_BONUS_COL = 9   # 年間賞与その他特別給与額（千円）

# 年齢階級別給与の列位置
# 意味: 年齢階級別の表で、名前・月給・賞与が入っている列の番号（上の行が2020年以降の新形式、下の行が2019年以前の旧形式）。
# 注意: 様式が変わったら Excel で列を数え直して合わせる。
AGE_NEW_NAME_COL, AGE_NEW_WAGE_COL, AGE_NEW_BONUS_COL = 1, 7, 9
AGE_OLD_NAME_COL, AGE_OLD_WAGE_COL, AGE_OLD_BONUS_COL = 0, 5, 7

# 経験年数階級別給与
# 意味: 経験年数別の表で、「０年」「１～４年」などの見出しを探す先頭からの行数。
# 影響: 見出しがこれより下にあると列が見つからず、そのファイルを読み飛ばす（ログに「列マップ取得失敗」と出る）。
EXP_HEADER_ROWS       = 10   # 列マップを探すヘッダーの行数
# 意味: 経験年数別の表で賞与が空欄のとき、賞与を月収の何か月分とみなすか。
# 影響: 大きくすると経験年数カーブの年収が上がる。注意: 職種別・年齢別の値（step1_common.py の DEFAULT_BONUS_MONTHS = 2）とは別に設定している。
EXP_BONUS_MONTHS      = 1.5  # 賞与が欠損のとき月収の何か月分とみなすか
EXP_TOTAL_KEY         = "total"
# 意味: 経験年数別の表で職種名と「経験年数計」がある列の番号（上の行が新形式、下の行が旧形式）。
# 注意: 「経験年数計」の列は見出しから自動で探し、見つからないときだけこの番号を使う。
EXP_NEW_NAME_COL, EXP_NEW_TOTAL_COL = 1, 3
EXP_OLD_NAME_COL, EXP_OLD_TOTAL_COL = 0, 1

# 意味: 実行ログの区切り線（=）の長さ。見た目だけで、処理結果は変わらない。
SECTION_RULE_WIDTH = 60


def _readFiles(key: str, parse: Callable[[pd.DataFrame, int], list[dict]]) -> list[dict]:
    """key のファイルを順に読み、parse(df, year) の結果をまとめる。読めないファイルは警告して飛ばす"""
    files = listXlsx(key)
    logger.info("  対象: %d ファイル", len(files))
    records: list[dict] = []
    for path in files:
        year = extractYear(os.path.basename(path))
        try:
            records.extend(parse(pd.read_excel(path, header=None, dtype=str), year))
        except Exception as e:
            logger.warning("  ⚠ %s: %s", os.path.basename(path), e)
    return records


def _isAgeRowName(name: str) -> bool:
    return bool(re.search(r"\d+\s*[～~]\s*\d+", name)) or "１９歳" in name or "19歳" in name


def _findOldDataStart(df: pd.DataFrame, shouldExcludeTilde: bool) -> int | None:
    """旧形式: 職種名らしい最初の行（見出し・番号・年齢行を除く）を探す"""
    for i, row in df.iterrows():
        v = str(row.iloc[0]).replace("　", "").strip()
        if (len(v) > MIN_NAME_LENGTH and "区" not in v and "nan" not in v
                and not v.startswith("第") and not re.match(r"^\d", v)
                and "歳" not in v
                and not (shouldExcludeTilde and ("〜" in v or "～" in v))):
            return i
    return None


# ══════════════════════════════════════════════════════
# 1. 職種別給与（産業計）
# ══════════════════════════════════════════════════════
def _parseOccupation(df: pd.DataFrame, year: int) -> list[dict]:
    records: list[dict] = []
    dataStart = findDataStart(df)
    if dataStart is None:
        return records

    for i in range(dataStart, len(df)):
        row  = df.iloc[i]
        name = cleanName(str(row.iloc[OCC_NAME_COL]))
        if not name or name == "nan" or len(name) < MIN_NAME_LENGTH or _isAgeRowName(name):
            continue

        wage = safeNum(row.iloc[OCC_WAGE_COL])
        if np.isnan(wage) or wage <= MIN_VALID_WAGE:
            continue
        records.append({
            "year": year, "occupation": name,
            "monthly_wage": toManYen(wage),
            "annual_bonus": toManYen(safeNum(row.iloc[OCC_BONUS_COL])),
        })
    return records


def processOccupationWage() -> pd.DataFrame:
    logger.info("[職種別給与]")
    dfOut = addAnnualIncome(pd.DataFrame(_readFiles("occ", _parseOccupation)))
    saveCsv(dfOut, "occupation_wage_all.csv")
    logger.info("  ✅ occupation_wage_all.csv: %s レコード (%s〜%s年, %d 職種)",
                f"{len(dfOut):,}", dfOut["year"].min(), dfOut["year"].max(), dfOut["occupation"].nunique())
    return dfOut


# ══════════════════════════════════════════════════════
# 2. 年齢階級別給与
# ══════════════════════════════════════════════════════
def _ageRecord(year: int, occ: str, ageMatch: re.Match | None,
               wage: float, bonus: float) -> dict:
    """年齢行の値から 1 レコードを作る。ageMatch が None なら 19 歳以下の行"""
    if ageMatch is None:
        ageMid, ageLabel = UNDER19_AGE_MID, UNDER19_AGE_LABEL
    else:
        ageFrom, ageTo = int(ageMatch.group(1)), int(ageMatch.group(2))
        ageMid, ageLabel = (ageFrom + ageTo) / 2, f"{ageFrom}〜{ageTo}歳"
    return {
        "year": year, "occupation": occ,
        "age_label": ageLabel, "age_mid": ageMid,
        "monthly_wage": toManYen(wage),
        "annual_bonus": toManYen(bonus),
    }


def _parseAgeNew(df: pd.DataFrame, year: int) -> list[dict]:
    """2020〜: col1=職種名/年齢, col7=給与, col9=賞与"""
    records: list[dict] = []
    currentOcc: str | None = None
    dataStart = findDataStart(df)
    if dataStart is None:
        return records

    for i in range(dataStart, len(df)):
        row  = df.iloc[i]
        raw  = str(row.iloc[AGE_NEW_NAME_COL]).replace("\n", "")
        name = raw.replace("　", "").strip()

        ageMatch = re.search(r"(\d+)\s*[～~]\s*(\d+)", name)
        isUnder19 = "１９歳" in name or "19歳" in name

        if ageMatch or isUnder19:
            if currentOcc is None:
                continue
            wage = safeNum(row.iloc[AGE_NEW_WAGE_COL])
            if np.isnan(wage) or wage <= 0:
                continue
            records.append(_ageRecord(year, currentOcc, None if isUnder19 else ageMatch,
                                      wage, safeNum(row.iloc[AGE_NEW_BONUS_COL])))
        elif name and name != "nan" and len(name) > 1 and not raw.startswith(DEEP_INDENT):
            currentOcc = cleanName(name)
    return records


def _parseAgeOld(df: pd.DataFrame, year: int) -> list[dict]:
    """
    〜2019: col0=職種名(男)/年齢階級, col5=給与, col7=賞与
    ※ 旧形式は産業計ではなく性別ファイルのため、男女計のデータはない。
      「職種名(男)」行の総計値（年齢小計）を使用する。
    """
    records: list[dict] = []
    currentOcc: str | None = None
    dataStart = _findOldDataStart(df, shouldExcludeTilde=True)
    if dataStart is None:
        return records

    for i in range(dataStart, len(df)):
        row  = df.iloc[i]
        raw0 = str(row.iloc[AGE_OLD_NAME_COL]).replace("　", "").strip()

        ageMatch = re.search(r"(\d+)\s*[～~\s]+\s*(\d+)", raw0)
        isUnder19 = "17歳" in raw0 or "19歳" in raw0 or "18　～　19" in raw0

        if ageMatch or isUnder19:
            if currentOcc is None:
                continue
            wage = safeNum(row.iloc[AGE_OLD_WAGE_COL])
            if np.isnan(wage) or wage <= 0:
                continue
            records.append(_ageRecord(year, currentOcc, None if isUnder19 else ageMatch,
                                      wage, safeNum(row.iloc[AGE_OLD_BONUS_COL])))
        elif raw0 and raw0 != "nan" and len(raw0) > MIN_NAME_LENGTH:
            occ = re.sub(r"\s*\(.*?\)\s*$", "", raw0).strip()
            if occ and not re.match(r"^\d", occ):
                currentOcc = occ
    return records


def _parseAge(df: pd.DataFrame, year: int) -> list[dict]:
    return _parseAgeNew(df, year) if year >= NEW_FORMAT_FROM_YEAR else _parseAgeOld(df, year)


def processAgeWage() -> pd.DataFrame:
    logger.info("[年齢階級別給与]")
    dfOut = addAnnualIncome(pd.DataFrame(_readFiles("age", _parseAge)))
    saveCsv(dfOut, "age_wage_all.csv")
    logger.info("  ✅ age_wage_all.csv: %s レコード (%s〜%s年)",
                f"{len(dfOut):,}", dfOut["year"].min(), dfOut["year"].max())
    return dfOut


# ══════════════════════════════════════════════════════
# 3. 経験年数階級別給与
# ══════════════════════════════════════════════════════
# 意味: 経験年数の見出しの表記と、それを何年として扱うか（各階級の真ん中の年数）。
# 影響: 左の数値を変えると、経験年数カーブと学習データの経験年数がずれ、予測年収が変わる。
# 注意: 見出しの表記が変わったら、右側の一覧に新しい表記を追加する。
_EXP_PATTERNS: list[tuple[str | float, list[str]]] = [
    (EXP_TOTAL_KEY, ["経験年数計"]),
    (0.0,  ["０年", "0年"]),
    (2.5,  ["１～４年", "1～4年", "1 ～ 4 年"]),
    (7.0,  ["５～９年", "5～9年", "5 ～ 9 年"]),
    (12.0, ["１０～１４年", "10～14年"]),
    (17.0, ["１５～１９年", "15～19年", "１５年以上", "15年以上"]),
    (22.0, ["２０年以上", "20年以上"]),
]


def _getExpColMap(df: pd.DataFrame) -> dict[str | float, int]:
    """
    ヘッダー行を走査して 経験年数バンド→列インデックス のマップを返す。
    新形式(2020〜): 0年/1〜4年/5〜9年/10〜14年/15年以上  (5バンド)
    旧形式(〜2019): 0年/1〜4年/5〜9年/10〜14年/15〜19年/20年以上 (6バンド)
    """
    colMap: dict[str | float, int] = {}
    for _, row in df.head(EXP_HEADER_ROWS).iterrows():
        for colIdx, val in enumerate(row):
            v = str(val).replace("　", "").replace(" ", "").strip()
            for key, labels in _EXP_PATTERNS:
                if key not in colMap and any(lbl.replace(" ", "") in v for lbl in labels):
                    colMap[key] = colIdx
    return colMap


def _expRecords(row: pd.Series, year: int, occ: str, colMap: dict[str | float, int]) -> list[dict]:
    """1 職種の行から、経験年数バンドごとのレコードを作る（給与・賞与は隣り合う列）"""
    records: list[dict] = []
    for expYears, col in colMap.items():
        if expYears == EXP_TOTAL_KEY or col >= len(row):
            continue
        wage  = safeNum(row.iloc[col])
        bonus = safeNum(row.iloc[col + 1]) if col + 1 < len(row) else np.nan
        if np.isnan(wage) or wage <= 0:
            continue
        records.append({
            "year": year, "occupation": occ,
            "experience_years": float(expYears),
            "monthly_wage": toManYen(wage),
            "annual_bonus": toManYen(bonus),
        })
    return records


def _parseExp(df: pd.DataFrame, year: int, colMap: dict[str | float, int], isNewFormat: bool) -> list[dict]:
    records: list[dict] = []
    totalCol  = colMap.get(EXP_TOTAL_KEY, EXP_NEW_TOTAL_COL if isNewFormat else EXP_OLD_TOTAL_COL)
    nameCol   = EXP_NEW_NAME_COL if isNewFormat else EXP_OLD_NAME_COL
    dataStart = findDataStart(df) if isNewFormat else _findOldDataStart(df, shouldExcludeTilde=False)
    if dataStart is None:
        return records

    for i in range(dataStart, len(df)):
        row  = df.iloc[i]
        raw  = str(row.iloc[nameCol]).replace("\n", "")
        name = raw.replace("　", "").strip()

        # 年齢行スキップ
        if re.search(r"\d+\s*[～~\s]+\s*\d+", name) or "17歳" in name or "19歳" in name or "１９歳" in name:
            continue
        # 深いインデントスキップ（新形式）
        if isNewFormat and raw.startswith(DEEP_INDENT):
            continue
        if not name or name == "nan" or len(name) < MIN_NAME_LENGTH:
            continue

        # 旧形式: 職種名末尾の(男)除去
        if not isNewFormat:
            name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()

        totalWage = safeNum(row.iloc[totalCol])
        if np.isnan(totalWage) or totalWage <= MIN_VALID_WAGE:
            continue
        records.extend(_expRecords(row, year, cleanName(name), colMap))
    return records


def _parseExpFile(df: pd.DataFrame, year: int) -> list[dict]:
    colMap = _getExpColMap(df)
    if not colMap:
        logger.warning("  ⚠ 列マップ取得失敗: %d 年のファイル", year)
        return []
    return _parseExp(df, year, colMap, isNewFormat=(year >= NEW_FORMAT_FROM_YEAR))


def processExperienceWage() -> pd.DataFrame:
    logger.info("[経験年数別給与]")
    dfOut = addAnnualIncome(pd.DataFrame(_readFiles("exp", _parseExpFile)), bonusMonths=EXP_BONUS_MONTHS)
    saveCsv(dfOut, "experience_wage_all.csv")
    logger.info("  ✅ experience_wage_all.csv: %s レコード (%s〜%s年)",
                f"{len(dfOut):,}", dfOut["year"].min(), dfOut["year"].max())
    return dfOut


# ══════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════
def main() -> None:
    rule = "=" * SECTION_RULE_WIDTH
    logger.info("\n%s\n  Step1: data/raw → data/processed  変換開始\n%s\n", rule, rule)

    for process in (processOccupationWage, processAgeWage, processExperienceWage,
                    processMonthlyLabor, processGdp):
        process()
        logger.info("")
    processCpi()

    logger.info("\n%s\n  Step1 完了 → data/processed/\n%s\n", rule, rule)


if __name__ == "__main__":
    main()
