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

# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# warnings: Python の警告メッセージを制御する標準モジュール。
import os, re, warnings
# Callable: 「呼び出せるもの（関数）」を表す型ヒント。
from collections.abc import Callable
import pandas as pd
import numpy as np

from log_config import getLogger
# 括弧で囲むと、読み込む名前を複数行に分けて書ける。
from step1_common import (
    listXlsx, safeNum, extractYear, cleanName, findDataStart, toManYen,
    addAnnualIncome, saveCsv, MIN_VALID_WAGE, MIN_NAME_LENGTH, NEW_FORMAT_FROM_YEAR,
    UNDER19_AGE_MID, UNDER19_AGE_LABEL, DEEP_INDENT,
)
from step1_macro import processMonthlyLabor, processGdp, processCpi

# ライブラリが出す警告（Excel の書式に関する警告など）をすべて表示しない。
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


# 引数 parse には「関数」を渡せる。Callable[[引数の型, ...], 戻り値の型] は「DataFrame と int を受け取り、辞書のリストを返す関数」という意味。
def _readFiles(key: str, parse: Callable[[pd.DataFrame, int], list[dict]]) -> list[dict]:
    """key のファイルを順に読み、parse(df, year) の結果をまとめる。読めないファイルは警告して飛ばす"""
    files = listXlsx(key)
    logger.info("  対象: %d ファイル", len(files))
    # 変数にも型ヒントを付けられる（ここでは辞書のリスト）。
    records: list[dict] = []
    for path in files:
        year = extractYear(os.path.basename(path))
        try:
            # 渡された関数 parse を呼び、その結果（リスト）を extend で records の末尾にまとめて追加する。append だとリストごと1つの要素として入ってしまう。
            records.extend(parse(pd.read_excel(path, header=None, dtype=str), year))
        except Exception as e:
            logger.warning("  ⚠ %s: %s", os.path.basename(path), e)
    return records


# 名前が年齢の行（「20～24歳」など）かどうかを True/False で返す。
def _isAgeRowName(name: str) -> bool:
    # re.search は見つかれば一致の情報、なければ None を返す。bool() で True/False に直す。or はどれか1つでも真なら真。
    return bool(re.search(r"\d+\s*[～~]\s*\d+", name)) or "１９歳" in name or "19歳" in name


# 旧形式の表でデータが始まる行を探す。shouldExcludeTilde で「〜」を含む行を除くかどうかを切り替える。
def _findOldDataStart(df: pd.DataFrame, shouldExcludeTilde: bool) -> int | None:
    """旧形式: 職種名らしい最初の行（見出し・番号・年齢行を除く）を探す"""
    for i, row in df.iterrows():
        v = str(row.iloc[0]).replace("　", "").strip()
        # 条件を括弧で囲むと複数行に分けて書ける。len は文字数、startswith は先頭が一致するか、not は否定。
        if (len(v) > MIN_NAME_LENGTH and "区" not in v and "nan" not in v
                and not v.startswith("第") and not re.match(r"^\d", v)
                and "歳" not in v
                and not (shouldExcludeTilde and ("〜" in v or "～" in v))):
            return i
    return None


# ══════════════════════════════════════════════════════
# 1. 職種別給与（産業計）
# ══════════════════════════════════════════════════════
# 職種別の表1ファイル分から、職種ごとの月給・賞与のレコードを作る。
def _parseOccupation(df: pd.DataFrame, year: int) -> list[dict]:
    records: list[dict] = []
    # データ本体が始まる行。見つからなければ、次の if で空のリストを返して終わる（早期リターン）。
    dataStart = findDataStart(df)
    if dataStart is None:
        return records

    # range(開始, 終了) は、開始から終了の1つ手前までの整数を順に返す。
    for i in range(dataStart, len(df)):
        # iloc[i] で i 番目の行を取り出す。
        row  = df.iloc[i]
        name = cleanName(str(row.iloc[OCC_NAME_COL]))
        # 空の名前・"nan"（空のセルを文字列にしたもの）・短すぎる名前・年齢の行は飛ばす。
        if not name or name == "nan" or len(name) < MIN_NAME_LENGTH or _isAgeRowName(name):
            continue

        wage = safeNum(row.iloc[OCC_WAGE_COL])
        if np.isnan(wage) or wage <= MIN_VALID_WAGE:
            continue
        # append でリストの末尾に辞書を1つ追加する。
        records.append({
            "year": year, "occupation": name,
            "monthly_wage": toManYen(wage),
            "annual_bonus": toManYen(safeNum(row.iloc[OCC_BONUS_COL])),
        })
    return records


# 職種別給与の全ファイルを処理し、年収の列を足して CSV に保存する。
def processOccupationWage() -> pd.DataFrame:
    logger.info("[職種別給与]")
    # 関数 _parseOccupation を、呼び出さずに（括弧を付けずに）そのまま引数として渡している。
    dfOut = addAnnualIncome(pd.DataFrame(_readFiles("occ", _parseOccupation)))
    saveCsv(dfOut, "occupation_wage_all.csv")
    logger.info("  ✅ occupation_wage_all.csv: %s レコード (%s〜%s年, %d 職種)",
                # f"..." は f文字列で、{ } の中に式を書ける。:, は3桁ごとのカンマ区切り。nunique は重複を除いた件数。
                f"{len(dfOut):,}", dfOut["year"].min(), dfOut["year"].max(), dfOut["occupation"].nunique())
    return dfOut


# ══════════════════════════════════════════════════════
# 2. 年齢階級別給与
# ══════════════════════════════════════════════════════
# 年齢の行の値から1レコードを作る共通の処理。re.Match | None は「正規表現の一致の情報か None」という型。
def _ageRecord(year: int, occ: str, ageMatch: re.Match | None,
               wage: float, bonus: float) -> dict:
    """年齢行の値から 1 レコードを作る。ageMatch が None なら 19 歳以下の行"""
    if ageMatch is None:
        ageMid, ageLabel = UNDER19_AGE_MID, UNDER19_AGE_LABEL
    else:
        # カンマで区切って、2つの変数に同時に代入できる。
        ageFrom, ageTo = int(ageMatch.group(1)), int(ageMatch.group(2))
        ageMid, ageLabel = (ageFrom + ageTo) / 2, f"{ageFrom}〜{ageTo}歳"
    return {
        "year": year, "occupation": occ,
        "age_label": ageLabel, "age_mid": ageMid,
        "monthly_wage": toManYen(wage),
        "annual_bonus": toManYen(bonus),
    }


# 新形式（2020年以降）の年齢階級別の表を解析する。
def _parseAgeNew(df: pd.DataFrame, year: int) -> list[dict]:
    """2020〜: col1=職種名/年齢, col7=給与, col9=賞与"""
    records: list[dict] = []
    # 直前に見つけた職種名。年齢の行は、この職種に属するものとして記録する。
    currentOcc: str | None = None
    dataStart = findDataStart(df)
    if dataStart is None:
        return records

    for i in range(dataStart, len(df)):
        row  = df.iloc[i]
        raw  = str(row.iloc[AGE_NEW_NAME_COL]).replace("\n", "")
        name = raw.replace("　", "").strip()

        # 「数字 ～ 数字」の形を探す。\d+ は1文字以上の数字、\s* は0個以上の空白、[～~] は全角か半角のチルダ。
        ageMatch = re.search(r"(\d+)\s*[～~]\s*(\d+)", name)
        isUnder19 = "１９歳" in name or "19歳" in name

        # 年齢の行なら記録し、そうでなければ職種名の行として currentOcc を更新する。
        if ageMatch or isUnder19:
            if currentOcc is None:
                continue
            wage = safeNum(row.iloc[AGE_NEW_WAGE_COL])
            if np.isnan(wage) or wage <= 0:
                continue
            # 19歳以下の行は、ageMatch の代わりに None を渡す。
            records.append(_ageRecord(year, currentOcc, None if isUnder19 else ageMatch,
                                      wage, safeNum(row.iloc[AGE_NEW_BONUS_COL])))
        # elif は「そうでなく、もし〜なら」。字下げの深い内訳の行は職種名として扱わない。
        elif name and name != "nan" and len(name) > 1 and not raw.startswith(DEEP_INDENT):
            currentOcc = cleanName(name)
    return records


# 旧形式（2019年以前）の年齢階級別の表を解析する。新形式とは列の位置と名前の書き方が違う。
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
            # 末尾の「(男)」のような括弧書きを消す。.*? はできるだけ短く一致させる指定、$ は末尾。
            occ = re.sub(r"\s*\(.*?\)\s*$", "", raw0).strip()
            if occ and not re.match(r"^\d", occ):
                currentOcc = occ
    return records


# 年によって、新形式・旧形式のどちらの解析を使うかを振り分ける。
def _parseAge(df: pd.DataFrame, year: int) -> list[dict]:
    return _parseAgeNew(df, year) if year >= NEW_FORMAT_FROM_YEAR else _parseAgeOld(df, year)


# 年齢階級別給与の全ファイルを処理して CSV に保存する。
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
# 型ヒント: (キー, 見出しの表記のリスト) の組のリスト。キーは "total"（文字列）か経験年数（小数）。
_EXP_PATTERNS: list[tuple[str | float, list[str]]] = [
    (EXP_TOTAL_KEY, ["経験年数計"]),
    (0.0,  ["０年", "0年"]),
    (2.5,  ["１～４年", "1～4年", "1 ～ 4 年"]),
    (7.0,  ["５～９年", "5～9年", "5 ～ 9 年"]),
    (12.0, ["１０～１４年", "10～14年"]),
    (17.0, ["１５～１９年", "15～19年", "１５年以上", "15年以上"]),
    (22.0, ["２０年以上", "20年以上"]),
]


# 見出しを探して {経験年数: 列番号} の辞書を作る。
def _getExpColMap(df: pd.DataFrame) -> dict[str | float, int]:
    """
    ヘッダー行を走査して 経験年数バンド→列インデックス のマップを返す。
    新形式(2020〜): 0年/1〜4年/5〜9年/10〜14年/15年以上  (5バンド)
    旧形式(〜2019): 0年/1〜4年/5〜9年/10〜14年/15〜19年/20年以上 (6バンド)
    """
    # 空の辞書。見つかった見出しを順に登録する。
    colMap: dict[str | float, int] = {}
    # head(n) は先頭の n 行だけを取り出す。
    for _, row in df.head(EXP_HEADER_ROWS).iterrows():
        # enumerate は (番号, 値) の組を順に返す。列番号を数えながら各セルを見る。
        for colIdx, val in enumerate(row):
            v = str(val).replace("　", "").replace(" ", "").strip()
            for key, labels in _EXP_PATTERNS:
                # any は「1つでも真があれば真」。まだ登録していないキーで、表記のどれかがセルに含まれていれば登録する。
                if key not in colMap and any(lbl.replace(" ", "") in v for lbl in labels):
                    colMap[key] = colIdx
    return colMap


# 1職種の行から、経験年数の階級ごとのレコードを作る。給与の列の右隣が賞与の列。
def _expRecords(row: pd.Series, year: int, occ: str, colMap: dict[str | float, int]) -> list[dict]:
    """1 職種の行から、経験年数バンドごとのレコードを作る（給与・賞与は隣り合う列）"""
    records: list[dict] = []
    # items() で辞書の (キー, 値) の組を順に取り出す。
    for expYears, col in colMap.items():
        if expYears == EXP_TOTAL_KEY or col >= len(row):
            continue
        wage  = safeNum(row.iloc[col])
        # 右隣の列が表の外なら、賞与は NaN にする。
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


# 経験年数別の表1ファイル分を解析する。isNewFormat で新旧の列位置を切り替える。
def _parseExp(df: pd.DataFrame, year: int, colMap: dict[str | float, int], isNewFormat: bool) -> list[dict]:
    records: list[dict] = []
    # get(キー, 既定値) は、キーがなければ既定値を返す（[ ] と違ってエラーにならない）。
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


# 1ファイル分の処理。列の対応が取れなければ、警告して空のリストを返す。
def _parseExpFile(df: pd.DataFrame, year: int) -> list[dict]:
    colMap = _getExpColMap(df)
    if not colMap:
        logger.warning("  ⚠ 列マップ取得失敗: %d 年のファイル", year)
        return []
    return _parseExp(df, year, colMap, isNewFormat=(year >= NEW_FORMAT_FROM_YEAR))


# 経験年数別給与の全ファイルを処理して CSV に保存する。
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
# Step1 全体の処理。-> None は「値を返さない」という意味。
def main() -> None:
    # 「文字列 * 数」で、その文字列を繰り返した文字列になる。
    rule = "=" * SECTION_RULE_WIDTH
    logger.info("\n%s\n  Step1: data/raw → data/processed  変換開始\n%s\n", rule, rule)

    # 関数をタプル（丸括弧の組）に並べ、順に取り出して呼び出している。
    for process in (processOccupationWage, processAgeWage, processExperienceWage,
                    processMonthlyLabor, processGdp):
        # 取り出した関数を呼び出す。
        process()
        logger.info("")
    processCpi()

    logger.info("\n%s\n  Step1 完了 → data/processed/\n%s\n", rule, rule)


# このファイルを直接実行したとき（python step1_to_processed.py）だけ main() を呼ぶ。他のファイルから import したときは実行しない。
if __name__ == "__main__":
    main()
