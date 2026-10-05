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
# 意味: e-stat からダウンロードした元データ（Excel・CSV）を置くフォルダ。
# 注意: この下に、_SUB に書いた6つのサブフォルダが必要。場所を変えたらフォルダごと移す。
RAW_DIR = os.path.join(_HERE, "..", "data", "raw")
# 意味: Step1 が整形した CSV を書き出すフォルダ。
# 注意: 変えたら step2_to_master.py の PROC_DIR と main.py の AGE_ALL_PATH も同じ場所にする（Step2 と画面がここを読む）。
OUT_DIR = os.path.join(_HERE, "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# 意味: 統計の種類ごとに、元データを置くサブフォルダの名前。
# 影響: 実際のフォルダ名と一字でも違うと、その統計のファイルが見つからずに Step1 がエラーで止まる。
_SUB = {
    "occ":   "職種別きまって支給する現金給与額",
    "age":   "年齢階級別きまって支給する現金給与額",
    "exp":   "経験年数階級別きまって支給する現金給与額",
    "labor": "毎月勤労統計調査_結果確報",
    "gdp":   "国民経済計算_GDP統計",
    "cpi":   "消費者物価指数",
}

# 統計表で「値なし」を表す記号
# 意味: 統計表で「数値なし・秘匿」を表す記号の一覧。これらのセルは欠損として集計から外す。
# 注意: 一覧にない記号も数値に変換できなければ欠損になるため、記号を追加しても結果は変わらない（記号の目録として残している）。
_MISSING_MARKS = {"-", "－", "…", "***", "nan", "", "x", "X", "−"}
# 意味: 職種別の表で、データ本体の最初の行を見つける目印の文字（先頭の職種「管理的職業従事者」）。
# 影響: 表の先頭の職種名が変わったのに目印を変えないと、そのファイルの全職種が読み飛ばされる。
DEFAULT_DATA_START_KEYWORD = "管理的"

# ── 単位・集計の共通定数 ──────────────────────────────
# 意味: 統計表の金額（千円単位）を万円に直すための割り数。
# 注意: 単位の換算なので変えない。変えると全ての年収が桁違いになる。
THOUSAND_YEN_PER_MAN_YEN = 10       # 千円 → 万円
# 意味: 年収 = 月収 × 12 + 賞与 の「12」。
# 注意: 暦の月数なので変えない。
MONTHS_PER_YEAR          = 12
# 意味: 職種別・年齢別の表で賞与が空欄のとき、賞与を月収の何か月分とみなすか。
# 影響: 大きくすると賞与が空欄の職種の年収が上がり、学習データとその職種の予測年収も上がる。
DEFAULT_BONUS_MONTHS     = 2        # 賞与が欠損のとき月収の何か月分とみなすか
# 意味: 統計表の様式が新形式に変わった年。この年以降のファイルは新形式の列位置で読む。
# 注意: e-stat の様式変更の年に合わせた値。変えると列がずれ、別の列の数値を給与として読んでしまう。
NEW_FORMAT_FROM_YEAR     = 2020     # この年以降は新形式の表
# 意味: 月給（千円）がこの値以下の行は、小計行や異常値とみなして除外する。
# 影響: 大きくすると低賃金の職種まで除外され、画面で選べる職種が減る。
MIN_VALID_WAGE           = 10       # これ以下（千円）の給与は集計行・異常値とみなす
# 意味: 職種名とみなす最低の文字数。短い見出しや記号だけの行を除くためのもの。
# 影響: 3 以上にすると「医師」のような2文字の職種が除外される。
MIN_NAME_LENGTH          = 2
# 意味: 「〜19歳」の年齢階級を、計算上は何歳として扱うか。
# 影響: 若年層の年齢と年収の対応がずれ、学習データの最も若い点の位置が変わる。
UNDER19_AGE_MID          = 18.0
# 意味: CSV と画面に出る、19歳以下の年齢階級の名前。
# 注意: simulation.py の _AGE_LABELS の先頭と同じ表記にする。
UNDER19_AGE_LABEL        = "〜19歳"
# 意味: 新形式の表で、職種の内訳行を表す字下げ（全角スペース3つ）。この字下げの行は職種として扱わない。
# 注意: 表の様式に合わせた値。変えると内訳行が職種として混ざる。
DEEP_INDENT              = "　　　"  # 新形式で内訳行を表すインデント
# 意味: 出力する CSV の文字コード（BOM付き UTF-8）。
# 影響: Excel で開いたときに日本語が文字化けしないための設定。変えると Excel で文字化けすることがある。
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
