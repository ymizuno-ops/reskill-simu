"""
step1_common.py
===============
step1 の各処理が共有するパス設定と共通ユーティリティ（step1_to_processed.py から分離）。
"""

# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# os: ファイルやフォルダの操作、re: 正規表現（文字のパターンによる検索・置換）。カンマで区切ると複数まとめて読み込める。
import os, re
# pandas: 表形式のデータ（DataFrame）を扱うライブラリ。
import pandas as pd
# numpy: 数値計算のライブラリ。ここでは欠損値 np.nan と np.isnan を使う。
import numpy as np

# ── パス設定 ──────────────────────────────────────────
# __file__ はこのファイル自身のパス。abspath で絶対パスにし、dirname でフォルダ部分（src/）を取り出す。先頭の _ は「このファイルの中だけで使う」という慣習の印。
_HERE   = os.path.dirname(os.path.abspath(__file__))
# 意味: e-stat からダウンロードした元データ（Excel・CSV）を置くフォルダ。
# 注意: この下に、_SUB に書いた6つのサブフォルダが必要。場所を変えたらフォルダごと移す。
# os.path.join は OS に合った区切り文字でパスをつなぐ。".." は1つ上のフォルダ。
RAW_DIR = os.path.join(_HERE, "..", "data", "raw")
# 意味: Step1 が整形した CSV を書き出すフォルダ。
# 注意: 変えたら step2_to_master.py の PROC_DIR と main.py の AGE_ALL_PATH も同じ場所にする（Step2 と画面がここを読む）。
OUT_DIR = os.path.join(_HERE, "..", "data", "processed")
# 出力フォルダを作る。exist_ok=True で、すでにあってもエラーにしない。この行はファイルを import した時点で実行される。
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


# 統計の種類のキー（"occ" など）から、元データのフォルダのパスを返す。
def subdir(key: str) -> str:
    # 辞書[キー] で値を取り出す。存在しないキーだと KeyError（キーがないというエラー）になる。
    return os.path.join(RAW_DIR, _SUB[key])


# そのフォルダ内の .xlsx ファイルのパスを、名前順に並べたリストで返す。
def listXlsx(key: str) -> list[str]:
    dirPath = subdir(key)
    # ジェネレータ式（丸括弧の中に for を書く形）で各ファイル名をフルパスにし、endswith で拡張子が .xlsx のものだけを残して、sorted で並べ替える。
    return sorted(os.path.join(dirPath, f) for f in os.listdir(dirPath) if f.endswith(".xlsx"))


# ── 共通ユーティリティ ────────────────────────────────
# 型ヒントの object は「どんな型でも受け取れる」という意味。
def safeNum(value: object) -> float:
    """統計表のセル値を数値にする。値なしの記号や数値でない文字列は NaN を返す"""
    # 文字列にしてから、桁区切りのカンマ（半角・全角）を消し、strip で前後の空白を除く。
    text = str(value).replace(",", "").replace("，", "").strip()
    # in で集合（set）に含まれるかを調べる。
    if text in _MISSING_MARKS:
        # NaN（Not a Number）は「値なし」を表す特別な数値。
        return np.nan
    # try の中でエラー（例外）が起きたら、対応する except に処理が移る。
    try:
        # 文字列を小数に変換する。"abc" のような文字列だと ValueError が起きる。
        return float(text)
    # 数値に変換できなかったときだけここに来る。他の種類のエラーは捕まえずに呼び出し元へ伝える。
    except ValueError:
        return np.nan


# ファイル名から4桁の数字（年）を取り出す。見つからなければ 0 を返す。
def extractYear(fileName: str) -> int:
    # r"..." は生文字列（\ をそのまま書ける）。\d{4} は数字4桁で、( ) で囲んだ部分を後で group(1) として取り出せる。見つからなければ None が返る。
    match = re.search(r"(\d{4})", fileName)
    # 条件式: 「A if 条件 else B」は、条件が真なら A、偽なら B になる。
    return int(match.group(1)) if match else 0


# 職種名から全角スペース・改行と、「男女計」などの先頭の語を取り除く。
def cleanName(text: str) -> str:
    # 　 は全角スペース、\n と \r は改行の文字。
    name = str(text).replace("　", "").replace("\n", "").replace("\r", "").strip()
    # re.sub(パターン, 置換後, 文字列) は、パターンに合う部分を置き換える。^ は先頭、| は「または」。
    return re.sub(r"^(男女計|　男女計|男\s|女\s)", "", name).strip()


# 表の中でキーワードを含む最初の行の番号を返す。見つからなければ None。`int | None` は「int か None のどちらか」という型。
def findDataStart(df: pd.DataFrame, keyword: str = DEFAULT_DATA_START_KEYWORD) -> int | None:
    # iterrows は DataFrame を1行ずつ (行番号, 行のデータ) の組で取り出す。
    for i, row in df.iterrows():
        # 行の各セルの値を順に見る。
        for value in row.values:
            # 「文字列 in 文字列」で、含まれているかを調べる。
            if keyword in str(value):
                return i
    return None


# 千円 → 万円 の換算。値が NaN（空欄）なら NaN のまま返す。
def toManYen(thousandYen: float) -> float:
    """千円 → 万円。NaN はそのまま NaN を返す"""
    return thousandYen / THOUSAND_YEN_PER_MAN_YEN if not np.isnan(thousandYen) else np.nan


# DataFrame に年収の列を追加して返す。bonusMonths には既定値があり、呼び出し時に省略できる。
def addAnnualIncome(df: pd.DataFrame, bonusMonths: float = DEFAULT_BONUS_MONTHS) -> pd.DataFrame:
    """年収 = 月収 × 12 + 賞与（賞与が欠損なら月収 × bonusMonths）"""
    # df["列名"] = ... で列を追加する。列どうしの計算は全行まとめて行われる。fillna は欠損（NaN）を指定の値で埋める。
    df["annual_income"] = (
        df["monthly_wage"] * MONTHS_PER_YEAR
        + df["annual_bonus"].fillna(df["monthly_wage"] * bonusMonths)
    )
    return df


# DataFrame を出力フォルダに CSV で保存する。index=False で行番号の列を書き出さない。
def saveCsv(df: pd.DataFrame, fileName: str) -> None:
    df.to_csv(os.path.join(OUT_DIR, fileName), index=False, encoding=CSV_ENCODING)
