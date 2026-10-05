"""
step1_macro.py
==============
マクロ経済系データ（毎月勤労統計・GDP・CPI）の変換（step1_to_processed.py から分離）。
"""

# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# os: ファイルのパスの操作、re: 正規表現（文字のパターンによる検索）。
import os, re
import pandas as pd
import numpy as np

# 同じ src/ フォルダにある自作のモジュール（.py ファイル）から関数を読み込む。
from log_config import getLogger
from step1_common import subdir, listXlsx, safeNum, extractYear, saveCsv

# __name__ はこのモジュールの名前（"step1_macro"）。モジュールごとのロガーを作る。
logger = getLogger(__name__)

# % を割合に直すための数（5% → 0.05）。
PERCENT = 100

# 毎月勤労統計: 「調査産業計」行の列位置
# 意味: 毎月勤労統計の表で、全産業の合計行を見つける目印の文字。
# 影響: 表の行名が変わったのに目印を変えないと、そのファイルの年の賃金上昇率が取れない。全ファイルで取れないと Step1 がエラーで止まる。
LABOR_TOTAL_ROW_KEY = "調査産業計"
# 意味: 合計行で、給与と前年比が入っている列の番号（左端の列が 0）。
# 注意: 表の様式が変わったら Excel で列を数え直して合わせる。ずれると別の数値を読む。
LABOR_WAGE_COL      = 4   # きまって支給する給与（円）
LABOR_YOY_COL       = 5   # 前年比（%）

# GDP: 年次の実質成長率 CSV
# 意味: 読み込む GDP 成長率ファイルの名前。
# 影響: 新しい年のファイルに差し替えたら、ここを新しいファイル名にする。違うと GDP のデータが1件も読めず、Step1 がエラーで止まる。
GDP_FILE_NAME = "2024_年次GDP成長率_実質.csv"
# 意味: GDP の CSV の文字コード（配布元の内閣府の形式）。
# 注意: 配布元の形式に合わせた値。違うと読み込みエラーになる。
GDP_ENCODING  = "shift_jis"
# 意味: GDP の CSV で、年度と成長率が入っている列の番号（左端の列が 0）。
GDP_YEAR_COL  = 0         # 年度「YYYY/4-3.」
GDP_RATE_COL  = 1         # 実質GDP成長率（%）

# CPI: 中分類指数の年平均
# 意味: 読み込む消費者物価指数（CPI）ファイルの名前。
# 影響: 新しい年のファイルに差し替えたら、ここを新しいファイル名にする。違うと CPI のデータが1件も読めず、Step1 がエラーで止まる。
CPI_FILE_NAME = "2025_消費者物価指数_中分類指数_全国__年平均.xlsx"
# 意味: CPI の表の先頭にある見出し行の数。この行数だけ読み飛ばす。
# 注意: 表の見出し行数が変わったら合わせる。ずれると年や指数を読めず、物価上昇率が欠損する。
CPI_SKIP_ROWS = 14
# 意味: 見出しを読み飛ばした後の表で、年と総合指数が入っている列の番号（左端の列が 0）。
CPI_YEAR_COL  = 8         # 年（例: '2020年'）
CPI_INDEX_COL = 12        # 総合指数（2020年=100）


# 出力した件数と年の範囲をログに出す。先頭の _ は「このファイルの中だけで使う」という慣習の印。
def _logYearRange(fileName: str, df: pd.DataFrame) -> None:
    # %s（文字列）や %d（整数）の位置に、後ろの引数が順に入る。
    logger.info("  ✅ %s: %d 年分 (%s〜%s年)", fileName, len(df), df["year"].min(), df["year"].max())


# ══════════════════════════════════════════════════════
# 4. 毎月勤労統計調査
# ══════════════════════════════════════════════════════
# 1ファイル分の表から合計行を探し、1年分のレコード（辞書）を返す。見つからなければ None。
def _findLaborTotal(df: pd.DataFrame, year: int) -> dict[str, float] | None:
    """「調査産業計」行の給与と前年比を返す。行がない・給与が空なら None"""
    # _ は「使わない値」を受け取る変数名の慣習（ここでは行番号を使わない）。
    for _, row in df.iterrows():
        # iloc[0] は、位置（0 = 左端）でセルを指定して取り出す。
        cell = str(row.iloc[0]).replace("　", "").replace(" ", "")
        # 合計行でなければ、continue で次の行へ進む。
        if LABOR_TOTAL_ROW_KEY not in cell:
            continue
        wage = safeNum(row.iloc[LABOR_WAGE_COL])
        yoy  = safeNum(row.iloc[LABOR_YOY_COL])
        # NaN かどうかは == では判定できないため、np.isnan を使う。
        if np.isnan(wage):
            return None
        # 辞書を返す。キーは出力する CSV の列名になる。
        return {
            "year": year,
            "scheduled_wage_yen": wage,
            "yoy_pct": yoy,
            # 前年比が NaN なら NaN のまま、そうでなければ % を割合に直す。
            "yoy_rate": yoy / PERCENT if not np.isnan(yoy) else np.nan,
        }
    return None


# 毎月勤労統計の全ファイルを読み、年ごとの賃金と前年比の CSV を作る。
def processMonthlyLabor() -> pd.DataFrame:
    logger.info("[毎月勤労統計]")
    files = listXlsx("labor")
    logger.info("  対象: %d ファイル", len(files))
    # 1年分ずつ辞書を追加していくリスト。
    records = []

    for path in files:
        # basename はパスからファイル名の部分だけを取り出す。
        year = extractYear(os.path.basename(path))
        # 1ファイルの読み込みに失敗しても、警告を出して残りのファイルの処理を続ける。
        try:
            # read_excel で Excel を読む。header=None は1行目を見出しとして扱わない、dtype=str は全セルを文字列として読む指定。
            record = _findLaborTotal(pd.read_excel(path, header=None, dtype=str), year)
        # Exception はほぼすべての種類のエラーを表す。as e でエラーの内容を変数 e に入れる。
        except Exception as e:
            # warning は info より重要度が高い、警告のメッセージ。
            logger.warning("  ⚠ %s: %s", os.path.basename(path), e)
            continue
        # None との比較には == ではなく is を使うのが Python の慣習。
        if record is not None:
            records.append(record)

    # 辞書のリストから DataFrame を作り、年の順に並べ替え、行番号を 0 から振り直す（drop=True で古い行番号は捨てる）。メソッドを . で続けて呼ぶ書き方を「メソッドチェーン」と呼ぶ。
    dfOut = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    saveCsv(dfOut, "monthly_labor_all.csv")
    logger.info("  ✅ monthly_labor_all.csv: %d 年分", len(dfOut))
    # df[[列1, 列2]] のように角括弧を二重にすると、複数の列だけを取り出した表になる。to_string は表を文字列にする。
    logger.info("%s", dfOut[["year", "scheduled_wage_yen", "yoy_pct"]].to_string(index=False))
    return dfOut


# ══════════════════════════════════════════════════════
# 5. GDP 実質成長率（年次）
# ══════════════════════════════════════════════════════
# GDP 成長率の CSV を読み、年ごとの実質成長率の CSV を作る。
def processGdp() -> pd.DataFrame:
    logger.info("[GDP成長率]")
    path = os.path.join(subdir("gdp"), GDP_FILE_NAME)
    records = []
    try:
        # read_csv で CSV を読む。encoding で文字コードを指定する。
        df = pd.read_csv(path, encoding=GDP_ENCODING, header=None, dtype=str)
        for _, row in df.iterrows():
            # re.match は文字列の先頭から一致を調べる。「4桁の数字の後に /」で始まる行（年度の行）だけを処理する。
            match = re.match(r"(\d{4})/", str(row.iloc[GDP_YEAR_COL]).strip())
            # 一致しなければ（見出しの行など）次の行へ進む。
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
# CPI（消費者物価指数）の表を読み、年ごとの指数と前年比の CSV を作る。
def processCpi() -> pd.DataFrame:
    logger.info("[CPI]")
    path = os.path.join(subdir("cpi"), CPI_FILE_NAME)
    records = []
    try:
        # skiprows で、先頭の見出しの行を読み飛ばす。
        df = pd.read_excel(path, header=None, skiprows=CPI_SKIP_ROWS, dtype=str)
        for _, row in df.iterrows():
            match = re.match(r"(\d{4})", str(row.iloc[CPI_YEAR_COL]).strip())
            if not match:
                continue
            cpi = safeNum(row.iloc[CPI_INDEX_COL])
            # and は両方とも真のときに真。
            if not np.isnan(cpi) and cpi > 0:
                records.append({"year": int(match.group(1)), "cpi": cpi})
    except Exception as e:
        logger.warning("  ⚠ CPI: %s", e)

    dfOut = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    # pct_change は1つ前の行からの変化率（(今 − 前) ÷ 前）を計算する。先頭の行は前がないため NaN になる。
    dfOut["cpi_yoy"] = dfOut["cpi"].pct_change()
    saveCsv(dfOut, "cpi_annual.csv")
    _logYearRange("cpi_annual.csv", dfOut)
    return dfOut
