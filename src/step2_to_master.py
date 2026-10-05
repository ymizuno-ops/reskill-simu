"""
step2_to_master.py
==================
data/processed/ の各CSVを結合・整形し、
機械学習に使用する data/master/ のCSVを生成する。

出力ファイル一覧:
  master/
    ml_dataset.csv        # 訓練用メインデータ
                          #   occupation, age, experience_years, annual_income
    occupation_list.csv   # 職種マスタ（UI用）
                          #   occupation, latest_annual_income, latest_monthly_wage
    age_curve.csv         # 年齢別平均年収カーブ（全職種平均）
    exp_curve.csv         # 経験年数別平均年収カーブ（全職種平均）
    macro_params.json     # マクロ経済パラメータ（将来推計用）
"""

# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# json: JSON ファイルの読み書きをする標準モジュール。warnings: 警告メッセージの表示を制御する。
import os, json, warnings
import pandas as pd
import numpy as np

from log_config import getLogger
# 型ヒント用の型を読み込む。
from model_types import MacroParams

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

_HERE    = os.path.dirname(os.path.abspath(__file__))
# 意味: Step1 が出力した CSV を読み込む場所。
# 注意: step1_common.py の OUT_DIR と同じ場所にする。
PROC_DIR = os.path.join(_HERE, "..", "data", "processed")
# 意味: 学習データ・職種マスタ・マクロ経済パラメータの出力先。
# 注意: main.py と step3_train.py の MASTER_DIR と同じ場所にする（画面と学習がここを読む）。
OUT_DIR  = os.path.join(_HERE, "..", "data", "master")
os.makedirs(OUT_DIR, exist_ok=True)

# 意味: 職種マスタ・年齢カーブ・学習データを作るときに使う統計の年（物価の基準年も兼ねる）。
# 影響: 新しい年のデータを追加してここを上げると、その年の給与水準で学習し直せる。
# 注意: data/raw にその年のファイルと CPI の値がないと、職種リストが空になって止まる。simulation.py の BASE_YEAR も同じ年にする。
LATEST_YEAR  = 2024   # 基準年
# 意味: 学習データに加えるばらつき（ノイズ）の乱数の種。同じ値なら何度実行しても同じデータになる。
# 影響: 変えると学習データが少し変わり、モデルの精度と予測年収もわずかに変わる。
RANDOM_SEED  = 42
CSV_ENCODING = "utf-8-sig"
# 意味: 職種名にこの文字（歳・才）を含む行は、年齢の行が混ざったものとみなして除外する。
# 影響: 職種名に「歳」「才」を含む職種があると、その職種も除外される。
INVALID_OCC_PATTERN = r"歳|才"   # 年齢行など職種でない行

# 職種マスタ: 年収が現実的な範囲（万円）
# 意味: 職種マスタに載せる職種の年収の範囲（万円）。極端な値の職種を除くためのもの。
# 影響: 範囲を狭めると、画面で選べる職種と学習データの職種がどちらも減る。
MIN_ANNUAL_INCOME = 100
MAX_ANNUAL_INCOME = 3000

# 年齢カーブ: 昇給率の初期値と上下限
# 意味: 最も若い年齢階級の昇給率（比べる前の階級がないため仮に置く値）。
# 影響: age_curve.csv の先頭行の raise_rate だけが変わる。現在の画面・計算ではこの列を使っていないため、結果は変わらない。
DEFAULT_RAISE_RATE = 0.03
# 意味: 年齢階級の間の昇給率の下限・上限（0.12 = 12%）。極端な値を丸める。
# 影響: age_curve.csv の raise_rate 列が変わる。現在の画面・計算ではこの列を使っていないため、結果は変わらない。
MIN_RAISE_RATE     = -0.05
MAX_RAISE_RATE     = 0.12

# ML データセット
# 意味: 働き始める最も早い年齢。経験年数が「年齢 − この値」を超える組み合わせは現実にないため、学習データから除く。
# 影響: 上げると、若い年齢で経験年数が長い組み合わせが学習データから減る。
WORK_START_AGE = 18     # 経験年数 <= 年齢 - 18 を満たす組み合わせだけ使う
# 意味: 学習データの合成年収にかけるランダムなばらつきの大きさ（0.05 = ±5% 程度）。
# 影響: 大きくするとモデルが細かい差を覚えにくくなり、予測がなだらかになる。0 にすると同じ条件の年収がすべて同じ値になる。
NOISE_STD      = 0.05   # 汎化性能向上のためのノイズ（±5%）
# 意味: 学習データの合成年収の下限（万円）。これより低い値はこの値に切り上げる。
MIN_INCOME     = 50     # 合成年収の下限（万円）

# マクロ経済パラメータ
# 意味: 賃金上昇率・GDP 成長率・物価上昇率の平均を取る、直近の年数。
# 影響: 短くすると直近の景気をより強く反映し、画面の「期待GDP成長率」の初期値が変わる。
RECENT_YEARS = 10
# 意味: 将来の名目昇給率・実質昇給率の推計値の下限・上限（0.03 = 3%）。
# 影響: macro_params.json の forecast_* が変わる。現在の画面・計算ではこの値を使っていないため、結果は変わらない。
MIN_FORECAST_NOMINAL, MAX_FORECAST_NOMINAL = 0.0, 0.03
MIN_FORECAST_REAL,    MAX_FORECAST_REAL    = -0.01, 0.02
PERCENT = 100

SECTION_RULE_WIDTH = 60


# processed フォルダの CSV を読み込み、DataFrame で返す。
def _readProcessed(fileName: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROC_DIR, fileName))


# str.contains で正規表現に一致する行を探し、~（否定）で一致しない行だけを残す。na=False は空欄を「一致しない」とみなす指定。
def _dropInvalidOcc(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["occupation"].str.contains(INVALID_OCC_PATTERN, na=False)]


# master フォルダへ CSV を保存する。
def _saveCsv(df: pd.DataFrame, fileName: str) -> None:
    df.to_csv(os.path.join(OUT_DIR, fileName), index=False, encoding=CSV_ENCODING)


# ──────────────────────────────────────────────────────
# 1. 職種マスタ（UI用職種リスト）
# ──────────────────────────────────────────────────────
# 職種マスタ（画面の職種リスト）を作る。
def buildOccupationList() -> pd.DataFrame:
    occAll = _dropInvalidOcc(_readProcessed("occupation_wage_all.csv"))

    # 最新年のデータを基準とし、重複がある場合は月収の高いものを優先
    # 括弧で囲んで、メソッドチェーンを複数行に分けて書いている。
    latest = (
        # df[条件] で、条件が真の行だけを取り出す。
        occAll[occAll["year"] == LATEST_YEAR]
        # 月給の高い順（降順）に並べる。
        .sort_values("monthly_wage", ascending=False)
        # 職種名が重複する行は、最初の1行だけを残す。直前で月給の高い順にしたので、月給の高い方が残る。
        .drop_duplicates(subset="occupation")
        .reset_index(drop=True)
    )
    # 条件を & で組み合わせるときは、それぞれを括弧で囲む（pandas の決まり）。
    latest = latest[
        (latest["annual_income"] >= MIN_ANNUAL_INCOME) & (latest["annual_income"] <= MAX_ANNUAL_INCOME)
    ]

    # 必要な列だけを取り出し、copy で独立したコピーにする（元の表への変更に関する警告を避けるため）。
    result = latest[["occupation", "monthly_wage", "annual_bonus", "annual_income"]].copy()
    result = result.sort_values("occupation").reset_index(drop=True)

    _saveCsv(result, "occupation_list.csv")
    logger.info("  ✅ occupation_list.csv: %d 職種", len(result))
    return result


# ──────────────────────────────────────────────────────
# 2. 年齢カーブ（全職種平均・2024年基準）
# ──────────────────────────────────────────────────────
# 全職種平均の、年齢別の年収カーブを作る。
def buildAgeCurve() -> pd.DataFrame:
    ageAll = _dropInvalidOcc(_readProcessed("age_wage_all.csv"))

    # 最新年のみ使用、全職種平均
    latest = ageAll[ageAll["year"] == LATEST_YEAR]
    curve = (
        # groupby で年齢階級ごとにまとめる。
        latest.groupby(["age_label", "age_mid"])
        # agg(新しい列名=(元の列名, 集計方法)) で、グループごとの平均を計算する。
        .agg(monthly_wage=("monthly_wage", "mean"), annual_income=("annual_income", "mean"))
        # groupby でできた見出し（インデックス）を、普通の列に戻す。
        .reset_index()
        .sort_values("age_mid")
    )

    # 年齢別の昇給率（隣接年齢階級との比率）
    # 1つ前の年齢階級からの年収の伸び率。先頭は NaN になるので、fillna で既定値を入れる。
    curve["raise_rate"] = curve["annual_income"].pct_change().fillna(DEFAULT_RAISE_RATE)
    # clip(下限, 上限) で、範囲外の値を下限・上限に丸める。
    curve["raise_rate"] = curve["raise_rate"].clip(MIN_RAISE_RATE, MAX_RAISE_RATE)

    _saveCsv(curve, "age_curve.csv")
    logger.info("  ✅ age_curve.csv: %d 年齢階級", len(curve))
    logger.info("%s", curve[["age_label", "age_mid", "monthly_wage", "raise_rate"]].to_string(index=False))
    return curve


# ──────────────────────────────────────────────────────
# 3. 経験年数カーブ（全職種平均・2024年基準）
# ──────────────────────────────────────────────────────
# 全職種平均の、経験年数別の年収カーブを作る。
def buildExpCurve() -> pd.DataFrame:
    expAll = _dropInvalidOcc(_readProcessed("experience_wage_all.csv"))

    latest = expAll[expAll["year"] == LATEST_YEAR]
    curve = (
        latest.groupby("experience_years")
        .agg(monthly_wage=("monthly_wage", "mean"), annual_income=("annual_income", "mean"))
        .reset_index()
        .sort_values("experience_years")
    )

    _saveCsv(curve, "exp_curve.csv")
    logger.info("  ✅ exp_curve.csv: %d 経験年数階級", len(curve))
    logger.info("%s", curve.to_string(index=False))
    return curve


# ──────────────────────────────────────────────────────
# 4. 機械学習用メインデータセット
# ──────────────────────────────────────────────────────
# 1職種について、年齢階級 × 経験年数のすべての組み合わせで合成の年収を作る。
def _occupationRecords(occ: str, baseIncome: float,
                       occAge: pd.DataFrame, occExp: pd.DataFrame) -> list[dict]:
    """1 職種について、年齢階級 × 経験年数の合成年収レコードを作る"""
    records: list[dict] = []
    # 全組み合わせの平均年収（スケーリング基準）
    # mean は平均値。
    ageMean = occAge["annual_income"].mean()
    expMean = occExp["annual_income"].mean() if len(occExp) > 0 else baseIncome

    # 二重ループ: 年齢の行ごとに、経験年数の行をすべて組み合わせる。
    for _, ageRow in occAge.iterrows():
        ageMid = ageRow["age_mid"]
        # その年齢の年収が平均の何倍か。0 で割らないよう、平均が 0 以下なら 1.0 にする。
        ageRatio = ageRow["annual_income"] / ageMean if ageMean > 0 else 1.0

        for _, expRow in occExp.iterrows():
            expYears = expRow["experience_years"]
            # 経験年数が年齢的に不可能なケースを除外
            if expYears > ageMid - WORK_START_AGE:
                continue

            expRatio = expRow["annual_income"] / expMean if expMean > 0 else 1.0
            # 合成年収 = ベース × 年齢倍率 × 経験年数倍率（相乗平均で合成）
            # np.sqrt は平方根。2つの倍率の相乗平均（掛けて平方根）を取り、どちらか一方の影響が大きくなりすぎないようにする。
            syntheticIncome = baseIncome * np.sqrt(ageRatio * expRatio)
            # 平均 1.0・標準偏差 NOISE_STD の正規分布から乱数を1つ取る（だいたい 0.95〜1.05 の値）。
            noise = np.random.normal(1.0, NOISE_STD)

            records.append({
                "occupation": occ,
                "age": ageMid,
                "experience_years": expYears,
                # max で下限を保証し、round(値, 1) で小数第1位に丸める。
                "annual_income": round(max(syntheticIncome * noise, MIN_INCOME), 1),
            })
    return records


# 学習データ（ml_dataset.csv）を作る。引数は職種マスタ。
def buildMlDataset(occList: pd.DataFrame) -> pd.DataFrame:
    """
    年齢階級×経験年数×職種 の組み合わせで年収を計算したデータセットを作る。

    設計方針:
      - 職種別ベース年収（latest年）を基準とする
      - 年齢による倍率（age_curve基準）を乗算
      - 経験年数による倍率（exp_curve基準）を乗算
      - 全体の平均値でスケーリングして自然なレンジに調整
      - 年度推移（2006〜2024）の賃金上昇率を反映
    """
    ageAll = _dropInvalidOcc(_readProcessed("age_wage_all.csv"))
    expAll = _dropInvalidOcc(_readProcessed("experience_wage_all.csv"))
    ageLatest = ageAll[ageAll["year"] == LATEST_YEAR].copy()
    expLatest = expAll[expAll["year"] == LATEST_YEAR].copy()

    # 職種別データがない場合に使う全職種平均カーブ
    # 職種別のデータがない職種のために、全職種平均のカーブを先に作っておく。
    ageAverage = ageLatest.groupby(["age_label", "age_mid"]).agg(
        annual_income=("annual_income", "mean")
    ).reset_index()
    expAverage = expLatest.groupby("experience_years").agg(
        annual_income=("annual_income", "mean")
    ).reset_index()

    records: list[dict] = []
    # 職種マスタの職種ごとに処理する。
    for _, occRow in occList.iterrows():
        occ = occRow["occupation"]
        occAge = ageLatest[ageLatest["occupation"] == occ]
        occExp = expLatest[expLatest["occupation"] == occ]
        records.extend(_occupationRecords(
            occ, occRow["annual_income"],
            # その職種の年齢別データがあればそれを、なければ全職種平均を使う。
            occAge if len(occAge) > 0 else ageAverage,
            occExp if len(occExp) > 0 else expAverage,
        ))

    # 全職種のレコードをまとめて1つの表にする。
    df = pd.DataFrame(records)
    logger.info("\n  ML dataset: %s サンプル, %d 職種", f"{len(df):,}", df["occupation"].nunique())
    logger.info("  年収レンジ: %.0f〜%.0f 万円", df["annual_income"].min(), df["annual_income"].max())
    logger.info("  年収中央値: %.0f 万円", df["annual_income"].median())

    _saveCsv(df, "ml_dataset.csv")
    logger.info("  ✅ ml_dataset.csv: %s レコード", f"{len(df):,}")
    return df


# ──────────────────────────────────────────────────────
# 5. マクロ経済パラメータ
# ──────────────────────────────────────────────────────
# 直近の年だけに絞り、指定した列の平均を返す。dropna で欠損を除いてから平均する。
def _recentMean(df: pd.DataFrame, column: str) -> float:
    """直近 RECENT_YEARS 年の平均（より現実的な推計値）"""
    return float(df[df["year"] >= LATEST_YEAR - RECENT_YEARS][column].dropna().mean())


# 将来推計に使うマクロ経済のパラメータを計算し、JSON に保存する。
def buildMacroParams() -> MacroParams:
    avgWageGrowth = _recentMean(_readProcessed("monthly_labor_all.csv"), "yoy_rate")
    avgGdpGrowth  = _recentMean(_readProcessed("gdp_annual.csv"), "gdp_real_growth")
    avgCpiYoy     = _recentMean(_readProcessed("cpi_annual.csv"), "cpi_yoy")

    # 実質賃金成長率 = 名目賃金成長 - インフレ率
    realWageGrowth = avgWageGrowth - avgCpiYoy

    # 将来推計: 直近実績を重視し、上下限でクリップ
    # np.clip(値, 下限, 上限) で範囲内に丸める。float で numpy の数値を Python の数値に直す（JSON に保存するため）。
    forecastNominal = float(np.clip(avgWageGrowth, MIN_FORECAST_NOMINAL, MAX_FORECAST_NOMINAL))
    forecastReal    = float(np.clip(realWageGrowth, MIN_FORECAST_REAL, MAX_FORECAST_REAL))

    # JSON に保存する辞書。キーが macro_params.json の項目名になる。
    params: MacroParams = {
        "latest_year":            LATEST_YEAR,
        "avg_wage_growth_10yr":   avgWageGrowth,
        "avg_gdp_growth_10yr":    avgGdpGrowth,
        "avg_cpi_10yr":           avgCpiYoy,
        "real_wage_growth_10yr":  realWageGrowth,
        "forecast_nominal_raise": forecastNominal,
        "forecast_real_raise":    forecastReal,
    }

    # with 文でファイルを開くと、ブロックを抜けるときに自動で閉じられる。"w" は書き込みのモード。
    with open(os.path.join(OUT_DIR, "macro_params.json"), "w", encoding="utf-8") as f:
        # 辞書を JSON として書き込む。ensure_ascii=False で日本語をそのまま、indent=2 で字下げして見やすくする。
        json.dump(params, f, ensure_ascii=False, indent=2)

    logger.info("  ✅ macro_params.json")
    # %.2f は小数第2位までの数値、%% は「%」の文字そのもの。
    logger.info("     名目賃金上昇（10年平均）: %.2f%%", avgWageGrowth * PERCENT)
    logger.info("     実質GDP成長（10年平均）:  %.2f%%", avgGdpGrowth * PERCENT)
    logger.info("     CPI（10年平均）:          %.2f%%", avgCpiYoy * PERCENT)
    logger.info("     将来推計（名目）:          %.2f%%", forecastNominal * PERCENT)
    return params


# ──────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────
# Step2 全体の処理。
def main() -> None:
    # 乱数の種を固定する。何度実行してもノイズが同じになり、結果を再現できる。
    np.random.seed(RANDOM_SEED)

    rule = "=" * SECTION_RULE_WIDTH
    logger.info("\n%s\n  Step2: processed → master  データセット構築\n%s\n", rule, rule)

    logger.info("[職種マスタ]")
    occList = buildOccupationList()
    logger.info("\n[年齢カーブ]")
    buildAgeCurve()
    logger.info("\n[経験年数カーブ]")
    buildExpCurve()
    logger.info("\n[MLデータセット構築]")
    buildMlDataset(occList)
    logger.info("\n[マクロ経済パラメータ]")
    buildMacroParams()

    logger.info("\n%s\n  Step2 完了 → data/master/\n%s\n", rule, rule)


# このファイルを直接実行したときだけ main() を呼ぶ。
if __name__ == "__main__":
    main()
