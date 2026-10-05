# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# pandas: 表形式のデータを扱うライブラリ。
import pandas as pd

from log_config import getLogger
from model_types import ModelDict
# 学習のときと同じ特徴量の作り方を使う（予測のときに同じ列がないと、モデルが動かない）。
from model_wrappers import addFeatures

logger = getLogger(__name__)

# 意味: 統計の年齢階級（5歳刻み）それぞれの真ん中の年齢。転職直後の年収の基準になる「1つ下の年齢階級」を選ぶのに使う。
# 注意: すぐ下の _AGE_LABELS と同じ順番・同じ個数にし、age_wage_all.csv の age_mid と一致させる。
_AGE_MIDS: list[float] = [18.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 67.0]
_AGE_LABELS: list[str] = [
    "〜19歳", "20〜24歳", "25〜29歳", "30〜34歳", "35〜39歳",
    "40〜44歳", "45〜49歳", "50〜54歳", "55〜59歳", "60〜64歳", "65〜69歳",
]
# 意味: 予測の前に追加の項目（年齢² など）を計算して渡すモデルの一覧。
# 注意: model_wrappers.py の StackingEnsemble.FE_KEYS と同じ内容にする。
_FE_MODELS: frozenset[str] = frozenset({"custom", "xgboost", "elasticnet", "gradient_boosting"})

# 意味: 転職直後の年収の基準にする統計の年。
# 注意: step2_to_master.py の LATEST_YEAR と同じ年にする。データにない年だと、基準年収が既定値（FALLBACK_INCOME）になる。
BASE_YEAR = 2024                 # 統計データの基準年
# 意味: 何年先までシミュレーションするか。
# 影響: グラフの横軸と年次詳細の表の行数が変わる。注意: 退職年齢に届かない年数にすると、画面の生涯年収が途中までの合計になる。
SIMULATION_YEARS = 50
# 意味: 退職年齢。この年齢以降は予測をやめ、毎年一定の割合で年収を減らす。
# 影響: 上げると画面の「〇歳時点での生涯年収」が増え、見出しの年齢も変わる。サイドバーの年齢の上限（この値 − 1）も連動して変わる。
RETIREMENT_AGE = 65              # この年齢以降は予測せず、前年から一定率で減らす
# 意味: 退職年齢以降、年収を毎年何倍にするか（0.97 = 毎年 3% 減）。
# 影響: グラフと年次詳細の退職後の値が変わる。画面の「生涯年収」は退職年齢までの合計なので変わらない。
POST_RETIREMENT_DECAY = 0.97
# 意味: 統計データが読めないとき・目標職種と年齢のデータがないときに使う、転職直後の基準年収（万円）。
# 影響: その場合、転職後の年収推移全体がこの値に比例して上下する。
FALLBACK_INCOME = 300.0          # 統計データが読めないときの初年度ベース年収（万円）
# 意味: 上の既定値を使ったときに、スキル引継ぎ率ガイドの見出しに出す年齢階級名。
FALLBACK_AGE_LABEL = "〜"
# 意味: 転職初年度の年収の下限（基準年収の何倍か。0.8 = 8割）。
# 影響: 上げると、経験引継ぎ率が低いときや経験者の予測が低いときでも、初年度の年収が下がりにくくなる。
MIN_FIRST_INCOME_RATIO = 0.8     # 転職初年度年収の下限（ベース年収に対する比率）
# 意味: 「転職後の昇給抑制」が効く年数。転職直後が最も強く、この年数をかけて効果がなくなる。
# 影響: 長くすると抑制が長く続き、転職後の生涯年収が下がる。
RAISE_SUPPRESSION_YEARS = 10     # 昇給抑制が効く年数（転職直後ほど強い）
# 意味: 「キャリアリスク係数」による減額が始まる、転職後の年数。
# 影響: 小さくすると減額が早く始まり、転職後の生涯年収が下がる。
CAREER_RISK_START_YEAR = 10      # キャリアリスクが効き始める年
# 意味: 減額が始まってから、減額が最大（係数そのもの）に達するまでの年数。
# 影響: 短くすると減額が急に進む。
CAREER_RISK_SPAN_YEARS = 40      # キャリアリスクが最大になるまでの年数
# 意味: 転職後に、名目昇給率を毎年どれだけ積み上げるか（経過年数 × 昇給率 × この値）。
# 影響: 大きくすると転職後の年収が年を追うごとに大きく伸びる。注意: 現状維持側は毎年一律に（1 + 昇給率）倍しているため、この値で両者の伸び方の差が変わる。
NOMINAL_RAISE_WEIGHT = 0.05      # 転職後の名目昇給を毎年どれだけ積み上げるか
# 意味: 投資コストの回収期間を月で数えるための月数。暦の月数なので変えない。
MONTHS_PER_YEAR = 12


# 1人分の条件から、指定したモデルで年収を予測する。
def predict(
    models: ModelDict,
    modelKey: str,
    occupation: str,
    age: float,
    experience: float,
) -> float:
    # 1行だけの表を作る。モデルは表の形の入力を受け取るため。
    X = pd.DataFrame([{
        "occupation": occupation,
        "age": float(age),
        "experience_years": float(experience),
    }])
    # 特徴量が必要なモデルだけ列を追加する。Stacking は内部で、必要なモデルにだけ追加する。
    if modelKey != "stacking" and modelKey in _FE_MODELS:
        X = addFeatures(X)
    # predict は配列を返すので、[0] で1件目を取り出し、float で普通の数値にする。
    return float(models[modelKey]["pipeline"].predict(X)[0])


# min(リスト, key=関数) は、関数の値が最も小さい要素を選ぶ。ここでは年齢との差（abs = 絶対値）が最も小さい階級を選ぶ。
def _lowerAgeIndex(currentAge: float) -> int:
    """現在の年齢に最も近い年齢階級の、1 つ下の階級の位置を返す"""
    currentMid = min(_AGE_MIDS, key=lambda m: abs(m - currentAge))
    # index で位置を調べ、その1つ下の位置を返す。max(0, ...) で先頭より前にならないようにする。
    return max(0, _AGE_MIDS.index(currentMid) - 1)


# 戻り値は (年収, 年齢階級名) の組。
def getOneStepDownIncome(
    occName: str, currentAge: float, ageAllPath: str, year: int = BASE_YEAR
) -> tuple[float, str]:
    """
    目標職種の「1 つ下の年齢階級」の平均年収とその階級名を返す。
    統計データが読めない・該当がない場合は (FALLBACK_INCOME, FALLBACK_AGE_LABEL) を返す。
    """
    lowerIdx = _lowerAgeIndex(currentAge)
    lowerMid, lowerLabel = _AGE_MIDS[lowerIdx], _AGE_LABELS[lowerIdx]
    try:
        ageAll = pd.read_csv(ageAllPath)
    # 複数のエラーの種類をタプルで並べて捕まえる。OSError はファイルがない等、ValueError は中身が読めない等。
    except (OSError, ValueError) as e:
        logger.warning("年齢別年収データを読めないため既定値を使う: %s (%s)", ageAllPath, e)
        return FALLBACK_INCOME, FALLBACK_AGE_LABEL

    # 基準年で、かつ1つ下の年齢階級の行を取り出す。
    yearRows = ageAll[(ageAll["year"] == year) & (ageAll["age_mid"] == lowerMid)]
    occRows = yearRows[yearRows["occupation"] == occName]
    # 目標職種のデータがあればその平均、なければ全職種の平均を使う。
    if len(occRows) > 0:
        return float(occRows["annual_income"].mean()), lowerLabel
    if len(yearRows) > 0:
        return float(yearRows["annual_income"].mean()), lowerLabel
    return FALLBACK_INCOME, FALLBACK_AGE_LABEL


# years 年分の年収をリストで返す。
def _simulateStatusQuo(models: ModelDict, modelKey: str, currentOcc: str, currentAge: int,
                       currentExp: float, currentIncome: float, nominalRaise: float,
                       years: int) -> list[float]:
    """現職を続けた場合の年収推移。予測値を現在の年収に合わせて補正する"""
    # 補正係数 = 実際の年収 ÷ 予測の年収。max(..., 1) で 0 で割るのを防ぐ。
    correction = currentIncome / max(predict(models, modelKey, currentOcc, currentAge, currentExp), 1)
    statusQuo: list[float] = []
    income = currentIncome
    # i は今年からの経過年数（0, 1, 2, ...）。
    for i in range(years):
        if currentAge + i >= RETIREMENT_AGE:
            # *= は「掛けた結果を自分に代入する」短い書き方。
            income *= POST_RETIREMENT_DECAY
        else:
            income = (predict(models, modelKey, currentOcc, currentAge + i, currentExp + i)
                      * correction * (1 + nominalRaise))
        # マイナスにならないよう 0 以上にしてから追加する。
        statusQuo.append(max(income, 0))
    return statusQuo


# 現状維持と転職後の、2つの年収推移を計算する。
def simulate(
    models: ModelDict,
    modelKey: str,
    currentOcc: str,
    targetOcc: str,
    currentAge: int,
    currentExp: float,
    # 意味: 現在の年収（万円）。現状維持の推移は、モデルの予測をこの年収に合わせて補正する。
    # 影響: 現状維持の年収推移全体が、この値に比例して上下する。
    currentIncome: float,
    # 意味: 経験引継ぎ率（0〜1。1 = 前職の経験がすべて評価される）。
    # 影響: 上げると、転職初年度の年収が「同じ年齢・経験年数の経験者」の水準に近づく。
    skillTransfer: float,
    # 意味: 景気と物価による毎年の名目昇給率（0.01 = 1%）。サイドバーの GDP 成長率と CPI から計算される。
    # 影響: 上げると現状維持・転職後の両方の年収が上がる。
    nominalRaise: float,
    # 注意: 年齢カーブ（age_curve.csv）を受け取るが、現在の計算では使っていない。
    ageCurve: pd.DataFrame,
    years: int = SIMULATION_YEARS,
    ageAllPath: str = "",
    # 意味: 転職後の昇給抑制（0〜1）。
    # 影響: 上げると、転職直後の数年間の年収が下がる（RAISE_SUPPRESSION_YEARS の年数をかけて元に戻る）。
    raiseSuppression: float = 0.0,
    # 意味: キャリアリスク係数（0〜1）。
    # 影響: 上げると、転職から CAREER_RISK_START_YEAR 年後以降の年収が徐々に下がる。
    careerRisk: float = 0.0,
) -> tuple[list[float], list[float]]:
    statusQuo = _simulateStatusQuo(models, modelKey, currentOcc, currentAge,
                                   currentExp, currentIncome, nominalRaise, years)

    # 2つ目の戻り値（年齢階級名）は使わないので、_ で受け取る。
    baseIncome, _ = getOneStepDownIncome(targetOcc, currentAge, ageAllPath)
    experiencedIncome = predict(models, modelKey, targetOcc, currentAge, currentExp)
    # 初年度の年収 = 未経験の水準 + (経験者の水準 − 未経験の水準) × 引継ぎ率。下限を設けて下がりすぎないようにする。
    firstIncome = max(
        baseIncome + (experiencedIncome - baseIncome) * skillTransfer,
        baseIncome * MIN_FIRST_INCOME_RATIO,
    )

    # 1つ下の年齢階級の、真ん中の年齢。
    lowerMid = _AGE_MIDS[_lowerAgeIndex(currentAge)]
    # 転職後の補正係数: 1年目の年収が、上で計算した初年度の年収にそろうようにする。
    correction = firstIncome / max(predict(models, modelKey, targetOcc, lowerMid, 0), 1)

    careerChange: list[float] = []
    for i in range(years):
        if currentAge + i >= RETIREMENT_AGE:
            # [-1] はリストの最後の要素（前年の年収）。
            careerChange.append(max(careerChange[-1] * POST_RETIREMENT_DECAY, 0))
            continue
        pred = predict(models, modelKey, targetOcc, currentAge + i, float(i))
        # 転職直後ほど小さくなる係数（年数が経つと 1.0 に戻る）。
        suppression = 1.0 - raiseSuppression * max(0, (RAISE_SUPPRESSION_YEARS - i) / RAISE_SUPPRESSION_YEARS)
        # 一定の年数を過ぎると少しずつ小さくなる係数。
        riskDecay = 1.0 - careerRisk * max(0, (i - CAREER_RISK_START_YEAR) / CAREER_RISK_SPAN_YEARS)
        raiseFactor = 1 + nominalRaise * i * NOMINAL_RAISE_WEIGHT
        # 予測 × 補正 × 各係数 を、その年の転職後の年収とする。
        careerChange.append(max(pred * correction * suppression * riskDecay * raiseFactor, 0))

    return statusQuo, careerChange


def calcRoi(
    statusQuo: list[float], careerChange: list[float], cost: float
) -> tuple[int | None, float]:
    """投資コストの回収月（回収できなければ None）と、生涯の年収差の合計を返す"""
    # 累積の差額と、回収できた月（まだなら None）。
    cumulative, breakevenMonth = 0.0, None
    # zip で2つのリストを、同じ位置どうしの組にして取り出す。
    for i, (sq, cc) in enumerate(zip(statusQuo, careerChange)):
        annualDiff = cc - sq
        cumulative += annualDiff
        monthlyDiff = annualDiff / MONTHS_PER_YEAR
        if monthlyDiff > 0 and breakevenMonth is None:
            # その年の初めの時点で残っている回収額を、月あたりの差額で割り、何か月で回収できるかを求める。
            monthsToBreak = (-cumulative + annualDiff + cost) / monthlyDiff
            # その年のうちに回収できれば、回収月（通算の月数）を確定する。
            if monthsToBreak <= MONTHS_PER_YEAR:
                breakevenMonth = i * MONTHS_PER_YEAR + int(monthsToBreak)
    # 全年数の差額の合計。sum にジェネレータ式を渡している。
    lifetime = sum(cc - sq for sq, cc in zip(statusQuo, careerChange))
    return breakevenMonth, lifetime
