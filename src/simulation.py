from __future__ import annotations
import pandas as pd

from log_config import getLogger
from model_types import ModelDict
from model_wrappers import addFeatures

logger = getLogger(__name__)

_AGE_MIDS: list[float] = [18.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 67.0]
_AGE_LABELS: list[str] = [
    "〜19歳", "20〜24歳", "25〜29歳", "30〜34歳", "35〜39歳",
    "40〜44歳", "45〜49歳", "50〜54歳", "55〜59歳", "60〜64歳", "65〜69歳",
]
_FE_MODELS: frozenset[str] = frozenset({"custom", "xgboost", "elasticnet", "gradient_boosting"})

BASE_YEAR = 2024                 # 統計データの基準年
SIMULATION_YEARS = 50
RETIREMENT_AGE = 65              # この年齢以降は予測せず、前年から一定率で減らす
POST_RETIREMENT_DECAY = 0.97
FALLBACK_INCOME = 300.0          # 統計データが読めないときの初年度ベース年収（万円）
FALLBACK_AGE_LABEL = "〜"
MIN_FIRST_INCOME_RATIO = 0.8     # 転職初年度年収の下限（ベース年収に対する比率）
RAISE_SUPPRESSION_YEARS = 10     # 昇給抑制が効く年数（転職直後ほど強い）
CAREER_RISK_START_YEAR = 10      # キャリアリスクが効き始める年
CAREER_RISK_SPAN_YEARS = 40      # キャリアリスクが最大になるまでの年数
NOMINAL_RAISE_WEIGHT = 0.05      # 転職後の名目昇給を毎年どれだけ積み上げるか
MONTHS_PER_YEAR = 12


def predict(
    models: ModelDict,
    modelKey: str,
    occupation: str,
    age: float,
    experience: float,
) -> float:
    X = pd.DataFrame([{
        "occupation": occupation,
        "age": float(age),
        "experience_years": float(experience),
    }])
    if modelKey != "stacking" and modelKey in _FE_MODELS:
        X = addFeatures(X)
    return float(models[modelKey]["pipeline"].predict(X)[0])


def _lowerAgeIndex(currentAge: float) -> int:
    """現在の年齢に最も近い年齢階級の、1 つ下の階級の位置を返す"""
    currentMid = min(_AGE_MIDS, key=lambda m: abs(m - currentAge))
    return max(0, _AGE_MIDS.index(currentMid) - 1)


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
    except (OSError, ValueError) as e:
        logger.warning("年齢別年収データを読めないため既定値を使う: %s (%s)", ageAllPath, e)
        return FALLBACK_INCOME, FALLBACK_AGE_LABEL

    yearRows = ageAll[(ageAll["year"] == year) & (ageAll["age_mid"] == lowerMid)]
    occRows = yearRows[yearRows["occupation"] == occName]
    if len(occRows) > 0:
        return float(occRows["annual_income"].mean()), lowerLabel
    if len(yearRows) > 0:
        return float(yearRows["annual_income"].mean()), lowerLabel
    return FALLBACK_INCOME, FALLBACK_AGE_LABEL


def _simulateStatusQuo(models: ModelDict, modelKey: str, currentOcc: str, currentAge: int,
                       currentExp: float, currentIncome: float, nominalRaise: float,
                       years: int) -> list[float]:
    """現職を続けた場合の年収推移。予測値を現在の年収に合わせて補正する"""
    correction = currentIncome / max(predict(models, modelKey, currentOcc, currentAge, currentExp), 1)
    statusQuo: list[float] = []
    income = currentIncome
    for i in range(years):
        if currentAge + i >= RETIREMENT_AGE:
            income *= POST_RETIREMENT_DECAY
        else:
            income = (predict(models, modelKey, currentOcc, currentAge + i, currentExp + i)
                      * correction * (1 + nominalRaise))
        statusQuo.append(max(income, 0))
    return statusQuo


def simulate(
    models: ModelDict,
    modelKey: str,
    currentOcc: str,
    targetOcc: str,
    currentAge: int,
    currentExp: float,
    currentIncome: float,
    skillTransfer: float,
    nominalRaise: float,
    ageCurve: pd.DataFrame,
    years: int = SIMULATION_YEARS,
    ageAllPath: str = "",
    raiseSuppression: float = 0.0,
    careerRisk: float = 0.0,
) -> tuple[list[float], list[float]]:
    statusQuo = _simulateStatusQuo(models, modelKey, currentOcc, currentAge,
                                   currentExp, currentIncome, nominalRaise, years)

    baseIncome, _ = getOneStepDownIncome(targetOcc, currentAge, ageAllPath)
    experiencedIncome = predict(models, modelKey, targetOcc, currentAge, currentExp)
    firstIncome = max(
        baseIncome + (experiencedIncome - baseIncome) * skillTransfer,
        baseIncome * MIN_FIRST_INCOME_RATIO,
    )

    lowerMid = _AGE_MIDS[_lowerAgeIndex(currentAge)]
    correction = firstIncome / max(predict(models, modelKey, targetOcc, lowerMid, 0), 1)

    careerChange: list[float] = []
    for i in range(years):
        if currentAge + i >= RETIREMENT_AGE:
            careerChange.append(max(careerChange[-1] * POST_RETIREMENT_DECAY, 0))
            continue
        pred = predict(models, modelKey, targetOcc, currentAge + i, float(i))
        suppression = 1.0 - raiseSuppression * max(0, (RAISE_SUPPRESSION_YEARS - i) / RAISE_SUPPRESSION_YEARS)
        riskDecay = 1.0 - careerRisk * max(0, (i - CAREER_RISK_START_YEAR) / CAREER_RISK_SPAN_YEARS)
        raiseFactor = 1 + nominalRaise * i * NOMINAL_RAISE_WEIGHT
        careerChange.append(max(pred * correction * suppression * riskDecay * raiseFactor, 0))

    return statusQuo, careerChange


def calcRoi(
    statusQuo: list[float], careerChange: list[float], cost: float
) -> tuple[int | None, float]:
    """投資コストの回収月（回収できなければ None）と、生涯の年収差の合計を返す"""
    cumulative, breakevenMonth = 0.0, None
    for i, (sq, cc) in enumerate(zip(statusQuo, careerChange)):
        annualDiff = cc - sq
        cumulative += annualDiff
        monthlyDiff = annualDiff / MONTHS_PER_YEAR
        if monthlyDiff > 0 and breakevenMonth is None:
            monthsToBreak = (-cumulative + annualDiff + cost) / monthlyDiff
            if monthsToBreak <= MONTHS_PER_YEAR:
                breakevenMonth = i * MONTHS_PER_YEAR + int(monthsToBreak)
    lifetime = sum(cc - sq for sq, cc in zip(statusQuo, careerChange))
    return breakevenMonth, lifetime
