from __future__ import annotations
from typing import Any
import pandas as pd

_AGE_MIDS: list[float] = [18.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 67.0]
_AGE_LABELS: list[str] = [
    "〜19歳", "20〜24歳", "25〜29歳", "30〜34歳", "35〜39歳",
    "40〜44歳", "45〜49歳", "50〜54歳", "55〜59歳", "60〜64歳", "65〜69歳",
]
_FE_MODELS: frozenset[str] = frozenset({"custom", "xgboost", "elasticnet", "gradient_boosting"})


def _add_features(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    Xc["age_sq"] = Xc["age"] ** 2 / 1000
    Xc["age_x_exp"] = Xc["age"] * Xc["experience_years"] / 100
    Xc["exp_ratio"] = Xc["experience_years"] / Xc["age"].clip(lower=1)
    Xc["prime_age_flag"] = ((Xc["age"] >= 35) & (Xc["age"] <= 54)).astype(float)
    return Xc


def predict(
    models: dict[str, Any],
    model_key: str,
    occupation: str,
    age: float,
    experience: float,
) -> float:
    X = pd.DataFrame([{
        "occupation": occupation,
        "age": float(age),
        "experience_years": float(experience),
    }])
    if model_key != "stacking" and model_key in _FE_MODELS:
        X = _add_features(X)
    return float(models[model_key]["pipeline"].predict(X)[0])


def get_one_step_down_income(
    occ_name: str, current_age: float, age_all_path: str, year: int = 2024
) -> tuple[float, str]:
    try:
        current_mid = min(_AGE_MIDS, key=lambda m: abs(m - current_age))
        idx = _AGE_MIDS.index(current_mid)
        lower_mid = _AGE_MIDS[max(0, idx - 1)]
        lower_label = _AGE_LABELS[max(0, idx - 1)]

        age_all = pd.read_csv(age_all_path)
        occ_rows = age_all[(age_all["year"] == year) & (age_all["occupation"] == occ_name)]
        row = occ_rows[occ_rows["age_mid"] == lower_mid]
        if len(row) > 0:
            return float(row["annual_income"].mean()), lower_label

        all_row = age_all[(age_all["year"] == year) & (age_all["age_mid"] == lower_mid)]
        if len(all_row) > 0:
            return float(all_row["annual_income"].mean()), lower_label
    except Exception:
        pass
    return 300.0, "〜"


def simulate(
    models: dict[str, Any],
    model_key: str,
    current_occ: str,
    target_occ: str,
    current_age: int,
    current_exp: float,
    current_income: float,
    skill_transfer: float,
    nominal_raise: float,
    age_curve: pd.DataFrame,
    years: int = 50,
    age_all_path: str = "",
    raise_suppression: float = 0.0,
    career_risk: float = 0.0,
) -> tuple[list[float], list[float]]:
    base_pred = predict(models, model_key, current_occ, current_age, current_exp)
    correction = current_income / max(base_pred, 1)

    status_quo: list[float] = []
    income = current_income
    for i in range(years):
        age = current_age + i
        if age >= 65:
            income *= 0.97
        else:
            income = predict(models, model_key, current_occ, age, current_exp + i) * correction * (1 + nominal_raise)
        status_quo.append(max(income, 0))

    base_income, _ = get_one_step_down_income(target_occ, current_age, age_all_path)
    experienced_income = predict(models, model_key, target_occ, current_age, current_exp)
    first_income = max(
        base_income + (experienced_income - base_income) * skill_transfer,
        base_income * 0.8,
    )

    current_mid = min(_AGE_MIDS, key=lambda m: abs(m - current_age))
    lower_mid = _AGE_MIDS[max(0, _AGE_MIDS.index(current_mid) - 1)]
    corr2 = first_income / max(predict(models, model_key, target_occ, lower_mid, 0), 1)

    career_change: list[float] = []
    for i in range(years):
        age = current_age + i
        if age >= 65:
            career_change.append(max(career_change[-1] * 0.97, 0))
        else:
            pred = predict(models, model_key, target_occ, age, float(i))
            sf = 1.0 - raise_suppression * max(0, (10 - i) / 10)
            rd = 1.0 - career_risk * max(0, (i - 10) / 40)
            career_change.append(max(pred * corr2 * sf * rd * (1 + nominal_raise * i * 0.05), 0))

    return status_quo, career_change


def calc_roi(
    status_quo: list[float], career_change: list[float], cost: float
) -> tuple[int | None, float]:
    cumulative, breakeven_month = 0.0, None
    for i, (s, c) in enumerate(zip(status_quo, career_change)):
        annual_diff = c - s
        cumulative += annual_diff
        monthly_diff = annual_diff / 12
        if monthly_diff > 0 and breakeven_month is None:
            months_to_break = (-cumulative + annual_diff + cost) / monthly_diff
            if months_to_break <= 12:
                breakeven_month = i * 12 + int(months_to_break)
    lifetime = sum(c - s for s, c in zip(status_quo, career_change))
    return breakeven_month, lifetime
