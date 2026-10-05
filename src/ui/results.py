# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# Streamlit: Python だけで Web 画面を作るライブラリ。
import streamlit as st
from simulation import calcRoi, RETIREMENT_AGE, MONTHS_PER_YEAR

# 影響: 大きくすると画面の「月収」表示が小さくなる。年収・生涯年収の表示は変わらない。
_SALARY_MONTHS = 14  # 年収 ÷ 14 で月収換算（賞与2ヶ月分込み）
# 意味: 画面の「3. 転職から〇年後」で比べる年数。
# 影響: そのブロックの見出しと、月収・年収の比較の数値が変わる。
YEARS_AFTER_CHANGE = 5
PERCENT = 100
# 意味: ROI の計算で 0 で割らないための、自己投資費用の下限（万円）。
# 注意: 費用 0 のときは ROI を「∞」と表示するため、通常は変える必要はない。
MIN_COST_FOR_ROI = 1


# 値を HTML の <span> で囲んだ文字列を作る。cls は CSS のクラス名で、色や太さを切り替える。
def _valueSpan(val: float | str, fmt: str = ".1f", unit: str = "万円", cls: str = "val") -> str:
    # {val:{fmt}} は、書式（例: ".1f" = 小数第1位まで）を変数で指定する f文字列の書き方。
    return f"<span class='{cls}'>{val:{fmt}}{unit}</span>"


# 差額を、プラスなら緑、マイナスなら赤で表示する HTML を作る。
def _diffSpan(val: float, unit: str = "万円", fmt: str = ".1f") -> str:
    isPositive = val >= 0
    cls = "pos" if isPositive else "neg"
    sign = "+" if isPositive else ""
    return f"<span class='{cls}'>{sign}{val:{fmt}}{unit}</span>"


# 回収月から、表示用の文字列と色のクラスを決める。
def _breakevenText(breakevenMonth: int | None) -> tuple[str, str]:
    """回収期間の表示文字列と CSS クラスを返す"""
    if not breakevenMonth:
        return "回収困難", "neu"
    if breakevenMonth <= MONTHS_PER_YEAR:
        return "1年以内", "pos"
    # divmod は割り算の商と余りを同時に返す（例: 30 か月 → 2年 6か月）。
    years, months = divmod(breakevenMonth, MONTHS_PER_YEAR)
    return f"{years}年{months}か月", "neu"


# 分析結果の5つのブロックを、2列で描画する。
def renderAnalysisResults(
    statusQuo: list[float],
    careerChange: list[float],
    currentAge: int,
    currentOcc: str,
    targetOcc: str,
    currentIncome: float,
    skillTransfer: float,
    learningCost: float,
) -> None:
    # 退職までの年数。生涯年収は、この年数分の合計。
    yearsToRetire = max(0, RETIREMENT_AGE - currentAge)
    idxAfter = min(YEARS_AFTER_CHANGE, len(statusQuo) - 1)

    currentMonthly = currentIncome / _SALARY_MONTHS
    firstMonthly = careerChange[0] / _SALARY_MONTHS
    sqAfterAnnual, ccAfterAnnual = statusQuo[idxAfter], careerChange[idxAfter]
    sqAfterMonthly = sqAfterAnnual / _SALARY_MONTHS
    ccAfterMonthly = ccAfterAnnual / _SALARY_MONTHS
    # [:n] はリストの先頭から n 個を取り出すスライス。
    sqLifetime = sum(statusQuo[:yearsToRetire])
    ccLifetime = sum(careerChange[:yearsToRetire])
    netBenefit = ccLifetime - sqLifetime - learningCost
    breakevenMonth, _ = calcRoi(statusQuo, careerChange, learningCost)
    # 費用が 0 なら、ROI は無限大（float("inf")）とする。
    roiPct = (netBenefit / max(learningCost, MIN_COST_FOR_ROI)) * PERCENT if learningCost > 0 else float("inf")

    # 画面を2列に分け、それぞれの列の入れ物を受け取る。
    colLeft, colRight = st.columns(2)

    # with ブロックの中で置いた部品は、その列に表示される。
    with colLeft:
        # markdown で文章や HTML を表示する。unsafe_allow_html=True で、HTML のタグをそのまま使えるようにする。
        st.markdown(
            f"<div class='result-section'><h4>1. 現状（現在年齢）</h4><ul>"
            f"<li>月収: {_valueSpan(currentMonthly)}</li>"
            f"<li>年収: {_valueSpan(currentIncome)}</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='result-section'><h4>2. 転職直後（初年度）</h4><ul>"
            f"<li>月収: {_valueSpan(firstMonthly)}</li>"
            f"<li>年収: {_valueSpan(careerChange[0])}</li>"
            f"<li style='font-size:.78rem;opacity:0.7;list-style:none;margin-left:-1.2rem;margin-top:4px'>"
            f"※スキル引継ぎ率 {int(skillTransfer * PERCENT)}% を適用済み</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )
        netCls = "pos" if netBenefit >= 0 else "neg"
        st.markdown(
            f"<div class='result-section'><h4>4. {RETIREMENT_AGE}歳時点での生涯年収</h4><ul>"
            f"<li>転職しなかった場合: {_valueSpan(sqLifetime, fmt=',.0f')}</li>"
            f"<li>転職した場合: {_valueSpan(ccLifetime, fmt=',.0f')}</li>"
            f"</ul>"
            f"<div style='font-size:.78rem;opacity:0.7;margin-top:.4rem'>生涯差額（投資コスト控除後）</div>"
            f"<div class='lifetime-highlight {netCls}'>{netBenefit:+,.0f} 万円</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with colRight:
        st.markdown(
            f"<div class='result-section'><h4>3. 転職から{YEARS_AFTER_CHANGE}年後</h4><ul>"
            f"<li>転職前の月収: {_valueSpan(sqAfterMonthly)}</li>"
            f"<li>転職後の月収: {_valueSpan(ccAfterMonthly)}</li>"
            f"<li>月収差額: {_diffSpan(ccAfterMonthly - sqAfterMonthly)}</li>"
            f"</ul><ul style='margin-top:.5rem'>"
            f"<li>転職前の年収: {_valueSpan(sqAfterAnnual, fmt=',.1f')}</li>"
            f"<li>転職後の年収: {_valueSpan(ccAfterAnnual, fmt=',.1f')}</li>"
            f"<li>年収差額: {_diffSpan(ccAfterAnnual - sqAfterAnnual, fmt=',.1f')}</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )

        breakevenStr, breakevenCls = _breakevenText(breakevenMonth)
        roiStr = f"{roiPct:,.1f} %" if roiPct != float("inf") else "∞（費用0円）"
        st.markdown(
            f"<div class='result-section'><h4>5. 費用対効果</h4><ul>"
            f"<li>投資コスト回収期間: {_valueSpan(breakevenStr, fmt='', unit='', cls=breakevenCls)}</li>"
            f"<li>生涯年収ベースのROI: <span class='{'pos' if roiPct > 0 else 'neg'}'>{roiStr}</span></li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )
