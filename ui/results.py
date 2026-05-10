from __future__ import annotations
from typing import Any
import streamlit as st
from simulation import calc_roi

_SALARY_MONTHS = 14  # 年収 ÷ 14 で月収換算（賞与2ヶ月分込み）


def render_analysis_results(
    status_quo: list[float],
    career_change: list[float],
    current_age: int,
    current_occ: str,
    target_occ: str,
    current_income: float,
    skill_transfer: float,
    learning_cost: float,
) -> None:
    ages_to_65 = max(0, 65 - current_age)
    idx5 = min(5, len(status_quo) - 1)

    current_monthly = current_income / _SALARY_MONTHS
    first_monthly = career_change[0] / _SALARY_MONTHS
    sq5_annual, cc5_annual = status_quo[idx5], career_change[idx5]
    sq5_monthly = sq5_annual / _SALARY_MONTHS
    cc5_monthly = cc5_annual / _SALARY_MONTHS
    sq_lifetime = sum(status_quo[:ages_to_65])
    cc_lifetime = sum(career_change[:ages_to_65])
    net_benefit = cc_lifetime - sq_lifetime - learning_cost
    breakeven_month, _ = calc_roi(status_quo, career_change, learning_cost)
    roi_pct = (net_benefit / max(learning_cost, 1)) * 100 if learning_cost > 0 else float("inf")

    def v(val: Any, fmt: str = ".1f", unit: str = "万円", cls: str = "val") -> str:
        return f"<span class='{cls}'>{val:{fmt}}{unit}</span>"

    def diff_span(val: float, unit: str = "万円", fmt: str = ".1f") -> str:
        cls = "pos" if val >= 0 else "neg"
        sign = "+" if val >= 0 else ""
        return f"<span class='{cls}'>{sign}{val:{fmt}}{unit}</span>"

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            f"<div class='result-section'><h4>1. 現状（現在年齢）</h4><ul>"
            f"<li>月収: {v(current_monthly)}</li>"
            f"<li>年収: {v(current_income)}</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='result-section'><h4>2. 転職直後（初年度）</h4><ul>"
            f"<li>月収: {v(first_monthly)}</li>"
            f"<li>年収: {v(career_change[0])}</li>"
            f"<li style='font-size:.78rem;opacity:0.7;list-style:none;margin-left:-1.2rem;margin-top:4px'>"
            f"※スキル引継ぎ率 {int(skill_transfer * 100)}% を適用済み</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )
        net_cls = "pos" if net_benefit >= 0 else "neg"
        st.markdown(
            f"<div class='result-section'><h4>4. 65歳時点での生涯年収</h4><ul>"
            f"<li>転職しなかった場合: {v(sq_lifetime, fmt=',.0f')}</li>"
            f"<li>転職した場合: {v(cc_lifetime, fmt=',.0f')}</li>"
            f"</ul>"
            f"<div style='font-size:.78rem;opacity:0.7;margin-top:.4rem'>生涯差額（投資コスト控除後）</div>"
            f"<div class='lifetime-highlight {net_cls}'>{net_benefit:+,.0f} 万円</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            f"<div class='result-section'><h4>3. 転職から5年後</h4><ul>"
            f"<li>転職前の月収: {v(sq5_monthly)}</li>"
            f"<li>転職後の月収: {v(cc5_monthly)}</li>"
            f"<li>月収差額: {diff_span(cc5_monthly - sq5_monthly)}</li>"
            f"</ul><ul style='margin-top:.5rem'>"
            f"<li>転職前の年収: {v(sq5_annual, fmt=',.1f')}</li>"
            f"<li>転職後の年収: {v(cc5_annual, fmt=',.1f')}</li>"
            f"<li>年収差額: {diff_span(cc5_annual - sq5_annual, fmt=',.1f')}</li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )

        if breakeven_month:
            be_str = "1年以内" if breakeven_month <= 12 else f"{breakeven_month // 12}年{breakeven_month % 12}か月"
            be_cls = "pos" if breakeven_month <= 12 else "neu"
        else:
            be_str = "回収困難"
            be_cls = "neu"
        roi_str = f"{roi_pct:,.1f} %" if roi_pct != float("inf") else "∞（費用0円）"

        st.markdown(
            f"<div class='result-section'><h4>5. 費用対効果</h4><ul>"
            f"<li>投資コスト回収期間: {v(be_str, fmt='', unit='', cls=be_cls)}</li>"
            f"<li>生涯年収ベースのROI: <span class='{'pos' if roi_pct > 0 else 'neg'}'>{roi_str}</span></li>"
            f"</ul></div>",
            unsafe_allow_html=True,
        )
