from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_STATUS_QUO = "#4F8EF7"
COLOR_CAREER_CHANGE = "#FF5B5B"
COLOR_CUMULATIVE = "#43a047"
COLOR_COST = "#FB8C00"
COLOR_BREAKEVEN = "gold"
MAX_GRID_COLS = 3
SUBPLOT_HEIGHT = 300
MONTHS_PER_YEAR = 12


def plotMainPlotly(
    statusQuo: list[float],
    careerChange: list[float],
    currentAge: int,
    currentOcc: str,
    targetOcc: str,
    cost: float,
) -> go.Figure:
    ages = [currentAge + i for i in range(len(statusQuo))]
    cumulative = np.cumsum([cc - sq for sq, cc in zip(statusQuo, careerChange)])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=ages, y=statusQuo, name="現状維持", line=dict(color=COLOR_STATUS_QUO, width=3)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=ages, y=careerChange, name="転職後", line=dict(color=COLOR_CAREER_CHANGE, width=3)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ages, y=cumulative, name="累積収支差額",
            line=dict(color=COLOR_CUMULATIVE, width=2, dash="dash"),
            fill="tozeroy", opacity=0.3,
        ),
        secondary_y=True,
    )
    if cost > 0:
        fig.add_hline(
            y=-cost, line_dash="dot", line_color=COLOR_COST,
            annotation_text=f"学習コスト ▲{cost:,}万円",
            secondary_y=True,
        )
    fig.update_layout(
        title=f"年収推移シミュレーション ({currentOcc} vs {targetOcc})",
        xaxis_title="年齢",
        yaxis_title="年収（万円）",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="累積収支差額（万円）", secondary_y=True)
    return fig


def plotAllModelsPlotly(
    sqAll: list[list[float]],
    ccAll: list[list[float]],
    currentAge: int,
    breakevens: list[int | None],
    modelLabels: list[str],
) -> go.Figure:
    count = len(sqAll)
    nCols = min(MAX_GRID_COLS, count)
    nRows = (count + nCols - 1) // nCols

    fig = make_subplots(rows=nRows, cols=nCols, subplot_titles=modelLabels, shared_xaxes=True)
    ages = [currentAge + i for i in range(len(sqAll[0]))]

    for i, (sq, cc, breakeven) in enumerate(zip(sqAll, ccAll, breakevens)):
        row, col = (i // nCols) + 1, (i % nCols) + 1
        fig.add_trace(
            go.Scatter(x=ages, y=sq, line=dict(color=COLOR_STATUS_QUO, width=2),
                       showlegend=(i == 0), name="現状維持"),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=ages, y=cc, line=dict(color=COLOR_CAREER_CHANGE, width=2),
                       showlegend=(i == 0), name="転職後"),
            row=row, col=col,
        )
        if breakeven:
            fig.add_vline(
                x=currentAge + breakeven / MONTHS_PER_YEAR, line_dash="dash", line_color=COLOR_BREAKEVEN,
                row=row, col=col,
                annotation_text="回収", annotation_position="top right",
            )

    fig.update_layout(height=SUBPLOT_HEIGHT * nRows, title_text="モデル別 年収推移比較", showlegend=True)
    return fig
