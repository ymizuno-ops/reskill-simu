from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_main_plotly(
    status_quo: list[float],
    career_change: list[float],
    current_age: int,
    current_occ: str,
    target_occ: str,
    cost: float,
) -> go.Figure:
    ages = [current_age + i for i in range(len(status_quo))]
    cumul = np.cumsum([c - s for s, c in zip(status_quo, career_change)])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=ages, y=status_quo, name="現状維持", line=dict(color="#4F8EF7", width=3)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=ages, y=career_change, name="転職後", line=dict(color="#FF5B5B", width=3)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ages, y=cumul, name="累積収支差額",
            line=dict(color="#43a047", width=2, dash="dash"),
            fill="tozeroy", opacity=0.3,
        ),
        secondary_y=True,
    )
    if cost > 0:
        fig.add_hline(
            y=-cost, line_dash="dot", line_color="#FB8C00",
            annotation_text=f"学習コスト ▲{cost:,}万円",
            secondary_y=True,
        )
    fig.update_layout(
        title=f"年収推移シミュレーション ({current_occ} vs {target_occ})",
        xaxis_title="年齢",
        yaxis_title="年収（万円）",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="累積収支差額（万円）", secondary_y=True)
    return fig


def plot_all_models_plotly(
    sq_all: list[list[float]],
    cc_all: list[list[float]],
    current_age: int,
    breakevens: list[int | None],
    model_labels: list[str],
) -> go.Figure:
    n = len(sq_all)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=model_labels, shared_xaxes=True)
    ages = [current_age + i for i in range(len(sq_all[0]))]

    for i, (sq, cc, be) in enumerate(zip(sq_all, cc_all, breakevens)):
        row, col = (i // ncols) + 1, (i % ncols) + 1
        fig.add_trace(
            go.Scatter(x=ages, y=sq, line=dict(color="#4F8EF7", width=2),
                       showlegend=(i == 0), name="現状維持"),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=ages, y=cc, line=dict(color="#FF5B5B", width=2),
                       showlegend=(i == 0), name="転職後"),
            row=row, col=col,
        )
        if be:
            fig.add_vline(
                x=current_age + be / 12, line_dash="dash", line_color="gold",
                row=row, col=col,
                annotation_text="回収", annotation_position="top right",
            )

    fig.update_layout(height=300 * nrows, title_text="モデル別 年収推移比較", showlegend=True)
    return fig
