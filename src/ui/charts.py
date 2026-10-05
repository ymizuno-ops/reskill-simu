# 型ヒントの新しい書き方を使えるようにする指定（詳しくは log_config.py）。
from __future__ import annotations
# numpy: 数値計算のライブラリ（累積和の計算に使う）。
import numpy as np
# Plotly: 操作できる（拡大・マウスを重ねて値を表示する等）グラフを作るライブラリ。go.Scatter などの部品でグラフを組み立てる。
import plotly.graph_objects as go
# 1つの図に、複数のグラフや2本目の縦軸を作るための関数。
from plotly.subplots import make_subplots

# 意味: グラフの色（現状維持・転職後・累積収支差額・学習コストの線・回収時点の線）。変えると色だけが変わる。
COLOR_STATUS_QUO = "#4F8EF7"
COLOR_CAREER_CHANGE = "#FF5B5B"
COLOR_CUMULATIVE = "#43a047"
COLOR_COST = "#FB8C00"
COLOR_BREAKEVEN = "gold"
# 意味: 全モデル比較グラフを横に何個並べるか。
MAX_GRID_COLS = 3
# 意味: 全モデル比較グラフの1段の高さ（ピクセル）。
SUBPLOT_HEIGHT = 300
MONTHS_PER_YEAR = 12


# 現状維持・転職後の年収と、累積の差額を1つのグラフにする。
def plotMainPlotly(
    statusQuo: list[float],
    careerChange: list[float],
    currentAge: int,
    currentOcc: str,
    targetOcc: str,
    cost: float,
) -> go.Figure:
    # リスト内包表記で、横軸の年齢のリストを作る。
    ages = [currentAge + i for i in range(len(statusQuo))]
    # np.cumsum は累積和（1年目、1+2年目、1+2+3年目…の合計）を計算する。
    cumulative = np.cumsum([cc - sq for sq, cc in zip(statusQuo, careerChange)])

    # 右側にもう1本の縦軸（累積の差額用）を持つ図を作る。
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # add_trace で線を1本追加する。secondary_y=False は左の縦軸を使う指定。
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
            # 線と 0 の間を塗りつぶし、opacity で半透明にする。
            fill="tozeroy", opacity=0.3,
        ),
        secondary_y=True,
    )
    # 学習コストがあるときだけ、マイナス側に水平の点線を引く。
    if cost > 0:
        fig.add_hline(
            y=-cost, line_dash="dot", line_color=COLOR_COST,
            annotation_text=f"学習コスト ▲{cost:,}万円",
            secondary_y=True,
        )
    # タイトル・軸の名前・凡例の位置など、図全体の見た目を設定する。hovermode="x unified" で同じ年齢の値をまとめて表示する。
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


# モデルごとの小さなグラフを、格子状に並べる。
def plotAllModelsPlotly(
    sqAll: list[list[float]],
    ccAll: list[list[float]],
    currentAge: int,
    breakevens: list[int | None],
    modelLabels: list[str],
) -> go.Figure:
    count = len(sqAll)
    nCols = min(MAX_GRID_COLS, count)
    # // は割り算の整数部分。足してから割ることで「切り上げ」の割り算になる。
    nRows = (count + nCols - 1) // nCols

    fig = make_subplots(rows=nRows, cols=nCols, subplot_titles=modelLabels, shared_xaxes=True)
    ages = [currentAge + i for i in range(len(sqAll[0]))]

    for i, (sq, cc, breakeven) in enumerate(zip(sqAll, ccAll, breakevens)):
        # % は割り算の余り。i 番目のグラフを何段目・何列目に置くかを計算する（Plotly の位置は 1 から数える）。
        row, col = (i // nCols) + 1, (i % nCols) + 1
        fig.add_trace(
            go.Scatter(x=ages, y=sq, line=dict(color=COLOR_STATUS_QUO, width=2),
                       # 凡例は最初のグラフの分だけ出す（同じ凡例がいくつも並ばないように）。
                       showlegend=(i == 0), name="現状維持"),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=ages, y=cc, line=dict(color=COLOR_CAREER_CHANGE, width=2),
                       showlegend=(i == 0), name="転職後"),
            row=row, col=col,
        )
        # 回収月があれば（None でなければ）、回収の時点に縦線を引く。
        if breakeven:
            fig.add_vline(
                x=currentAge + breakeven / MONTHS_PER_YEAR, line_dash="dash", line_color=COLOR_BREAKEVEN,
                row=row, col=col,
                annotation_text="回収", annotation_position="top right",
            )

    fig.update_layout(height=SUBPLOT_HEIGHT * nRows, title_text="モデル別 年収推移比較", showlegend=True)
    return fig
