import os
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from flask import Flask
from flask_autoindex import AutoIndex
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dao.golden_ai_backtest_metrics_dao import GoldenAIBacktestMetricsDAO

flask_server = Flask(__name__)

_assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
_ai = AutoIndex(flask_server, browse_root=_assets, add_url_rules=False)


_ALLOWED_DIRS = {'GoldenAITWStrategyWeekly', 'GoldenAITWStrategyMonthly'}


@flask_server.route('/reports/')
def reports_root():
    return '''<ul>
        <li><a href="/reports/GoldenAITWStrategyWeekly/">Weekly 報告</a></li>
        <li><a href="/reports/GoldenAITWStrategyMonthly/">Monthly 報告</a></li>
    </ul>'''


@flask_server.route('/reports/<path:path>')
def autoindex(path):
    if path.split('/')[0] not in _ALLOWED_DIRS:
        return 'Forbidden', 403
    return _ai.render_autoindex(path)

METRICS = [
    ('annual_return', '年化報酬 (%)',    True),
    ('sharpe',        'Sharpe Ratio',   False),
    ('max_drawdown',  'Max Drawdown (%)', True),
    ('win_ratio',     '勝率 (%)',        True),
]

_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]

dao = GoldenAIBacktestMetricsDAO()


def _load_all(strategy: str) -> dict:
    df_all = dao.load(strategy=strategy)
    if df_all.empty:
        return {}

    result = {}
    for top_n in sorted(df_all['top_n'].unique()):
        df = df_all[df_all['top_n'] == top_n].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()

        if strategy == 'monthly':
            df = (
                df.groupby('timestamp')[['annual_return', 'sharpe', 'sortino', 'max_drawdown', 'win_ratio']]
                .mean()
                .reset_index()
            )

        result[int(top_n)] = df.sort_values('timestamp')

    return result


def _build_figure(data: dict, strategy: str) -> go.Figure:
    titles = [m[1] for m in METRICS]
    fig = make_subplots(
        rows=len(METRICS), cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.08,
    )

    if not data:
        fig.update_layout(
            height=700,
            title=f"尚無資料（{strategy}）",
            plot_bgcolor='white',
        )
        return fig

    for i, (col, label, is_pct) in enumerate(METRICS, 1):
        for top_n, df in sorted(data.items()):
            color = _COLORS[(top_n - 1) % len(_COLORS)]
            y = df[col] * 100 if is_pct else df[col]
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=y,
                    mode='lines+markers',
                    name=f'Top{top_n}',
                    legendgroup=f'Top{top_n}',
                    showlegend=(i == 1),
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                    hovertemplate=f'%{{x|%Y-%m-%d}}<br>Top{top_n} {label}: %{{y:.2f}}<extra></extra>',
                ),
                row=i, col=1,
            )
        if is_pct:
            fig.update_yaxes(ticksuffix='%', row=i, col=1)

    fig.update_layout(
        height=820,
        margin=dict(l=70, r=20, t=80, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eeeeee', tickformat='%Y-%m-%d')
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee', zeroline=True, zerolinecolor='#cccccc')

    return fig


app = dash.Dash(
    __name__,
    server=flask_server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = 'GoldenAI Backtest Dashboard'

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2('GoldenAI 回測績效', className='mt-3 mb-3'), width='auto'),
        dbc.Col([
            html.A('Weekly 報告', href='/reports/GoldenAITWStrategyWeekly/', target='_blank',
                   className='btn btn-outline-secondary btn-sm me-2 mt-3'),
            html.A('Monthly 報告', href='/reports/GoldenAITWStrategyMonthly/', target='_blank',
                   className='btn btn-outline-secondary btn-sm mt-3'),
        ], className='d-flex align-items-start'),
    ], className='mb-1'),

    dbc.Row([
        dbc.Col([
            html.Label('策略'),
            dcc.Dropdown(
                id='strategy-dropdown',
                options=[
                    {'label': 'Weekly（週策略）', 'value': 'weekly'},
                    {'label': 'Monthly（月策略 Week1~4 平均）', 'value': 'monthly'},
                ],
                value='weekly',
                clearable=False,
            ),
        ], width=4),
    ], className='mb-4'),

    dcc.Graph(id='metrics-graph', config={'displayModeBar': False}),
], fluid=True)


@app.callback(
    Output('metrics-graph', 'figure'),
    Input('strategy-dropdown', 'value'),
)
def update_graph(strategy):
    data = _load_all(strategy)
    return _build_figure(data, strategy)


server = flask_server


if __name__ == '__main__':
    app.run(debug=True, port=8051)
