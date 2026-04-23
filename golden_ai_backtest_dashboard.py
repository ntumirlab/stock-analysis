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

dao = GoldenAIBacktestMetricsDAO()


def _load(strategy: str, top_n: int) -> pd.DataFrame:
    df = dao.load(strategy=strategy, top_n=top_n)
    if df.empty:
        return df

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()

    if strategy == 'monthly':
        # Average Week1~4 per execution timestamp
        df = (
            df.groupby('timestamp')[['annual_return', 'sharpe', 'sortino', 'max_drawdown', 'win_ratio']]
            .mean()
            .reset_index()
        )

    return df.sort_values('timestamp')


def _build_figure(df: pd.DataFrame, strategy: str, top_n: int) -> go.Figure:
    titles = [m[1] for m in METRICS]
    fig = make_subplots(
        rows=len(METRICS), cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.08,
    )

    if df.empty:
        fig.update_layout(
            height=700,
            title=f"尚無資料（{strategy} / Top{top_n}）",
            plot_bgcolor='white',
        )
        return fig

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']

    for i, (col, label, is_pct) in enumerate(METRICS, 1):
        y = df[col] * 100 if is_pct else df[col]
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=y,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[i - 1], width=2),
                marker=dict(size=6),
                showlegend=False,
                hovertemplate=f'%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}<extra></extra>',
            ),
            row=i, col=1,
        )
        if is_pct:
            fig.update_yaxes(ticksuffix='%', row=i, col=1)

    fig.update_layout(
        height=760,
        margin=dict(l=70, r=20, t=60, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eeeeee', tickformat='%Y-%m-%d')
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee', zeroline=True, zerolinecolor='#cccccc')

    return fig


def _top_n_options(strategy: str) -> list:
    df = dao.load(strategy=strategy)
    if df.empty:
        return [{'label': f'Top{n}', 'value': n} for n in range(1, 9)]
    max_n = int(df['top_n'].max())
    return [{'label': f'Top{n}', 'value': n} for n in range(1, max_n + 1)]


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
        dbc.Col([
            html.Label('持股數'),
            dcc.Dropdown(
                id='topn-dropdown',
                value=1,
                clearable=False,
            ),
        ], width=3),
    ], className='mb-4'),

    dcc.Graph(id='metrics-graph', config={'displayModeBar': False}),
], fluid=True)


@app.callback(
    Output('topn-dropdown', 'options'),
    Output('topn-dropdown', 'value'),
    Input('strategy-dropdown', 'value'),
)
def update_topn_options(strategy):
    options = _top_n_options(strategy)
    return options, options[0]['value'] if options else 1


@app.callback(
    Output('metrics-graph', 'figure'),
    Input('strategy-dropdown', 'value'),
    Input('topn-dropdown', 'value'),
)
def update_graph(strategy, top_n):
    if top_n is None:
        top_n = 1
    df = _load(strategy, int(top_n))
    return _build_figure(df, strategy, int(top_n))


server = flask_server


if __name__ == '__main__':
    app.run(debug=True, port=8051)
