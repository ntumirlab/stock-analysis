import os
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from flask import Flask
from flask_autoindex import AutoIndex
import plotly.graph_objects as go
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


# col -> (label, is_pct, positive_is_good)
_METRIC_META = {
    'annual_return': ('年化報酬', True,  True),
    'sharpe':        ('Sharpe Ratio', False, True),
    'max_drawdown':  ('Max Drawdown', True,  False),
    'win_ratio':     ('勝率', True,  True),
}

# Tableau 10
_COLORS = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
    '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7',
]

_PAGE_BG = '#f0f2f5'
_CARD_STYLE = {
    'boxShadow': '0 1px 4px rgba(0,0,0,0.08)',
    'borderRadius': '8px',
    'border': 'none',
}

dao = GoldenAIBacktestMetricsDAO()

_KPI_COLS = ['annual_return', 'sharpe', 'max_drawdown', 'win_ratio']


def _latest_kpi(strategy: str) -> dict:
    df_all = dao.load(strategy=strategy)
    if df_all.empty:
        return {}

    df_all['timestamp'] = pd.to_datetime(df_all['timestamp']).dt.normalize()

    if strategy == 'monthly':
        df_all = (
            df_all.groupby(['timestamp', 'top_n'])[_KPI_COLS]
            .mean()
            .reset_index()
        )

    sorted_ts = sorted(df_all['timestamp'].unique())
    latest_ts = sorted_ts[-1]
    avg = df_all[df_all['timestamp'] == latest_ts][_KPI_COLS].mean()
    top_ns = sorted(df_all['top_n'].unique())
    result = {'timestamp': latest_ts, 'top_n_min': int(top_ns[0]), 'top_n_max': int(top_ns[-1]), **avg.to_dict()}

    if len(sorted_ts) >= 2:
        prev_ts = sorted_ts[-2]
        prev_avg = df_all[df_all['timestamp'] == prev_ts][_KPI_COLS].mean()
        result['prev'] = prev_avg.to_dict()

    return result


def _kpi_card(title: str, value, is_pct: bool, positive_is_good: bool, delta=None) -> dbc.Col:
    if value is None or pd.isna(value):
        display, color = '—', '#9ca3af'
    else:
        display = f"{value * 100:.2f}%" if is_pct else f"{value:.2f}"
        good = value > 0 if positive_is_good else value > -0.2
        color = '#dc2626' if good else '#059669'  # 台股：紅漲綠跌

    arrow = None
    if delta is not None and not pd.isna(delta):
        if abs(delta) < 1e-6:
            arrow = html.Span('—', style={'color': '#9ca3af', 'fontSize': '16px', 'marginLeft': '6px'})
        elif delta > 0:
            arrow = html.Span('▲', style={'color': '#dc2626', 'fontSize': '16px', 'marginLeft': '6px'})
        else:
            arrow = html.Span('▼', style={'color': '#059669', 'fontSize': '16px', 'marginLeft': '6px'})

    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.P(title, style={
                    'fontSize': '16px', 'color': '#6b7280',
                    'fontWeight': '500', 'marginBottom': '6px',
                    'textTransform': 'uppercase', 'letterSpacing': '0.05em',
                }),
                html.Div([
                    html.H3(display, style={
                        'color': color, 'fontWeight': '700',
                        'marginBottom': '0', 'display': 'inline',
                    }),
                    arrow,
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={'padding': '20px 24px'}),
            style=_CARD_STYLE,
        ),
        xs=6, lg=3,
    )


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


def _build_figure(data: dict, strategy: str, metric: str) -> go.Figure:
    label, is_pct, _ = _METRIC_META[metric]
    fig = go.Figure()

    if not data:
        fig.update_layout(
            height=500,
            title=f"尚無資料（{strategy}）",
            plot_bgcolor='white',
        )
        return fig

    for top_n, df in sorted(data.items()):
        color = _COLORS[(top_n - 1) % len(_COLORS)]
        y = df[metric] * 100 if is_pct else df[metric]
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=y,
            mode='lines+markers',
            name=f'持 {top_n} 檔',
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f'%{{x|%Y-%m-%d}}<br>持 {top_n} 檔: %{{y:.2f}}{"%" if is_pct else ""}<extra></extra>',
        ))

    avg_df = (
        pd.concat([df.set_index('timestamp')[[metric]] for df in data.values()], axis=1)
        .mean(axis=1)
        .reset_index()
    )
    avg_df.columns = ['timestamp', metric]
    avg_y = avg_df[metric] * 100 if is_pct else avg_df[metric]
    fig.add_trace(go.Scatter(
        x=avg_df['timestamp'],
        y=avg_y,
        mode='lines+markers',
        name='平均',
        line=dict(color='#94a3b8', width=3),
        marker=dict(size=6),
        hovertemplate=f'%{{x|%Y-%m-%d}}<br>平均: %{{y:.2f}}{"%" if is_pct else ""}<extra></extra>',
    ))

    fig.update_layout(
        height=500,
        margin=dict(l=60, r=20, t=30, b=40),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title=f'{label} (%)' if is_pct else label,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=16),
        ),
        font=dict(family='system-ui, -apple-system, sans-serif', color='#374151', size=16),
        yaxis=dict(title_font=dict(size=16)),
    )
    all_dates = pd.concat([df['timestamp'] for df in data.values()])
    x_min, x_max = all_dates.min(), all_dates.max()
    delta_days = (x_max - x_min).days
    if delta_days <= 14:
        freq = 'D'
    elif delta_days <= 60:
        freq = 'W-MON'
    elif delta_days <= 180:
        freq = 'MS'
    else:
        freq = 'QS'
    interior = pd.date_range(x_min, x_max, freq=freq).tolist()
    tickvals = sorted(set([x_min] + interior + [x_max]))
    ticktext = [d.strftime('%Y-%m-%d') for d in tickvals]

    if is_pct:
        fig.update_yaxes(ticksuffix='%')
    fig.update_xaxes(
        showgrid=True, gridcolor='#e5e7eb',
        tickmode='array', tickvals=tickvals, ticktext=ticktext,
        tickfont=dict(size=16), tickangle=-30,
    )
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb', zeroline=True, zerolinecolor='#d1d5db', tickfont=dict(size=16))

    return fig


app = dash.Dash(
    __name__,
    server=flask_server,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = 'GoldenAI Backtest Dashboard'

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            #metric-selector .form-check { padding-left: 0; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(
    style={'backgroundColor': _PAGE_BG, 'minHeight': '100vh'},
    children=[
        # Navbar
        html.Div(
            style={
                'backgroundColor': 'white',
                'borderBottom': '1px solid #e5e7eb',
                'padding': '0 24px',
            },
            children=dbc.Container([
                dbc.Row([
                    dbc.Col(
                        html.H5(
                            'GoldenAI 回測績效',
                            className='mb-0 py-3',
                            style={'fontWeight': '600', 'color': '#1a202c', 'fontSize': '24px'},
                        ),
                        width='auto',
                    ),
                    dbc.Col(
                        html.Div([
                            html.A('Weekly 報告',
                                   href='/reports/GoldenAITWStrategyWeekly/',
                                   target='_blank',
                                   className='btn btn-outline-secondary me-2'),
                            html.A('Monthly 報告',
                                   href='/reports/GoldenAITWStrategyMonthly/',
                                   target='_blank',
                                   className='btn btn-outline-secondary'),
                        ], className='d-flex align-items-center h-100'),
                    ),
                ], align='center'),
            ], fluid=True),
        ),

        # Content
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Label(
                        '策略',
                        style={'fontWeight': '500', 'color': '#374151', 'marginBottom': '6px', 'fontSize': '18px'},
                    ),
                    dcc.Dropdown(
                        id='strategy-dropdown',
                        options=[
                            {'label': 'Weekly（週策略）', 'value': 'weekly'},
                            {'label': 'Monthly（月策略 Week 1~4 平均）', 'value': 'monthly'},
                        ],
                        value='weekly',
                        clearable=False,
                    ),
                ], xs=12, md=5, lg=4),
            ], className='mt-4 mb-3'),

            html.Div(id='kpi-row', className='mb-3'),

            dbc.Card([
                dbc.CardBody([
                    dbc.RadioItems(
                        id='metric-selector',
                        options=[
                            {'label': '年化報酬', 'value': 'annual_return'},
                            {'label': 'Sharpe Ratio', 'value': 'sharpe'},
                            {'label': 'Max Drawdown', 'value': 'max_drawdown'},
                            {'label': '勝率', 'value': 'win_ratio'},
                        ],
                        value='annual_return',
                        inputClassName='btn-check',
                        labelClassName='btn btn-outline-secondary',
                        labelCheckedClassName='active',
                        inline=True,
                        className='mb-3',
                    ),
                    dcc.Graph(id='metrics-graph', config={'displayModeBar': False}),
                ], style={'padding': '20px 24px'}),
            ], style=_CARD_STYLE, className='mb-4'),
        ], fluid=True),
    ],
)


@app.callback(
    Output('kpi-row', 'children'),
    Input('strategy-dropdown', 'value'),
)
def update_kpi(strategy):
    kpi = _latest_kpi(strategy)
    if not kpi:
        return []

    ts_str = kpi['timestamp'].strftime('%Y-%m-%d')
    return [
        html.P(
            f'最新回測績效（持 {kpi["top_n_min"]}~{kpi["top_n_max"]} 檔平均）　·　{ts_str}',
            style={'fontSize': '18px', 'color': '#6b7280', 'fontWeight': '600',
                   'marginBottom': '10px', 'letterSpacing': '0.02em'},
        ),
        dbc.Row([
            _kpi_card('年化報酬',   kpi['annual_return'], is_pct=True,  positive_is_good=True,
                      delta=kpi['annual_return'] - kpi['prev']['annual_return'] if 'prev' in kpi else None),
            _kpi_card('Sharpe Ratio', kpi['sharpe'],     is_pct=False, positive_is_good=True,
                      delta=kpi['sharpe'] - kpi['prev']['sharpe'] if 'prev' in kpi else None),
            _kpi_card('Max Drawdown', kpi['max_drawdown'], is_pct=True, positive_is_good=False,
                      delta=kpi['max_drawdown'] - kpi['prev']['max_drawdown'] if 'prev' in kpi else None),
            _kpi_card('勝率',       kpi['win_ratio'],    is_pct=True,  positive_is_good=True,
                      delta=kpi['win_ratio'] - kpi['prev']['win_ratio'] if 'prev' in kpi else None),
        ], className='g-3'),
    ]


@app.callback(
    Output('metrics-graph', 'figure'),
    Input('strategy-dropdown', 'value'),
    Input('metric-selector', 'value'),
)
def update_graph(strategy, metric):
    data = _load_all(strategy)
    return _build_figure(data, strategy, metric)


server = flask_server


if __name__ == '__main__':
    app.run(debug=True, port=8051)
