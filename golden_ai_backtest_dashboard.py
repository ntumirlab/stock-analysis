import os
import re
import dash
from dash import dcc, html, Input, Output, State, ALL, ctx, dash_table
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


@flask_server.route('/reports/<path:path>')
def autoindex(path):
    if path.split('/')[0] not in _ALLOWED_DIRS:
        return 'Forbidden', 403
    return _ai.render_autoindex(path)


_METRIC_META = {
    'annual_return': ('年化報酬',    True),
    'sharpe':        ('Sharpe Ratio', False),
    'max_drawdown':  ('Max Drawdown', True),
    'win_ratio':     ('勝率',        True),
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
            df_all.groupby(['timestamp', 'ranks'])[_KPI_COLS]
            .mean()
            .reset_index()
        )

    full_ranks = max(df_all['ranks'].unique(), key=lambda r: len(r.split(',')))
    df_full = df_all[df_all['ranks'] == full_ranks]

    sorted_ts = sorted(df_full['timestamp'].unique())
    if not sorted_ts:
        return {}
    latest_ts = sorted_ts[-1]
    avg = df_full[df_full['timestamp'] == latest_ts][_KPI_COLS].mean()
    result = {'timestamp': latest_ts, 'full_ranks': full_ranks, **avg.to_dict()}

    if len(sorted_ts) >= 2:
        prev_ts = sorted_ts[-2]
        prev_avg = df_full[df_full['timestamp'] == prev_ts][_KPI_COLS].mean()
        result['prev'] = prev_avg.to_dict()

    return result


def _kpi_card(title: str, value, is_pct: bool, delta=None) -> dbc.Col:
    if pd.isna(value):
        display, color = '—', '#9ca3af'
    else:
        display = f"{value * 100:.2f}%" if is_pct else f"{value:.2f}"
        color = '#1e293b'

    arrow = None
    if delta is not None and not pd.isna(delta):
        if abs(delta) < 1e-6:
            arrow = html.Span('－', style={'color': '#9ca3af', 'fontSize': '16px', 'marginLeft': '6px', 'fontWeight': '700'})
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

    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=3)
    result = {}
    for ranks in df_all['ranks'].unique():
        df = df_all[df_all['ranks'] == ranks].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
        df = df[df['timestamp'] >= cutoff]

        if strategy == 'monthly':
            df = (
                df.groupby('timestamp')[['annual_return', 'sharpe', 'sortino', 'max_drawdown', 'win_ratio']]
                .mean()
                .reset_index()
            )

        if not df.empty:
            result[ranks] = df.sort_values('timestamp')

    return result


def _rank_label(ranks_str: str) -> str:
    nums = list(map(int, ranks_str.split(',')))
    if len(nums) == 1:
        return f'第 {nums[0]} 支'
    if nums == list(range(1, len(nums) + 1)):
        return f'第 1~{len(nums)} 支'
    return '第 ' + ', '.join(map(str, nums)) + ' 支'


def _build_tags(ranks_list: list) -> list:
    tags = []
    for ranks in (ranks_list or []):
        tags.append(html.Div([
            html.Span(_rank_label(ranks), style={'fontSize': '13px', 'marginRight': '4px'}),
            html.Span(
                '×',
                id={'type': 'remove-rank', 'index': ranks},
                n_clicks=0,
                style={'cursor': 'pointer', 'fontWeight': '700', 'fontSize': '14px', 'lineHeight': '1'},
            ),
        ], style={
            'backgroundColor': '#dbeafe',
            'color': '#1d4ed8',
            'borderRadius': '12px',
            'padding': '4px 10px',
            'display': 'inline-flex',
            'alignItems': 'center',
        }))
    return tags


def _modal_sort_key(r: str):
    nums = list(map(int, r.split(',')))
    is_consec_from_1 = nums == list(range(1, len(nums) + 1))
    return (0 if is_consec_from_1 else 1, len(nums), nums)


def _build_figure(data: dict, metric: str) -> go.Figure:
    label, is_pct = _METRIC_META[metric]
    fig = go.Figure()

    if not data:
        fig.update_layout(height=500, title='尚無資料', plot_bgcolor='white')
        return fig

    for i, (ranks, df) in enumerate(sorted(data.items())):
        color = _COLORS[i % len(_COLORS)]
        y = df[metric] * 100 if is_pct else df[metric]
        rl = _rank_label(ranks)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=y,
            mode='lines+markers',
            name=rl,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f'%{{x|%Y-%m-%d}}<br>{rl}: %{{y:.2f}}{"%" if is_pct else ""}<extra></extra>',
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


_REPORT_FILE_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_Ranks\[(.+)\]\.html$'
)


def _parse_report_files(strategy: str) -> pd.DataFrame:
    dir_name = 'GoldenAITWStrategyWeekly' if strategy == 'weekly' else 'GoldenAITWStrategyMonthly'
    report_dir = os.path.join(_assets, dir_name)
    if not os.path.isdir(report_dir):
        return pd.DataFrame(columns=['date', 'ranks', 'fname'])

    rows = []
    for fname in os.listdir(report_dir):
        m = _REPORT_FILE_PATTERN.match(fname)
        if not m:
            continue
        date_str, time_str, ranks_str = m.groups()
        rows.append({
            'date': date_str,
            'datetime_key': f'{date_str}_{time_str}',
            'ranks': ranks_str,
            'fname': fname,
        })

    if not rows:
        return pd.DataFrame(columns=['date', 'ranks', 'fname'])

    df = pd.DataFrame(rows)
    df = df.sort_values('datetime_key', ascending=False)
    df = df.drop_duplicates(subset=['date', 'ranks'], keep='first')
    df = df.sort_values(['date', 'ranks'], ascending=[False, True]).reset_index(drop=True)
    return df[['date', 'ranks', 'fname']]


def _main_layout():
    return html.Div([
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('選擇 Rank 組合')),
            dbc.ModalBody(
                dcc.Dropdown(
                    id='rank-picker',
                    multi=True,
                    placeholder='搜尋或選擇 Rank 組合...',
                    optionHeight=35,
                )
            ),
            dbc.ModalFooter([
                dbc.Button('取消', id='cancel-rank-modal', color='secondary', outline=True, className='me-2'),
                dbc.Button('確認', id='confirm-rank-modal', color='primary'),
            ]),
        ], id='rank-modal', is_open=False, size='lg'),
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
                    html.Div([
                        html.Div(id='rank-tags', style={
                            'display': 'inline-flex', 'flexWrap': 'wrap',
                            'gap': '6px', 'alignItems': 'center',
                        }),
                        dbc.Button(
                            '+', id='open-rank-modal',
                            color='secondary', outline=True, size='sm',
                            style={'borderRadius': '20px', 'padding': '2px 14px', 'fontWeight': '600'},
                        ),
                    ], className='mb-3 d-flex align-items-center gap-2 flex-wrap'),
                    dcc.Graph(id='metrics-graph', config={'displayModeBar': False}),
                ], style={'padding': '20px 24px'}),
            ], style=_CARD_STYLE, className='mb-4'),
        ], fluid=True),
    ])


def _report_browser_layout(strategy: str):
    title = 'Weekly 回測報告' if strategy == 'weekly' else 'Monthly 回測報告'
    df = _parse_report_files(strategy)
    latest_date = df['date'].iloc[0] if not df.empty else str(pd.Timestamp.today().date())
    return dbc.Container([
        dbc.Row(
            dbc.Col(
                dcc.Link('← 返回', href='/', className='btn btn-outline-secondary btn-sm'),
            ),
            className='mt-4 mb-3',
        ),
        html.H5(
            title,
            style={'fontWeight': '600', 'color': '#1a202c', 'fontSize': '20px', 'marginBottom': '20px'},
        ),
        dbc.Row([
            dbc.Col([
                html.Label('日期範圍', style={
                    'fontWeight': '500', 'color': '#374151',
                    'marginBottom': '6px', 'display': 'block',
                }),
                dcc.DatePickerRange(
                    id='report-date-range',
                    start_date=latest_date,
                    end_date=latest_date,
                    display_format='YYYY-MM-DD',
                    clearable=True,
                ),
            ], xs=12, md='auto', className='mb-3'),
            dbc.Col([
                html.Label('Rank 組合', style={
                    'fontWeight': '500', 'color': '#374151',
                    'marginBottom': '6px', 'display': 'block',
                }),
                dcc.Dropdown(
                    id='report-rank-filter',
                    placeholder='所有 Rank 組合',
                    multi=True,
                    style={'minWidth': '200px'},
                ),
            ], xs=12, md='auto', className='mb-3'),
        ], className='mb-2 align-items-end'),
        dbc.Card([
            dbc.CardBody(
                dash_table.DataTable(
                    id='report-table',
                    columns=[
                        {'name': '日期',     'id': 'date'},
                        {'name': 'Rank 組合', 'id': 'rank_label'},
                        {'name': '報告',     'id': 'link', 'presentation': 'markdown'},
                    ],
                    data=[],
                    page_size=20,
                    page_action='native',
                    sort_action='native',
                    sort_by=[{'column_id': 'date', 'direction': 'desc'}],
                    markdown_options={'link_target': '_blank'},
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'fontFamily': 'system-ui, -apple-system, sans-serif',
                        'fontSize': '14px',
                        'padding': '10px 16px',
                        'textAlign': 'left',
                        'color': '#374151',
                        'border': '1px solid #e5e7eb',
                    },
                    style_header={
                        'backgroundColor': '#f8fafc',
                        'fontWeight': '600',
                        'border': '1px solid #e5e7eb',
                        'color': '#374151',
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9fafb'},
                    ],
                    style_cell_conditional=[
                        {'if': {'column_id': 'date'},       'width': '130px'},
                        {'if': {'column_id': 'link'},       'width': '80px', 'textAlign': 'center'},
                    ],
                ),
                style={'padding': '20px 24px'},
            ),
        ], style=_CARD_STYLE, className='mb-4'),
    ], fluid=True)


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
            #report-rank-filter .Select-control { min-height: 44px !important; }
            #report-rank-filter .Select-arrow-zone,
            #report-rank-filter .Select-clear-zone { vertical-align: middle !important; }
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
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='displayed-ranks', data=None),

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
                        dcc.Link(
                            'GoldenAI 回測績效',
                            href='/',
                            style={
                                'fontWeight': '600', 'color': '#1a202c',
                                'fontSize': '24px', 'textDecoration': 'none',
                            },
                            className='mb-0 py-3 d-block',
                        ),
                        width='auto',
                    ),
                    dbc.Col(
                        html.Div([
                            dcc.Link('Weekly 報告',
                                     href='/reports/weekly/',
                                     className='btn btn-outline-secondary me-2'),
                            dcc.Link('Monthly 報告',
                                     href='/reports/monthly/',
                                     className='btn btn-outline-secondary'),
                        ], className='d-flex align-items-center h-100'),
                    ),
                ], align='center'),
            ], fluid=True),
        ),

        # Page content
        html.Div(id='page-content'),
    ],
)


# Flask SPA routes
@flask_server.route('/reports/weekly/')
@flask_server.route('/reports/monthly/')
def reports_spa():
    return app.index()


# ── Routing ──────────────────────────────────────────────────────────────────

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
)
def render_page(pathname):
    if pathname == '/reports/weekly/':
        return _report_browser_layout('weekly')
    if pathname == '/reports/monthly/':
        return _report_browser_layout('monthly')
    return _main_layout()


# ── Report browser ────────────────────────────────────────────────────────────

@app.callback(
    Output('report-table', 'data'),
    Output('report-rank-filter', 'options'),
    Input('report-date-range', 'start_date'),
    Input('report-date-range', 'end_date'),
    Input('report-rank-filter', 'value'),
    State('url', 'pathname'),
)
def update_report_table(start_date, end_date, rank_filter, pathname):
    strategy = 'weekly' if pathname == '/reports/weekly/' else 'monthly'
    dir_name = 'GoldenAITWStrategyWeekly' if strategy == 'weekly' else 'GoldenAITWStrategyMonthly'
    df = _parse_report_files(strategy)

    all_ranks = sorted(df['ranks'].unique(), key=_modal_sort_key) if not df.empty else []
    options = [{'label': _rank_label(r), 'value': r} for r in all_ranks]

    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    if rank_filter:
        df = df[df['ranks'].isin(rank_filter)]

    table_data = [
        {
            'date':       row['date'],
            'rank_label': _rank_label(row['ranks']),
            'link':       f'[開啟](/reports/{dir_name}/{row["fname"]})',
        }
        for _, row in df.iterrows()
    ]
    return table_data, options


# ── Main dashboard ────────────────────────────────────────────────────────────

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
            f'最新回測績效　{_rank_label(kpi["full_ranks"])}　·　{ts_str}',
            style={'fontSize': '18px', 'color': '#6b7280', 'fontWeight': '600',
                   'marginBottom': '10px', 'letterSpacing': '0.02em'},
        ),
        dbc.Row([
            _kpi_card('年化報酬',   kpi['annual_return'], is_pct=True,
                      delta=kpi['annual_return'] - kpi['prev']['annual_return'] if 'prev' in kpi else None),
            _kpi_card('Sharpe Ratio', kpi['sharpe'],     is_pct=False,
                      delta=kpi['sharpe'] - kpi['prev']['sharpe'] if 'prev' in kpi else None),
            _kpi_card('Max Drawdown', kpi['max_drawdown'], is_pct=True,
                      delta=kpi['max_drawdown'] - kpi['prev']['max_drawdown'] if 'prev' in kpi else None),
            _kpi_card('勝率',       kpi['win_ratio'],    is_pct=True,
                      delta=kpi['win_ratio'] - kpi['prev']['win_ratio'] if 'prev' in kpi else None),
        ], className='g-3'),
    ]


@app.callback(
    Output('displayed-ranks', 'data'),
    Output('rank-tags', 'children'),
    Output('metrics-graph', 'figure'),
    Input('strategy-dropdown', 'value'),
    Input('metric-selector', 'value'),
    Input('confirm-rank-modal', 'n_clicks'),
    Input({'type': 'remove-rank', 'index': ALL}, 'n_clicks'),
    State('rank-picker', 'value'),
    State('displayed-ranks', 'data'),
)
def update_main(strategy, metric, confirm_n, _remove_n_list, picker_value, current_ranks):
    triggered = ctx.triggered_id
    strategy = strategy or 'weekly'
    metric = metric or 'annual_return'

    def _fig(ranks_list):
        d = _load_all(strategy)
        filtered = {r: df for r, df in d.items() if r in (ranks_list or [])}
        return _build_figure(filtered, metric)

    if current_ranks is None:
        d = _load_all(strategy)
        full = max(d.keys(), key=lambda r: len(r.split(','))) if d else None
        new_ranks = [full] if full else []
        filtered = {r: df for r, df in d.items() if r in new_ranks}
        return new_ranks, _build_tags(new_ranks), _build_figure(filtered, metric)

    if triggered == 'confirm-rank-modal' and confirm_n:
        if picker_value:
            return picker_value, _build_tags(picker_value), _fig(picker_value)
        return dash.no_update, dash.no_update, dash.no_update

    if isinstance(triggered, dict) and triggered.get('type') == 'remove-rank':
        if not any(_remove_n_list):
            # remove-rank components just appeared in layout (n_clicks=0), not actually clicked
            return dash.no_update, dash.no_update, dash.no_update
        rank_to_remove = triggered['index']
        new_ranks = [r for r in current_ranks if r != rank_to_remove]
        return new_ranks, _build_tags(new_ranks), _fig(new_ranks)

    # strategy-dropdown or metric-selector triggered (change or navigation back)
    return dash.no_update, _build_tags(current_ranks), _fig(current_ranks)


@app.callback(
    Output('rank-modal', 'is_open'),
    Output('rank-picker', 'options'),
    Output('rank-picker', 'value'),
    Input('open-rank-modal', 'n_clicks'),
    Input('cancel-rank-modal', 'n_clicks'),
    Input('confirm-rank-modal', 'n_clicks'),
    State('strategy-dropdown', 'value'),
    State('displayed-ranks', 'data'),
    prevent_initial_call=True,
)
def toggle_rank_modal(open_n, _cancel_n, confirm_n, strategy, current_ranks):
    triggered = ctx.triggered_id
    if triggered == 'open-rank-modal' and open_n:
        data = _load_all(strategy)
        options = [
            {'label': _rank_label(r), 'value': r}
            for r in sorted(data.keys(), key=_modal_sort_key)
        ]
        return True, options, current_ranks or []
    return False, dash.no_update, dash.no_update


server = flask_server


if __name__ == '__main__':
    app.run(debug=True, port=8051)
