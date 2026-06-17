import os
import logging
from zoneinfo import ZoneInfo
import dash
from dash import dcc, html, Input, Output, State, ALL, ctx, dash_table
import dash_bootstrap_components as dbc
from flask import Flask, request
import plotly.graph_objects as go
import pandas as pd
from dao.golden_ai_backtest_metrics_dao import GoldenAIBacktestMetricsDAO
from dao.recommendation_dao import RecommendationDAO

logger = logging.getLogger(__name__)

_DB_PATH = os.getenv('GOLDEN_AI_DB_PATH', 'data_prod.db')
_TZ = ZoneInfo('Asia/Taipei')

flask_server = Flask(__name__)

_assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

_REPORT_TEMPLATE = None


def _load_report_template():
    global _REPORT_TEMPLATE
    if _REPORT_TEMPLATE is not None:
        return _REPORT_TEMPLATE
    path = os.path.join(_assets, 'finlab_report_template.html')
    if not os.path.exists(path):
        logger.warning('finlab_report_template.html not found — run scripts/extract_report_template.py first')
        return None
    with open(path, 'r', encoding='utf-8') as f:
        _REPORT_TEMPLATE = f.read()
    return _REPORT_TEMPLATE


@flask_server.route('/report/view')
def report_view():
    strategy = request.args.get('strategy', '')
    timestamp = request.args.get('timestamp', '')
    ranks = request.args.get('ranks')
    week = request.args.get('week') or None

    result = dao.get_report(timestamp, strategy, week=week, ranks=ranks)
    if not result:
        return 'Report not found', 404

    template = _load_report_template()
    if not template:
        return 'Report template not configured. Run: python3 scripts/extract_report_template.py', 500

    report_json, position_json = result
    return template.replace('{{REPORT_JSON}}', report_json).replace('{{POSITION_JSON}}', position_json)


_METRIC_META = {
    'annual_return': ('年化報酬',    True),
    'sharpe':        ('Sharpe Ratio', False),
    'max_drawdown':  ('Max Drawdown', True),
    'win_ratio':     ('勝率',        True),
}

# Tableau 10 (advanced view multi-line chart palette)
_COLORS = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
    '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7',
]

_COLOR = {
    'text_heading':   '#1a202c',
    'text_value':     '#1e293b',
    'text_secondary': '#374151',
    'text_muted':     '#6b7280',
    'text_disabled':  '#9ca3af',
    'accent':         '#1d4ed8',
    'accent_bg':      '#dbeafe',
    'accent_fill':    'rgba(29, 78, 216, 0.12)',
    'up':             '#dc2626',
    'down':           '#059669',
    'border':         '#e5e7eb',
    'grid_zero':      '#d1d5db',
    'bg_page':        '#f0f2f5',
    'bg_table_head':  '#f8fafc',
    'bg_table_alt':   '#f9fafb',
    'transparent':    'rgba(0,0,0,0)',
}

_FONT_FAMILY = 'system-ui, -apple-system, sans-serif'

_BORDER = f"1px solid {_COLOR['border']}"

_TYPO = {
    'card_title':    {'fontSize': '18px', 'fontWeight': '600', 'color': _COLOR['text_heading']},
    'section_label': {'fontSize': '18px', 'fontWeight': '500', 'color': _COLOR['text_secondary']},
    'form_label':    {'fontWeight': '500', 'color': _COLOR['text_secondary'], 'marginBottom': '6px', 'display': 'block'},
    'muted':         {'fontSize': '13px', 'color': _COLOR['text_muted']},
}

_CARD_STYLE = {
    'boxShadow': '0 1px 4px rgba(0,0,0,0.08)',
    'borderRadius': '8px',
    'border': 'none',
}

_CARD_BODY_STYLE = {'padding': '20px 24px'}

_TABLE_STYLE = {
    'style_table': {'overflowX': 'auto'},
    'style_cell': {
        'fontFamily': _FONT_FAMILY,
        'fontSize': '14px',
        'padding': '10px 16px',
        'textAlign': 'left',
        'color': _COLOR['text_secondary'],
        'border': _BORDER,
    },
    'style_header': {
        'backgroundColor': _COLOR['bg_table_head'],
        'fontWeight': '600',
        'border': _BORDER,
        'color': _COLOR['text_secondary'],
    },
    'style_data_conditional': [
        {'if': {'row_index': 'odd'}, 'backgroundColor': _COLOR['bg_table_alt']},
    ],
}

dao = GoldenAIBacktestMetricsDAO(db_path=_DB_PATH)
_KPI_COLS = ['annual_return', 'sharpe', 'max_drawdown', 'win_ratio']

_REC_DAOS = {
    'weekly':    RecommendationDAO(_DB_PATH, frequency='weekly'),
    'monthly':   RecommendationDAO(_DB_PATH, frequency='monthly'),
    'weekly_4w': RecommendationDAO(_DB_PATH, frequency='weekly'),  # 共用 weekly 推薦清單
}

_STRATEGY_META = {
    'weekly':    {'label': '週策略'},
    'monthly':   {'label': '月策略'},
    'weekly_4w': {'label': '週策略（4週）'},
}

_PERIOD_MONTHS = {'1M': 1, '3M': 3, '6M': 6, '1Y': 12, 'All': None}


def _longest_ranks(ranks_iter):
    """Pick the ranks string with the most members. For the 1~8 powerset stored in DB,
    this is the canonical "full set" `1,2,3,4,5,6,7,8` — callers use this as the
    default highlighted line in charts and KPI cards."""
    return max(ranks_iter, key=lambda r: len(r.split(',')))


def _normalized(strategy: str) -> pd.DataFrame:
    """Load metrics for a strategy, normalize timestamp, and monthly-aggregate.

    Both `_latest_kpi` and `_load_all` consume this — callbacks should fetch
    once and pass the result in to avoid hitting the DB twice.
    """
    df_all = dao.load(strategy=strategy)
    if df_all.empty:
        return df_all

    df_all['timestamp'] = pd.to_datetime(df_all['timestamp']).dt.normalize()
    if strategy in ('monthly', 'weekly_4w'):
        df_all = (
            df_all.groupby(['timestamp', 'ranks'])[_KPI_COLS]
            .mean()
            .reset_index()
        )
    return df_all


def _latest_kpi(strategy: str, df_normalized=None) -> dict:
    if df_normalized is None:
        df_normalized = _normalized(strategy)
    if df_normalized.empty:
        return {}

    full_ranks = _longest_ranks(df_normalized['ranks'].unique())
    df_full = df_normalized[df_normalized['ranks'] == full_ranks]

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


_KPI_CARD_SIZES = {
    'compact': {
        'title': {
            'fontSize': '16px', 'marginBottom': '6px',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        },
        'value_tag': html.H3,
        'value': {'fontWeight': '700', 'marginBottom': '0', 'display': 'inline'},
        'arrow_size': '16px',
        'arrow_ml': '6px',
        'body_padding': '20px 24px',
    },
    'hero': {
        'title': {'fontSize': '15px', 'marginBottom': '14px'},
        'value_tag': html.Span,
        'value': {'fontWeight': '700', 'fontSize': '40px', 'lineHeight': '1'},
        'arrow_size': '24px',
        'arrow_ml': '10px',
        'body_padding': '24px 28px',
    },
}


_KPI_HERO_LABEL_OVERRIDES = {
    # In hero view (simple page), use shorter / friendlier Chinese for these
    'sharpe': '夏普值',
    'max_drawdown': '最大回檔',
}


_ARROW_SPEC = {
    # symbol, color key in _COLOR, bold
    'flat': ('－', 'text_disabled', True),
    'up':   ('▲', 'up',             False),
    'down': ('▼', 'down',           False),
}


def _kpi_title(metric: str, size: str) -> str:
    base = _METRIC_META[metric][0]
    if size == 'hero':
        return _KPI_HERO_LABEL_OVERRIDES.get(metric, base)
    return base


def _kpi_header(kpi: dict) -> html.Div:
    """Header strip above KPI cards: '第 1~8 支績效' on the left, latest-backtest date on the right."""
    return html.Div([
        html.Div(f'{_rank_label(kpi["full_ranks"])}績效', style=_TYPO['section_label']),
        html.Div(f'最新回測：{kpi["timestamp"].strftime("%Y-%m-%d")}', style=_TYPO['muted']),
    ], className='d-flex justify-content-between align-items-baseline mb-2')


def _render_kpi_row(kpi: dict, size: str) -> dbc.Row:
    def _delta(k):
        return kpi[k] - kpi['prev'][k] if 'prev' in kpi else None
    return dbc.Row(
        [
            _kpi_card(
                _kpi_title(k, size),
                kpi[k],
                is_pct=_METRIC_META[k][1],
                delta=_delta(k),
                size=size,
            )
            for k in _KPI_COLS
        ],
        className='g-3',
    )


def _kpi_card(title: str, value, is_pct: bool, delta=None, size: str = 'compact') -> dbc.Col:
    spec = _KPI_CARD_SIZES[size]

    if pd.isna(value):
        display, color = '—', _COLOR['text_disabled']
    else:
        display = f"{value * 100:.2f}%" if is_pct else f"{value:.2f}"
        color = _COLOR['text_value']

    arrow = None
    if delta is not None and not pd.isna(delta):
        direction = 'flat' if abs(delta) < 1e-6 else ('up' if delta > 0 else 'down')
        symbol, color_key, bold = _ARROW_SPEC[direction]
        arrow_style = {
            'fontSize': spec['arrow_size'],
            'marginLeft': spec['arrow_ml'],
            'color': _COLOR[color_key],
        }
        if bold:
            arrow_style['fontWeight'] = '700'
        arrow = html.Span(symbol, style=arrow_style)

    title_style = {'color': _COLOR['text_muted'], 'fontWeight': '500', **spec['title']}
    value_style = {'color': color, **spec['value']}

    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.P(title, style=title_style),
                html.Div([
                    spec['value_tag'](display, style=value_style),
                    arrow,
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={'padding': spec['body_padding']}),
            style=_CARD_STYLE,
        ),
        xs=6, lg=3,
    )


def _load_all(strategy: str, months=3, df_normalized=None) -> dict:
    if df_normalized is None:
        df_normalized = _normalized(strategy)
    if df_normalized.empty:
        return {}

    cutoff = (
        pd.Timestamp.now(tz=_TZ).normalize().tz_localize(None) - pd.DateOffset(months=months)
        if months is not None else None
    )
    result = {}
    for ranks in df_normalized['ranks'].unique():
        df = df_normalized[df_normalized['ranks'] == ranks]
        if cutoff is not None:
            df = df[df['timestamp'] >= cutoff]
        if not df.empty:
            result[ranks] = df.sort_values('timestamp')

    return result


def _rank_label(ranks_str: str) -> str:
    nums = list(map(int, ranks_str.split(',')))
    if len(nums) == 1:
        return f'第 {nums[0]} 支'
    if nums == list(range(1, len(nums) + 1)):
        return f'第 1~{len(nums)} 支'
    return f'第 {", ".join(map(str, nums))} 支'


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
            'backgroundColor': _COLOR['accent_bg'],
            'color': _COLOR['accent'],
            'borderRadius': '12px',
            'padding': '4px 10px',
            'display': 'inline-flex',
            'alignItems': 'center',
        }))
    return tags


def _ranks_sort_key(r: str):
    nums = list(map(int, r.split(',')))
    is_consec_from_1 = nums == list(range(1, len(nums) + 1))
    return (0 if is_consec_from_1 else 1, len(nums), nums)


def _compute_date_ticks(x_min, x_max):
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
    return tickvals, ticktext


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
        paper_bgcolor=_COLOR['transparent'],
        yaxis_title=f'{label} (%)' if is_pct else label,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=16),
        ),
        font=dict(family=_FONT_FAMILY, color=_COLOR['text_secondary'], size=16),
        yaxis=dict(title_font=dict(size=16)),
    )
    all_dates = pd.concat([df['timestamp'] for df in data.values()])
    tickvals, ticktext = _compute_date_ticks(all_dates.min(), all_dates.max())

    if is_pct:
        fig.update_yaxes(ticksuffix='%')
    fig.update_xaxes(
        showgrid=True, gridcolor=_COLOR['border'],
        tickmode='array', tickvals=tickvals, ticktext=ticktext,
        tickfont=dict(size=16), tickangle=-30,
    )
    fig.update_yaxes(showgrid=True, gridcolor=_COLOR['border'], zeroline=True, zerolinecolor=_COLOR['grid_zero'], tickfont=dict(size=16))

    return fig


def _build_simple_figure(df, metric: str) -> go.Figure:
    label, is_pct = _METRIC_META[metric]
    fig = go.Figure()

    if df is None or df.empty:
        fig.update_layout(height=400, title='尚無資料', plot_bgcolor='white')
        return fig

    y = df[metric] * 100 if is_pct else df[metric]

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=y,
        mode='lines',
        line=dict(color=_COLOR['accent'], width=2.5),
        fill='tozeroy',
        fillcolor=_COLOR['accent_fill'],
        hovertemplate=f'%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}{"%" if is_pct else ""}<extra></extra>',
        showlegend=False,
    ))

    last_x = df['timestamp'].iloc[-1]
    last_y = y.iloc[-1]
    last_text = f'{last_y:.2f}%' if is_pct else f'{last_y:.2f}'

    fig.add_trace(go.Scatter(
        x=[last_x], y=[last_y],
        mode='markers',
        marker=dict(size=10, color=_COLOR['accent']),
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.add_annotation(
        x=last_x, y=last_y,
        text=f'<b>{last_text}</b>',
        xanchor='left', yanchor='middle',
        xshift=14,
        showarrow=False,
        font=dict(size=15, color=_COLOR['accent']),
    )

    fig.add_hline(
        y=0,
        line=dict(color=_COLOR['text_disabled'], width=1, dash='dash'),
    )

    fig.update_layout(
        height=400,
        margin=dict(l=60, r=90, t=20, b=40),
        plot_bgcolor='white',
        paper_bgcolor=_COLOR['transparent'],
        yaxis_title=f'{label} (%)' if is_pct else label,
        font=dict(family=_FONT_FAMILY, color=_COLOR['text_secondary'], size=14),
        yaxis=dict(title_font=dict(size=14)),
    )

    tickvals, ticktext = _compute_date_ticks(df['timestamp'].min(), df['timestamp'].max())

    if is_pct:
        fig.update_yaxes(ticksuffix='%')
    fig.update_xaxes(
        showgrid=True, gridcolor=_COLOR['border'],
        tickmode='array', tickvals=tickvals, ticktext=ticktext,
        tickfont=dict(size=13), tickangle=-30,
    )
    fig.update_yaxes(showgrid=True, gridcolor=_COLOR['border'], tickfont=dict(size=13))

    return fig


def _query_report_list(strategy: str) -> pd.DataFrame:
    df = dao.list_reports(strategy)
    if df.empty:
        return pd.DataFrame(columns=['date', 'timestamp', 'ranks', 'week'])
    df['date'] = df['timestamp'].str[:10]
    df['week'] = df['week'].fillna('')
    df = df.drop_duplicates(subset=['date', 'ranks', 'week'], keep='first')
    df = df.sort_values(['date', 'ranks', 'week'], ascending=[False, True, True]).reset_index(drop=True)
    return df[['date', 'timestamp', 'ranks', 'week']]


def _recommendation_card(strategy: str):
    record = _REC_DAOS[strategy].get_latest()

    subtitle_text = f'更新於 {record.date}' if record else ''

    header_row = html.Div([
        html.Div('本週推薦清單', style=_TYPO['card_title']),
        html.Div(subtitle_text, style=_TYPO['muted']),
    ], className='d-flex justify-content-between align-items-center flex-wrap gap-2')

    header_children = [header_row]
    if strategy == 'monthly':
        header_children.append(
            html.Div(
                '月策略每 4 週換倉一次。此處顯示 AI 對月線最新觀點，實際進場時間由策略決定。',
                style={**_TYPO['muted'], 'fontStyle': 'italic', 'marginTop': '4px'},
            )
        )
    elif strategy == 'weekly_4w':
        header_children.append(
            html.Div(
                '此策略採用週推薦清單、買進後持有約 4 週。此處顯示最新一期週清單，實際進場時間由策略決定。',
                style={**_TYPO['muted'], 'fontStyle': 'italic', 'marginTop': '4px'},
            )
        )
    header = html.Div(header_children, className='mb-3')

    if record and record.stocks:
        rows = [
            {
                '#': i + 1,
                '代號': s.id,
                '名稱': s.name or '',
                '目標價': f'{s.TP:,.2f}' if s.TP is not None else '—',
            }
            for i, s in enumerate(record.stocks)
        ]
        body = dash_table.DataTable(
            columns=[
                {'name': '#',     'id': '#'},
                {'name': '代號',  'id': '代號'},
                {'name': '名稱',  'id': '名稱'},
                {'name': '目標價', 'id': '目標價'},
            ],
            data=rows,
            **_TABLE_STYLE,
            style_cell_conditional=[
                {'if': {'column_id': '#'},      'width': '50px',  'textAlign': 'center'},
                {'if': {'column_id': '代號'},   'width': '90px',  'textAlign': 'center'},
                {'if': {'column_id': '目標價'}, 'width': '110px', 'textAlign': 'right'},
            ],
        )
    else:
        body = html.Div(
            '目前無推薦資料',
            className='text-muted text-center py-4',
            style={'fontSize': '14px'},
        )

    return dbc.Card(
        dbc.CardBody([header, body], style=_CARD_BODY_STYLE),
        style=_CARD_STYLE,
        className='mb-3',
    )


def _simple_layout():
    strategy_options = [
        {
            'label': html.Span([
                html.Span(_STRATEGY_META[s]['label'], style={'fontWeight': '600', 'display': 'block', 'fontSize': '15px'}),
                html.Span(sub, style={'fontSize': '12px', 'opacity': '0.75'}),
            ]),
            'value': s,
        }
        for s, sub in [('weekly', '週清單 · 持有 1 週'), ('weekly_4w', '週清單 · 持有 4 週'), ('monthly', '月清單 · 持有 4 週')]
    ]

    period_options = [{'label': k, 'value': k} for k in _PERIOD_MONTHS]

    return dbc.Container([
        dbc.RadioItems(
            id='simple-strategy',
            options=strategy_options,
            value='weekly',
            inputClassName='btn-check',
            labelClassName='btn btn-outline-secondary',
            labelCheckedClassName='active',
            inline=True,
            className='mt-4 mb-2',
        ),

        html.Div(id='simple-kpi-row', className='mb-3 mt-2'),

        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Div('年化報酬走勢', style=_TYPO['card_title']),
                    dbc.RadioItems(
                        id='simple-period',
                        options=period_options,
                        value='3M',
                        inputClassName='btn-check',
                        labelClassName='btn btn-outline-secondary btn-sm',
                        labelCheckedClassName='active',
                        inline=True,
                    ),
                ], className='d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2'),
                dcc.Graph(id='simple-graph', config={'displayModeBar': False}),
            ], style=_CARD_BODY_STYLE),
        ], style=_CARD_STYLE, className='mb-3'),

        html.Div(id='simple-recommendations', className='mb-4'),
    ], fluid=True)


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
                    html.Label('策略', style={**_TYPO['section_label'], 'marginBottom': '6px'}),
                    dcc.Dropdown(
                        id='strategy-dropdown',
                        options=[
                            {'label': 'Weekly（週清單 · 持有 1 週）', 'value': 'weekly'},
                            {'label': 'Weekly 4W（週清單 · 持有 4 週，Week 1~4 平均）', 'value': 'weekly_4w'},
                            {'label': 'Monthly（月清單 · 持有 4 週，Week 1~4 平均）', 'value': 'monthly'},
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
                            {'label': label, 'value': key}
                            for key, (label, _) in _METRIC_META.items()
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
                ], style=_CARD_BODY_STYLE),
            ], style=_CARD_STYLE, className='mb-4'),

            html.Div(id='advanced-recommendations', className='mb-4'),
        ], fluid=True),
    ])


def _report_browser_layout(strategy: str):
    title = f'{_STRATEGY_META[strategy]["label"]}回測報告'
    df = _query_report_list(strategy)
    latest_date = df['date'].iloc[0] if not df.empty else str(pd.Timestamp.today().date())
    return dbc.Container([
        dbc.Row(
            dbc.Col(
                dcc.Link('← 返回', href='/advanced', className='btn btn-outline-secondary btn-sm'),
            ),
            className='mt-4 mb-3',
        ),
        html.H5(
            title,
            style={'fontWeight': '600', 'color': _COLOR['text_heading'], 'fontSize': '20px', 'marginBottom': '20px'},
        ),
        dbc.Row([
            dbc.Col([
                html.Label('日期範圍', style=_TYPO['form_label']),
                dcc.DatePickerRange(
                    id='report-date-range',
                    start_date=latest_date,
                    end_date=latest_date,
                    display_format='YYYY-MM-DD',
                    clearable=True,
                ),
            ], xs=12, md='auto', className='mb-3'),
            dbc.Col([
                html.Label('Rank 組合', style=_TYPO['form_label']),
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
                    **_TABLE_STYLE,
                    style_cell_conditional=[
                        {'if': {'column_id': 'date'},       'width': '130px'},
                        {'if': {'column_id': 'link'},       'width': '80px', 'textAlign': 'center'},
                    ],
                ),
                style=_CARD_BODY_STYLE,
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
            #metric-selector .form-check,
            #simple-strategy .form-check,
            #simple-period .form-check { padding-left: 0; }
            #report-rank-filter .Select-control { min-height: 44px !important; }
            #report-rank-filter .Select-arrow-zone,
            #report-rank-filter .Select-clear-zone { vertical-align: middle !important; }
            /* Pill RadioItems active state: use accent blue instead of FLATLY's gray.
               Color matches _COLOR['accent'] — update both if the accent changes. */
            #simple-strategy .btn-check:checked + .btn-outline-secondary,
            #simple-period   .btn-check:checked + .btn-outline-secondary,
            #metric-selector .btn-check:checked + .btn-outline-secondary,
            #simple-strategy .btn-outline-secondary.active,
            #simple-period   .btn-outline-secondary.active,
            #metric-selector .btn-outline-secondary.active {
                background-color: #1d4ed8;
                border-color: #1d4ed8;
                color: #ffffff;
            }
            #simple-strategy .btn-outline-secondary:hover,
            #simple-period   .btn-outline-secondary:hover,
            #metric-selector .btn-outline-secondary:hover {
                background-color: #dbeafe;
                border-color: #1d4ed8;
                color: #1d4ed8;
            }
            #simple-strategy .btn-check:checked + .btn-outline-secondary:hover,
            #simple-period   .btn-check:checked + .btn-outline-secondary:hover,
            #metric-selector .btn-check:checked + .btn-outline-secondary:hover,
            #simple-strategy .btn-outline-secondary.active:hover,
            #simple-period   .btn-outline-secondary.active:hover,
            #metric-selector .btn-outline-secondary.active:hover {
                background-color: #1d4ed8;
                border-color: #1d4ed8;
                color: #ffffff;
            }
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
    style={'backgroundColor': _COLOR['bg_page'], 'minHeight': '100vh'},
    children=[
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='displayed-ranks', data=None),

        # Navbar
        html.Div(
            id='navbar',
            style={
                'backgroundColor': 'white',
                'borderBottom': _BORDER,
                'padding': '0 24px',
            },
        ),

        # Page content
        html.Div(id='page-content'),
    ],
)


# Flask SPA routes
@flask_server.route('/advanced')
@flask_server.route('/advanced/')
@flask_server.route('/reports/weekly/')
@flask_server.route('/reports/monthly/')
@flask_server.route('/reports/weekly-4w/')
def spa_routes():
    return app.index()


def _canon_path(p: str) -> str:
    """Strip trailing slash for route comparison, except for root."""
    if not p or p == '/':
        return '/'
    return p.rstrip('/') or '/'


_REPORT_PATH_TO_STRATEGY = {
    '/reports/weekly':    'weekly',
    '/reports/monthly':   'monthly',
    '/reports/weekly-4w': 'weekly_4w',
}


# ── Navbar ───────────────────────────────────────────────────────────────────

_TOGGLE_LINK_STYLE = {
    'color': _COLOR['text_muted'],
    'textDecoration': 'none',
    'fontSize': '14px',
    'fontWeight': '500',
}


@app.callback(
    Output('navbar', 'children'),
    Input('url', 'pathname'),
)
def render_navbar(pathname):
    p = _canon_path(pathname)
    is_advanced = p == '/advanced'
    is_reports = p.startswith('/reports/')

    brand_text = 'GoldenAI 回測績效' if (is_advanced or is_reports) else 'GoldenAI 策略表現'
    brand = dcc.Link(
        brand_text,
        href='/',
        style={
            'fontWeight': '600', 'color': _COLOR['text_heading'],
            'fontSize': '24px', 'textDecoration': 'none',
        },
    )

    if is_advanced:
        center = html.Div([
            dcc.Link('Weekly 報告', href='/reports/weekly/',
                     className='btn btn-outline-secondary me-2'),
            dcc.Link('Monthly 報告', href='/reports/monthly/',
                     className='btn btn-outline-secondary me-2'),
            dcc.Link('Weekly 4W 報告', href='/reports/weekly-4w/',
                     className='btn btn-outline-secondary'),
        ], className='d-flex align-items-center')
    else:
        center = None

    if is_reports:
        toggle = None
    elif is_advanced:
        toggle = dcc.Link('簡易檢視 →', href='/', style=_TOGGLE_LINK_STYLE)
    else:
        toggle = dcc.Link('進階檢視 →', href='/advanced', style=_TOGGLE_LINK_STYLE)

    left_children = [brand]
    if center is not None:
        left_children.append(center)
    left_group = html.Div(
        left_children,
        style={'display': 'flex', 'alignItems': 'center', 'gap': '32px'},
    )

    return dbc.Container([
        html.Div(
            [left_group, toggle if toggle else html.Div()],
            style={
                'display': 'flex', 'alignItems': 'center',
                'justifyContent': 'space-between',
                'padding': '16px 0',
            },
        ),
    ], fluid=True)


# ── Routing ──────────────────────────────────────────────────────────────────

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
)
def render_page(pathname):
    p = _canon_path(pathname)
    if p in _REPORT_PATH_TO_STRATEGY:
        return _report_browser_layout(_REPORT_PATH_TO_STRATEGY[p])
    if p == '/advanced':
        return _main_layout()
    return _simple_layout()


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
    strategy = _REPORT_PATH_TO_STRATEGY.get(_canon_path(pathname), 'weekly')
    df = _query_report_list(strategy)

    all_ranks = sorted(df['ranks'].unique(), key=_ranks_sort_key) if not df.empty else []
    options = [{'label': _rank_label(r), 'value': r} for r in all_ranks]

    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    if rank_filter:
        df = df[df['ranks'].isin(rank_filter)]

    table_data = []
    for _, row in df.iterrows():
        params = f'strategy={strategy}&timestamp={row["timestamp"]}&ranks={row["ranks"]}'
        if row['week']:
            params += f'&week={row["week"]}'
        table_data.append({
            'date':       row['date'],
            'rank_label': _rank_label(row['ranks']) + (f' · {row["week"]}' if row['week'] else ''),
            'link':       f'[開啟](/report/view?{params})',
        })
    return table_data, options


# ── Simple dashboard ─────────────────────────────────────────────────────────


@app.callback(
    Output('simple-kpi-row', 'children'),
    Output('simple-graph', 'figure'),
    Input('simple-strategy', 'value'),
    Input('simple-period', 'value'),
)
def update_simple_view(strategy, period):
    strategy = strategy or 'weekly'
    period = period or '3M'

    df_norm = _normalized(strategy)
    kpi = _latest_kpi(strategy, df_normalized=df_norm)
    if not kpi:
        empty = html.Div(
            '目前尚無回測資料',
            className='text-center text-muted py-4',
            style={'fontSize': '15px'},
        )
        return empty, _build_simple_figure(None, 'annual_return')

    months = _PERIOD_MONTHS.get(period, 3)
    data = _load_all(strategy, months=months, df_normalized=df_norm)
    chart_df = None
    if data:
        full_ranks = _longest_ranks(data.keys())
        chart_df = data[full_ranks]

    fig = _build_simple_figure(chart_df, 'annual_return')

    return [_kpi_header(kpi), _render_kpi_row(kpi, size='hero')], fig


@app.callback(
    Output('simple-recommendations', 'children'),
    Input('simple-strategy', 'value'),
)
def update_simple_recommendations(strategy):
    return _recommendation_card(strategy or 'weekly')


# ── Main dashboard ────────────────────────────────────────────────────────────

@app.callback(
    Output('kpi-row', 'children'),
    Input('strategy-dropdown', 'value'),
)
def update_kpi(strategy):
    kpi = _latest_kpi(strategy)
    if not kpi:
        return []
    return [_kpi_header(kpi), _render_kpi_row(kpi, size='compact')]


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

    d = _load_all(strategy)

    def _fig(ranks_list):
        filtered = {r: df for r, df in d.items() if r in (ranks_list or [])}
        return _build_figure(filtered, metric)

    if current_ranks is None:
        full = _longest_ranks(d.keys()) if d else None
        new_ranks = [full] if full else []
        return new_ranks, _build_tags(new_ranks), _fig(new_ranks)

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
            for r in sorted(data.keys(), key=_ranks_sort_key)
        ]
        return True, options, current_ranks or []
    return False, dash.no_update, dash.no_update


@app.callback(
    Output('advanced-recommendations', 'children'),
    Input('strategy-dropdown', 'value'),
)
def update_advanced_recommendations(strategy):
    return _recommendation_card(strategy or 'weekly')


server = flask_server


if __name__ == '__main__':
    app.run(debug=True, port=8051)
