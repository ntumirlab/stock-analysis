"""
台股多空轉折模型頁面
====================
顯示 TAIEX K 線 + 均線條件式 S4 分數柱狀圖，下方以明細表列出每日各條件的數值與分數。
預設顯示最近 2 個月，可切換時間範圍（最長 1 年）。

評分邏輯見 alan_dashboard/market_regime_model.py。
資料與分數於啟動時計算一次並快取於伺服器記憶體（見 alan_dashboard/cache.py），
每次請求只回傳選定期間的圖表與明細。
"""

from datetime import timedelta

import dash
from dash import dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from finlab import data

from alan_dashboard.cache import PageCache, REFRESH_LABEL
from alan_dashboard.market_regime_model import (
    BREADTH_THRESHOLD, TOP_N, compute_components, listed_common_stocks,
    ma_direction_breadth, top_n_membership,
)
from alan_dashboard.theme import COLOR, FONT, CARD_STYLE, kpi_card

dash.register_page(__name__, path='/', name='多空轉折模型', title='台股多空轉折模型', order=0)

# ── Constants ─────────────────────────────────────────────────────────────────

_SCORE_COLOR = {
    'score_long':    '#ef5350',   # score > 0  → 做多（紅）
    'score_short':   '#26a69a',   # score < -1 → 出場（綠）
    'score_neutral': '#9ca3af',   # -1 ≤ score ≤ 0 → 觀望（灰）
    'candle_up':     '#ef5350',   # K 線漲（紅）— 台灣慣例
    'candle_down':   '#26a69a',   # K 線跌（綠）— 台灣慣例
}

_PERIOD_DAYS = {
    '1M': 30, '2M': 60, '3M': 90,
    '6M': 180, '1Y': 365,
}
_DEFAULT_PERIOD = '2M'

# 市場設定（目前只有 TAIEX，保留結構方便擴充）
# breadth：現貨分的成分股來源；None 表示該市場不計現貨分
_MARKET_CONFIG = {
    'TAIEX': {'label': 'TAIEX（台股）', 'dmi': (35, 21, 18), 'breadth': 'tw_top150'},
    # 未來可新增：'QQQ': {'label': 'QQQ（NASDAQ-100）', 'dmi': (41, 26, 21), 'breadth': None}
}


# ── Data loading & cache ───────────────────────────────────────────────────────

_TAIEX_DATASETS = {
    'open':  'taiex_total_index:開盤指數',
    'high':  'taiex_total_index:最高指數',
    'low':   'taiex_total_index:最低指數',
    'close': 'taiex_total_index:收盤指數',
}


def _load_inputs() -> dict:
    """載入各市場 OHLC（全歷史，指標暖身用）與現貨分所需的個股資料。"""
    taiex = pd.DataFrame({
        col: data.get(name)['TAIEX'] for col, name in _TAIEX_DATASETS.items()
    }).dropna()
    return {
        'TAIEX': taiex,
        'adj_close': data.get('etl:adj_close'),  # 均線方向用還原價，避免除權息造成假跌
        'market_value': data.get('etl:market_value'),
        'categories': data.get('security_categories'),
    }


def _tw_top150_breadth(inputs: dict) -> pd.Series:
    """台灣50 + 台灣中型100（以市值前 150、季度審核與緩衝規則模擬）三線同向檔數差（還原價）。"""
    universe = listed_common_stocks(inputs['categories'])
    membership = top_n_membership(inputs['market_value'], universe, n=TOP_N)
    return ma_direction_breadth(inputs['adj_close'], membership)


def _build(inputs: dict) -> dict:
    """對每個市場計算各條件分數；快取內容為 {market: {'ohlc': DataFrame, 'components': DataFrame}}。"""
    breadth = {'tw_top150': _tw_top150_breadth(inputs)}
    markets = {}
    for market, cfg in _MARKET_CONFIG.items():
        ohlc = inputs[market]
        components = compute_components(ohlc, *cfg['dmi'], breadth_diff=breadth.get(cfg['breadth']))
        markets[market] = {'ohlc': ohlc, 'components': components}
    return {'markets': markets}


_page_cache = PageCache('market_regime', _load_inputs, _build)
_page_cache.refresh()  # 啟動時計算一次（失敗即讓 worker 啟動失敗，與先前行為一致）


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _signal_badge(score_val: int) -> tuple[str, str, str]:
    """回傳 (emoji, 文字, badge color)。"""
    if score_val > 0:
        return '🟢', '做多', 'primary'
    if score_val < -1:
        return '🔴', '出場', 'danger'
    return '⚪', '觀望', 'secondary'


def _signed(v) -> str:
    v = int(v)
    return f'+{v}' if v > 0 else str(v)


def _arrow(direction: int, score: int) -> str:
    """動能欄：方向箭頭 + 實際計分（小計未達門檻時方向成立但不計分，顯示 0）。"""
    sym = {1: '↑', -1: '↓'}.get(int(direction), '—')
    return f'{sym} {_signed(score)}'


_DETAIL_COLUMNS = [
    ('date', '日期'), ('close', '收盤指數'),
    ('ma_above', '站上均線'), ('ma_score', '均線分'),
    ('di_vals', '+DI / −DI'), ('di_score', 'DMI分'),
    ('subtotal', '小計'),
    ('dif', 'DIF'), ('macd', 'MACD'), ('kd', 'KD'),
    ('breadth_diff', '現貨檔數差'), ('breadth_score', '現貨分'),
    ('total', '總分'), ('signal', '訊號'),
]
# 有正負色的欄位：(顯示欄, 判斷正負用的資料欄)
_SIGNED_COLUMNS = [
    ('ma_score', 'ma_score'), ('di_score', 'di_score'), ('subtotal', 'subtotal'),
    ('dif', 'dif_score'), ('macd', 'macd_score'), ('kd', 'kd_score'),
    ('breadth_diff', 'breadth_diff'), ('breadth_score', 'breadth_score'), ('total', 'total'),
]


def _detail_rows(comp: pd.DataFrame) -> list[dict]:
    """明細表資料（最新日在前）；除顯示欄外另帶數值欄供條件式著色。"""
    rows = []
    for d, r in comp[::-1].iterrows():
        above = [str(w) for w, col in ((5, 'above_ma5'), (10, 'above_ma10'), (20, 'above_ma20')) if r[col]]
        emoji, sig_text, _ = _signal_badge(int(r['total']))
        rows.append({
            'date': d.strftime('%Y-%m-%d'),
            'close': f"{r['close']:,.0f}",
            'ma_above': '·'.join(above) if above else '—',
            'ma_score': _signed(r['ma_score']),
            'di_vals': f"{r['plus_di']:.1f} / {r['minus_di']:.1f}",
            'di_score': _signed(r['di_score']),
            'subtotal': _signed(r['subtotal']),
            'dif': _arrow(r['dif_dir'], r['dif_score']),
            'macd': _arrow(r['macd_dir'], r['macd_score']),
            'kd': _arrow(r['kd_dir'], r['kd_score']),
            'breadth_diff': '—' if pd.isna(r['breadth_diff']) else _signed(r['breadth_diff']),
            'breadth_score': _signed(r['breadth_score']),
            'total': _signed(r['total']),
            'signal': f'{emoji} {sig_text}',
            # 著色用數值欄（不顯示）
            'ma_score_v': int(r['ma_score']), 'di_score_v': int(r['di_score']),
            'subtotal_v': int(r['subtotal']),
            'dif_score_v': int(r['dif_score']), 'macd_score_v': int(r['macd_score']),
            'kd_score_v': int(r['kd_score']),
            'breadth_diff_v': 0 if pd.isna(r['breadth_diff']) else int(r['breadth_diff']),
            'breadth_score_v': int(r['breadth_score']),
            'total_v': int(r['total']),
        })
    return rows


def _detail_table(comp: pd.DataFrame) -> dash_table.DataTable:
    colored = []
    for shown, value_col in _SIGNED_COLUMNS:
        colored += [
            {'if': {'filter_query': f'{{{value_col}_v}} > 0', 'column_id': shown},
             'color': _SCORE_COLOR['score_long'], 'fontWeight': '600'},
            {'if': {'filter_query': f'{{{value_col}_v}} < 0', 'column_id': shown},
             'color': _SCORE_COLOR['score_short'], 'fontWeight': '600'},
        ]
    return dash_table.DataTable(
        data=_detail_rows(comp),
        columns=[{'name': name, 'id': cid} for cid, name in _DETAIL_COLUMNS],
        sort_action='native',
        page_size=20,
        style_table={'overflowX': 'auto'},
        style_cell={
            'fontFamily': FONT, 'fontSize': '13px', 'padding': '6px 10px',
            'textAlign': 'center', 'color': COLOR['text_secondary'], 'whiteSpace': 'nowrap',
        },
        style_header={
            'fontWeight': '600', 'backgroundColor': '#f9fafb',
            'color': COLOR['text_heading'],
            'borderBottom': f"1px solid {COLOR['border']}",
        },
        style_data={'borderBottom': f"1px solid {COLOR['border']}"},
        style_data_conditional=colored + [
            {'if': {'column_id': 'total'}, 'backgroundColor': '#f9fafb'},
        ],
    )


# ── Page layout ────────────────────────────────────────────────────────────────

layout = html.Div([
    # ── Control bar ──────────────────────────────────────────────────────────
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='mr-market-dropdown',
                        options=[
                            {'label': v['label'], 'value': k}
                            for k, v in _MARKET_CONFIG.items()
                        ],
                        value='TAIEX',
                        clearable=False,
                        style={'width': '220px', 'fontSize': '14px'},
                    ),
                    width='auto', className='d-flex align-items-center',
                ),
                dbc.Col(
                    dbc.RadioItems(
                        id='mr-period-selector',
                        options=[{'label': k, 'value': k} for k in _PERIOD_DAYS],
                        value=_DEFAULT_PERIOD,
                        inputClassName='btn-check',
                        labelClassName='btn btn-outline-secondary btn-sm',
                        labelCheckedClassName='active',
                        inline=True,
                    ),
                    width='auto', className='d-flex align-items-center',
                ),
            ], className='g-2 align-items-center'),
        ], fluid=True),
    ], style={
        'backgroundColor': 'white',
        'borderBottom': f"1px solid {COLOR['border']}",
        'padding': '12px 0',
        'position': 'sticky', 'top': 0, 'zIndex': 100,
    }),

    # ── Main content ──────────────────────────────────────────────────────────
    dbc.Container([
        # Signal status cards
        dbc.Row([
            dbc.Col(html.Div(id='mr-kpi-date'), width=4),
            dbc.Col(html.Div(id='mr-kpi-score'), width=4),
            dbc.Col(html.Div(id='mr-kpi-signal'), width=4),
        ], className='g-2 mt-2'),

        # Chart
        dbc.Card([
            dbc.CardBody(
                dcc.Graph(
                    id='mr-main-chart',
                    config={'displayModeBar': False, 'scrollZoom': False},
                    style={'height': '72vh'},
                ),
                style={'padding': '8px'},
            ),
        ], style={**CARD_STYLE, 'marginTop': '12px'}),

        # Detail table
        dbc.Card([
            dbc.CardBody([
                html.Div('分數明細', style={
                    'fontSize': '14px', 'fontWeight': '600',
                    'color': COLOR['text_heading'], 'margin': '4px 0 2px 8px',
                }),
                html.Div(
                    '均線分 + DMI分 = 小計；小計 < 5 時 DIF／MACD／KD 下彎各 −1，小計 > −5 時上彎各 +1'
                    '（箭頭為方向、數字為實際計分）；'
                    f'現貨 = 台灣50 + 中型100（市值前 {TOP_N} 檔模擬）中 5／10／20 日均線同時向上減同時向下的檔數'
                    f'（還原價），> +{BREADTH_THRESHOLD} 為 +1、< −{BREADTH_THRESHOLD} 為 −1。',
                    style={'fontSize': '11px', 'color': COLOR['text_muted'], 'margin': '0 0 8px 8px'},
                ),
                html.Div(id='mr-detail-table'),
            ], style={'padding': '8px'}),
        ], style={**CARD_STYLE, 'margin': '12px 0 24px'}),

    ], fluid=True),
])


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output('mr-main-chart', 'figure'),
    Output('mr-detail-table', 'children'),
    Output('mr-kpi-date', 'children'),
    Output('mr-kpi-score', 'children'),
    Output('mr-kpi-signal', 'children'),
    Input('mr-market-dropdown', 'value'),
    Input('mr-period-selector', 'value'),
)
def update_chart(market, period):
    """依 period 截切日期，繪製雙行圖表並產生明細表。"""
    cache = _page_cache.data  # 取快照：整個 callback 只讀這一份，不受背景重算影響
    cached = cache['markets'][market]
    updated = cache['updated']

    # Date filter（以最新交易日往前推算）
    days = _PERIOD_DAYS.get(period, _PERIOD_DAYS[_DEFAULT_PERIOD])
    cutoff = cached['ohlc'].index[-1] - timedelta(days=days)
    ohlc_df = cached['ohlc'].loc[cutoff:]
    comp    = cached['components'].loc[cutoff:]
    scores  = comp['total']

    dates = ohlc_df.index

    # Bar colors
    bar_colors = [
        _SCORE_COLOR['score_long']    if s > 0
        else _SCORE_COLOR['score_short']   if s < -1
        else _SCORE_COLOR['score_neutral']
        for s in scores
    ]

    # Subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.35],
        subplot_titles=('TAIEX 指數', '均線條件式 S4 分數'),
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=dates,
        open=ohlc_df['open'],
        high=ohlc_df['high'],
        low=ohlc_df['low'],
        close=ohlc_df['close'],
        name='TAIEX',
        increasing_line_color=_SCORE_COLOR['candle_up'],
        decreasing_line_color=_SCORE_COLOR['candle_down'],
        increasing_fillcolor=_SCORE_COLOR['candle_up'],
        decreasing_fillcolor=_SCORE_COLOR['candle_down'],
        showlegend=False,
    ), row=1, col=1)

    # Row 2: Score bars（hover 顯示各項分數）
    fig.add_trace(go.Bar(
        x=dates,
        y=scores,
        name='分數',
        marker_color=bar_colors,
        showlegend=False,
        customdata=comp[['ma_score', 'di_score', 'dif_score', 'macd_score', 'kd_score', 'breadth_score']]
                   .assign(breadth_diff=comp['breadth_diff'].map(
                       lambda v: '—' if pd.isna(v) else _signed(v))).to_numpy(),
        hovertemplate=(
            '總分 %{y}<br>均線 %{customdata[0]}｜DMI %{customdata[1]}｜'
            'DIF %{customdata[2]}｜MACD %{customdata[3]}｜KD %{customdata[4]}｜'
            '現貨 %{customdata[5]}（檔數差 %{customdata[6]}）<extra></extra>'
        ),
    ), row=2, col=1)

    # Reference lines (y=0, y=-1)
    for y_val, dash_style, label in [
        (0,  'dash',  '0'),
        (-1, 'dot',   '-1'),
    ]:
        fig.add_hline(
            y=y_val, row=2, col=1,
            line_dash=dash_style,
            line_color=COLOR['grid_zero'],
            line_width=1,
            annotation_text=label,
            annotation_position='left',
            annotation_font_size=10,
            annotation_font_color=COLOR['text_muted'],
        )

    # Layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT, color=COLOR['text_secondary'], size=12),
        margin=dict(l=60, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='white', font_size=12),
        dragmode=False,  # 鎖定圖表：禁止拖曳縮放，避免誤觸
    )

    # 隱藏非交易日：bounds 排除週末，values 只列舉 weekday 假日（颱風假等）
    bdays = pd.date_range(dates.min(), dates.max(), freq='B')
    holidays = bdays.difference(dates)

    _axis_style = dict(
        showgrid=True, gridcolor=COLOR['border'],
        zeroline=False,
        tickfont=dict(size=11, color=COLOR['text_muted']),
        linecolor=COLOR['border'],
        fixedrange=True,  # 鎖定座標軸範圍
    )
    fig.update_xaxes(
        **_axis_style,
        rangebreaks=[
            dict(bounds=['sat', 'mon']),
            dict(values=[d.strftime('%Y-%m-%d') for d in holidays]),
        ],
    )
    fig.update_yaxes(**_axis_style)

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12, color=COLOR['text_muted']), x=0, xanchor='left')

    # ── KPI cards ──────────────────────────────────────────────────────────
    last_score = int(scores.iloc[-1]) if len(scores) > 0 else 0
    emoji, sig_text, _ = _signal_badge(last_score)

    last_date = dates[-1].strftime('%Y-%m-%d') if len(dates) > 0 else '—'
    kpi_date    = kpi_card('訊號日（收盤資料）', last_date,
                           subtitle=f'供下一交易日操作參考｜資料更新：{updated}（{REFRESH_LABEL}）')
    kpi_score   = kpi_card('當前分數', _signed(last_score),
                           subtitle='範圍 -10 ~ +10（均線 + DMI + MACD + KD + 現貨）')
    kpi_signal  = kpi_card('訊號', f'{emoji} {sig_text}',
                           subtitle='> 0 做多 ｜ -1 ≤ 分數 ≤ 0 觀望 ｜ < -1 出場')

    return fig, _detail_table(comp), kpi_date, kpi_score, kpi_signal
