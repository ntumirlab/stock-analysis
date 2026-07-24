"""
台股多空轉折模型 Dashboard
===========================
顯示 TAIEX K 線 + 均線條件式 S4 分數柱狀圖。
預設顯示最近 2 個月，可切換時間範圍。

啟動：python tw_market_regime_dashboard.py
URL : http://localhost:8052
"""

import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv('.env')

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import finlab
from finlab import data
import talib
from talib import abstract

# ── Constants ─────────────────────────────────────────────────────────────────

_TZ = ZoneInfo('Asia/Taipei')

_COLOR = {
    'text_heading':   '#1a202c',
    'text_secondary': '#374151',
    'text_muted':     '#6b7280',
    'accent':         '#1d4ed8',
    'border':         '#e5e7eb',
    'grid_zero':      '#d1d5db',
    'bg_page':        '#f0f2f5',
    'score_long':     '#ef5350',   # score > 0  → 做多（紅）
    'score_short':    '#26a69a',   # score < -1 → 出場（綠）
    'score_neutral':  '#9ca3af',   # -1 ≤ score ≤ 0 → 觀望（灰）
    'candle_up':      '#ef5350',   # K 線漲（紅）— 台灣慣例
    'candle_down':    '#26a69a',   # K 線跌（綠）— 台灣慣例
}

_FONT = 'system-ui, -apple-system, sans-serif'

_PERIOD_DAYS = {
    '1M': 30, '2M': 60, '3M': 90,
    '6M': 180, '1Y': 365, 'All': None,
}
_DEFAULT_PERIOD = '2M'

# 市場設定（目前只有 TAIEX，保留結構方便擴充）
_MARKET_CONFIG = {
    'TAIEX': {'label': 'TAIEX（台股）', 'dmi': (35, 21, 18)},
    # 未來可新增：'QQQ': {'label': 'QQQ（NASDAQ-100）', 'dmi': (41, 26, 21)}
}

# ── Score computation ──────────────────────────────────────────────────────────
# 與 toy_example/tw_market_regime_s4_lo_ls.py 的 compute_score 相同

def compute_score(ohlc: pd.DataFrame, dmi_hi: int, dmi_mid: int, dmi_lo: int) -> pd.Series:
    """均線條件式 S4 評分（-9 ~ +9）。"""
    close = ohlc['close']
    high  = ohlc['high']
    low   = ohlc['low']

    plus_di  = abstract.Function('PLUS_DI')(ohlc, timeperiod=14)
    minus_di = abstract.Function('MINUS_DI')(ohlc, timeperiod=14)

    di_score = (
        (plus_di > dmi_hi).astype(int)                               *  2
      + ((plus_di >= dmi_mid) & (plus_di <= dmi_hi)).astype(int)    *  1
      + ((plus_di >= dmi_lo)  & (plus_di <  dmi_mid)).astype(int)   * -1
      + (plus_di < dmi_lo).astype(int)                              * -2
      + (minus_di > dmi_hi).astype(int)                             * -2
      + ((minus_di >= dmi_mid) & (minus_di <= dmi_hi)).astype(int)  * -1
      + ((minus_di >= dmi_lo)  & (minus_di <  dmi_mid)).astype(int) *  1
      + (minus_di < dmi_lo).astype(int)                             *  2
    )

    dif, macd, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    dif_rising   = dif  > dif.shift(1)
    dif_falling  = dif  < dif.shift(1)
    macd_rising  = macd > macd.shift(1)
    macd_falling = macd < macd.shift(1)

    low9  = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv   = ((close - low9) / (high9 - low9).replace(0, float('nan')) * 100).fillna(50)
    K = rsv.ewm(com=2, adjust=False).mean()
    D = K.ewm(com=2, adjust=False).mean()
    kd_sync_up   = (K > K.shift(1)) & (D > D.shift(1))
    kd_sync_down = (K < K.shift(1)) & (D < D.shift(1))

    ma5  = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    a5, a10, a20 = close > ma5, close > ma10, close > ma20

    ma_score = (
        ( a5 &  a10 &  a20).astype(int) *  2
      + ( a5 &  a10 & ~a20).astype(int) *  1
      + (~a5 &  a10 &  a20).astype(int) *  1
      + (~a5 & ~a10 &  a20).astype(int) * -1
      + (~a5 & ~a10 & ~a20).astype(int) * -2
    )

    sub  = ma_score + di_score
    lt5  = sub < 5
    gtn5 = sub > -5

    return sub + (
        - (lt5   & dif_falling).astype(int)
        - (lt5   & macd_falling).astype(int)
        - (lt5   & kd_sync_down).astype(int)
        + (gtn5  & dif_rising).astype(int)
        + (gtn5  & macd_rising).astype(int)
        + (gtn5  & kd_sync_up).astype(int)
    )


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_taiex() -> pd.DataFrame:
    """載入 TAIEX OHLC（全歷史）。"""
    return pd.DataFrame({
        'open':  data.get('taiex_total_index:開盤指數')['TAIEX'],
        'high':  data.get('taiex_total_index:最高指數')['TAIEX'],
        'low':   data.get('taiex_total_index:最低指數')['TAIEX'],
        'close': data.get('taiex_total_index:收盤指數')['TAIEX'],
    }).dropna()


def _build_store(market: str) -> dict:
    """拉取資料並計算分數，打包為可序列化的 dict。"""
    cfg = _MARKET_CONFIG[market]
    dmi_hi, dmi_mid, dmi_lo = cfg['dmi']

    if market == 'TAIEX':
        ohlc = _load_taiex()
    else:
        raise NotImplementedError(f'Market {market} not implemented yet.')

    score = compute_score(ohlc, dmi_hi, dmi_mid, dmi_lo)

    return {
        'ohlc': ohlc.reset_index().rename(columns={'index': 'date'}).assign(
            date=lambda df: df['date'].astype(str)
        ).to_dict('records'),
        'score': score.reset_index().rename(columns={'index': 'date', 0: 'score'}).assign(
            date=lambda df: df['date'].astype(str),
            score=lambda df: df['score'].fillna(0).astype(int),
        ).to_dict('records'),
        'updated': datetime.now(_TZ).strftime('%Y-%m-%d %H:%M'),
    }


# ── FinLab login（每個 worker process 只執行一次）─────────────────────────────

_FINLAB_TOKEN = os.environ.get('FINLAB_API_TOKEN', '')
if not _FINLAB_TOKEN:
    raise RuntimeError('FINLAB_API_TOKEN is not set. Check your .env file.')
finlab.login(_FINLAB_TOKEN)

# ── Dash app ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='台股多空轉折模型',
)
app.config.suppress_callback_exceptions = True
server = app.server  # gunicorn 進入點

# ── Layout helpers ─────────────────────────────────────────────────────────────

def _signal_badge(score_val: int) -> tuple[str, str, str]:
    """回傳 (emoji, 文字, badge color)。"""
    if score_val > 0:
        return '🟢', '做多', 'primary'
    if score_val < -1:
        return '🔴', '出場', 'danger'
    return '⚪', '觀望', 'secondary'


def _kpi_card(label: str, value, subtitle: str = '') -> dbc.Card:
    children = [
        html.Div(label, style={
            'fontSize': '12px', 'color': _COLOR['text_muted'],
            'marginBottom': '4px', 'fontWeight': '500',
        }),
        html.Div(str(value), style={
            'fontSize': '22px', 'fontWeight': '700',
            'color': _COLOR['text_heading'],
        }),
    ]
    if subtitle:
        children.append(html.Div(subtitle, style={
            'fontSize': '11px', 'color': _COLOR['text_muted'],
            'marginTop': '6px', 'lineHeight': '1.4',
        }))
    return dbc.Card(
        dbc.CardBody(children, style={'padding': '14px 18px'}),
        style={
            'border': f"1px solid {_COLOR['border']}",
            'borderRadius': '8px',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.06)',
            'backgroundColor': 'white',
        },
    )


# ── App layout ─────────────────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Store(id='market-data-store'),

    # ── Top bar ──────────────────────────────────────────────────────────────
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.H5('台股多空轉折模型', style={
                        'margin': 0, 'fontWeight': '700',
                        'color': _COLOR['text_heading'],
                        'fontSize': 'clamp(15px, 3vw, 18px)',
                    }),
                    width='auto', className='d-flex align-items-center',
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='market-dropdown',
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
                        id='period-selector',
                        options=[{'label': k, 'value': k} for k in _PERIOD_DAYS],
                        value=_DEFAULT_PERIOD,
                        inputClassName='btn-check',
                        labelClassName='btn btn-outline-secondary btn-sm',
                        labelCheckedClassName='active',
                        inline=True,
                    ),
                    width='auto', className='d-flex align-items-center',
                ),
                dbc.Col(
                    dbc.Button(
                        '↻ 重新整理', id='refresh-btn',
                        color='light', size='sm',
                        style={'border': f"1px solid {_COLOR['border']}", 'fontSize': '13px'},
                    ),
                    width='auto', className='d-flex align-items-center',
                ),
                dbc.Col(
                    html.Div(id='loading-indicator', style={
                        'fontSize': '12px', 'color': _COLOR['text_muted'],
                    }),
                    className='d-flex align-items-center ms-auto',
                ),
            ], className='g-2 align-items-center'),
        ], fluid=True),
    ], style={
        'backgroundColor': 'white',
        'borderBottom': f"1px solid {_COLOR['border']}",
        'padding': '12px 0',
        'position': 'sticky', 'top': 0, 'zIndex': 100,
    }),

    # ── Main content ──────────────────────────────────────────────────────────
    dbc.Container([
        # Chart
        dbc.Card([
            dbc.CardBody(
                dcc.Graph(
                    id='main-chart',
                    config={'displayModeBar': True, 'scrollZoom': True},
                    style={'height': '72vh'},
                ),
                style={'padding': '8px'},
            ),
        ], style={
            'border': f"1px solid {_COLOR['border']}",
            'borderRadius': '8px',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.06)',
            'marginTop': '16px',
        }),

        # Signal status cards
        dbc.Row([
            dbc.Col(html.Div(id='kpi-score'), width=4),
            dbc.Col(html.Div(id='kpi-signal'), width=4),
            dbc.Col(html.Div(id='kpi-updated'), width=4),
        ], className='g-2 mt-2 mb-3'),

    ], fluid=True),
], style={'backgroundColor': _COLOR['bg_page'], 'minHeight': '100vh'})


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output('market-data-store', 'data'),
    Output('loading-indicator', 'children'),
    Input('market-dropdown', 'value'),
    Input('refresh-btn', 'n_clicks'),
)
def load_data(market, _n_clicks):
    """載入 FinLab 資料並計算分數，存入 dcc.Store。"""
    try:
        store = _build_store(market)
        updated = store['updated']
        return json.dumps(store), f'更新：{updated}'
    except Exception as e:
        return None, f'載入失敗：{e}'


@app.callback(
    Output('main-chart', 'figure'),
    Output('kpi-score', 'children'),
    Output('kpi-signal', 'children'),
    Output('kpi-updated', 'children'),
    Input('market-data-store', 'data'),
    Input('period-selector', 'value'),
)
def update_chart(store_json, period):
    """依 period 截切日期並繪製雙行圖表。"""
    empty_fig = go.Figure().update_layout(
        plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
        annotations=[{'text': '載入中…', 'showarrow': False,
                      'font': {'size': 16, 'color': _COLOR['text_muted']}}],
    )
    blank = _kpi_card('—', '—')
    if not store_json:
        return empty_fig, blank, blank, blank

    store   = json.loads(store_json)
    ohlc_df = pd.DataFrame(store['ohlc']).set_index('date')
    ohlc_df.index = pd.to_datetime(ohlc_df.index)
    score_df = pd.DataFrame(store['score']).set_index('date')
    score_df.index = pd.to_datetime(score_df.index)
    updated = store.get('updated', '—')

    # Date filter
    days = _PERIOD_DAYS.get(period)
    if days:
        cutoff = ohlc_df.index[-1] - timedelta(days=days)
        ohlc_df  = ohlc_df[ohlc_df.index >= cutoff]
        score_df = score_df[score_df.index >= cutoff]

    dates  = ohlc_df.index
    scores = score_df['score']

    # Bar colors
    bar_colors = [
        _COLOR['score_long']    if s > 0
        else _COLOR['score_short']   if s < -1
        else _COLOR['score_neutral']
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
        increasing_line_color=_COLOR['candle_up'],
        decreasing_line_color=_COLOR['candle_down'],
        increasing_fillcolor=_COLOR['candle_up'],
        decreasing_fillcolor=_COLOR['candle_down'],
        showlegend=False,
    ), row=1, col=1)

    # Row 2: Score bars
    fig.add_trace(go.Bar(
        x=dates,
        y=scores,
        name='分數',
        marker_color=bar_colors,
        showlegend=False,
    ), row=2, col=1)

    # Reference lines (y=0, y=-1)
    for y_val, dash_style, label in [
        (0,  'dash',  '0'),
        (-1, 'dot',   '-1'),
    ]:
        fig.add_hline(
            y=y_val, row=2, col=1,
            line_dash=dash_style,
            line_color=_COLOR['grid_zero'],
            line_width=1,
            annotation_text=label,
            annotation_position='left',
            annotation_font_size=10,
            annotation_font_color=_COLOR['text_muted'],
        )

    # Layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family=_FONT, color=_COLOR['text_secondary'], size=12),
        margin=dict(l=60, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='white', font_size=12),
    )

    # 隱藏非交易日：bounds 排除週末，values 只列舉 weekday 假日（颱風假等）
    bdays = pd.date_range(dates.min(), dates.max(), freq='B')
    holidays = bdays.difference(dates)

    _axis_style = dict(
        showgrid=True, gridcolor=_COLOR['border'],
        zeroline=False,
        tickfont=dict(size=11, color=_COLOR['text_muted']),
        linecolor=_COLOR['border'],
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
        ann.update(font=dict(size=12, color=_COLOR['text_muted']), x=0, xanchor='left')

    # ── KPI cards ──────────────────────────────────────────────────────────
    last_score = int(scores.iloc[-1]) if len(scores) > 0 else 0
    emoji, sig_text, _ = _signal_badge(last_score)

    score_sign = f'+{last_score}' if last_score > 0 else str(last_score)
    kpi_score   = _kpi_card('當前分數', score_sign,
                            subtitle='範圍 -9 ~ +9（均線 + DMI + MACD + KD）')
    kpi_signal  = _kpi_card('訊號', f'{emoji} {sig_text}',
                            subtitle='> 0 做多 ｜ -1 ≤ 分數 ≤ 0 觀望 ｜ < -1 出場')
    kpi_updated = _kpi_card('資料更新', updated)

    return fig, kpi_score, kpi_signal, kpi_updated


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8052)
