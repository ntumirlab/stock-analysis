"""
台股多空轉折模型頁面
====================
顯示 TAIEX K 線 + 均線條件式 S4 分數柱狀圖，下方以明細表列出每日各條件的數值與分數
（含台指選擇權當月／遠月未平倉 P/C 與選擇權分）。預設顯示最近 2 個月，可切換時間範圍（最長 1 年）。

評分邏輯見 alan_dashboard/market_regime_model.py。
資料與分數於啟動時計算一次並快取於伺服器記憶體（見 alan_dashboard/cache.py），
每次請求只回傳選定期間的圖表與明細。
"""

from datetime import timedelta

import dash
from dash import dcc, html, dash_table, Input, Output, callback
from dash.dash_table.Format import Format, Scheme, Sign, Symbol
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from finlab import data

from alan_dashboard.cache import PageCache, REFRESH_LABEL
from alan_dashboard.market_regime_model import (
    BREADTH_THRESHOLD, PCR_FAR_SHORT, PCR_NEAR_LONG, PCR_NEAR_SHORT, PCR_NEAR_STRONG, TOP_N,
    compute_components, listed_common_stocks,
    ma_direction_breadth, top_n_membership, txo_put_call_ratio,
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
# 現貨檔數差只需要顯示範圍（最長 1Y）加 MA20 暖身；個股還原價先截到這段再算 rolling，省時間與記憶體。
# 成分股模擬仍用全歷史市值（緩衝規則有路徑依賴，截短會改變名單）。
_BREADTH_HISTORY_DAYS = max(_PERIOD_DAYS.values()) + 60

# 市場設定（目前只有 TAIEX，保留結構方便擴充）
# breadth：現貨分的成分股來源；None 表示該市場不計現貨分
# options：選擇權分的 P/C 資料來源；None 表示該市場不計選擇權分
_MARKET_CONFIG = {
    'TAIEX': {'label': 'TAIEX（台股）', 'dmi': (35, 21, 18), 'breadth': 'tw_top150', 'options': 'txo'},
    # 未來可新增：'QQQ': {'label': 'QQQ（NASDAQ-100）', 'dmi': (41, 26, 21), 'breadth': None, 'options': None}
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
        # 台指選擇權逐契約未沖銷部位（2025-09-25 起），算當月／遠月 P/C；只留 TXO，其他商品不用
        'txo_liquidity': data.get('tw_taifex_option_liquidity').query("symbol == 'TXO'"),
    }


def _tw_top150_breadth(inputs: dict) -> pd.Series:
    """台灣50 + 台灣中型100（以市值前 150、季度審核與緩衝規則模擬）三線同向檔數差（還原價）。"""
    universe = listed_common_stocks(inputs['categories'])
    membership = top_n_membership(inputs['market_value'], universe, n=TOP_N)
    adj_close = inputs['adj_close']
    recent = adj_close.loc[adj_close.index[-1] - timedelta(days=_BREADTH_HISTORY_DAYS):]
    return ma_direction_breadth(recent, membership)


def _build(inputs: dict) -> dict:
    """對每個市場計算各條件分數；快取內容為 {market: {'ohlc': DataFrame, 'components': DataFrame}}。"""
    breadth = {'tw_top150': _tw_top150_breadth(inputs)}
    options = {'txo': txo_put_call_ratio(inputs['txo_liquidity'])}
    markets = {}
    for market, cfg in _MARKET_CONFIG.items():
        ohlc = inputs[market]
        # P/C 資料起始前或尚未更新的日期為 NaN：計 0 分、明細表顯示「—」
        pcr = options[cfg['options']] if cfg['options'] else None
        components = compute_components(ohlc, *cfg['dmi'], breadth_diff=breadth.get(cfg['breadth']), pcr=pcr)
        markets[market] = {'ohlc': ohlc, 'components': components}
    return {'markets': markets}


_page_cache = PageCache('market_regime', _load_inputs, _build)
_page_cache.refresh()  # 啟動時計算一次（失敗即讓 worker 啟動失敗，與先前行為一致）


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _signal_badge(score_val: int) -> tuple[str, str, str]:
    """回傳 (emoji, 文字, badge color)。燈號依台灣慣例：紅 = 做多、綠 = 出場，與 K 線、分數柱同色。"""
    if score_val > 0:
        return '🔴', '做多', 'danger'
    if score_val < -1:
        return '🟢', '出場', 'success'
    return '⚪', '觀望', 'secondary'


def _signed(v) -> str:
    v = int(v)
    return f'+{v}' if v > 0 else str(v)


_FMT_SIGNED = Format(sign=Sign.positive, precision=0, scheme=Scheme.fixed, nully='—')
_FMT_INDEX  = Format(group=True, precision=0, scheme=Scheme.fixed)
_FMT_PCT    = Format(precision=1, scheme=Scheme.fixed, nully='—').symbol(Symbol.yes).symbol_suffix('%')
# (欄位 id, 顯示名稱, 型別)：numeric 欄以數值排序、由 Format 負責顯示；text 欄為字串
_DETAIL_COLUMNS = [
    ('date', '日期', 'text'), ('close', '收盤指數', 'numeric'),
    ('ma_above', '站上均線', 'text'), ('kd', 'KD', 'text'), ('ma_score', '均線分', 'numeric'),
    ('di_vals', '+DI / −DI', 'text'), ('di_score', 'DMI分', 'numeric'),
    ('subtotal', '小計', 'numeric'),
    ('dif', 'DIF', 'text'), ('macd', 'MACD', 'text'),
    ('breadth_diff', '現貨檔數差', 'numeric'), ('breadth_score', '現貨分', 'numeric'),
    ('pcr_near', '當月P/C', 'numeric'), ('pcr_far', '遠月P/C', 'numeric'), ('options_score', '選擇權分', 'numeric'),
    ('total', '總分', 'numeric'), ('signal', '訊號', 'text'),
]
_PCT_COLUMNS = ['pcr_near', 'pcr_far']
_SIGNED_NUMERIC = ['ma_score', 'di_score', 'subtotal', 'breadth_diff', 'breadth_score', 'options_score', 'total']
_ARROW_COLUMNS = ['dif', 'macd']  # 內容如「↑ +1」「↓ -1」「↑ 0」，依 +/- 著色


_DIR_SYMBOL = {1: '↑', -1: '↓', 0: '—'}


def _arrow(direction: int, score: int) -> str:
    """動能欄：方向箭頭 + 實際計分（小計未達門檻時方向成立但不計分，顯示 0）。"""
    return f'{_DIR_SYMBOL[int(direction)]} {_signed(score)}'


def _kd_cell(k_dir: int, d_dir: int) -> str:
    """KD 欄：K、D 同向（含同時持平）時只顯示一個符號（均線分 ±1 的確認條件）；不同向時分別列出 K、D 方向。"""
    if k_dir == d_dir:
        return _DIR_SYMBOL[int(k_dir)]
    return f'K{_DIR_SYMBOL[int(k_dir)]} D{_DIR_SYMBOL[int(d_dir)]}'


def _pct_or_none(v) -> float | None:
    return None if pd.isna(v) else float(v)


def _detail_rows(comp: pd.DataFrame) -> list[dict]:
    """明細表資料（最新日在前）；數值欄保留數值，顯示格式交給 DataTable 的 Format。"""
    rows = []
    for d, r in comp[::-1].iterrows():
        above = [str(w) for w, col in ((5, 'above_ma5'), (10, 'above_ma10'), (20, 'above_ma20')) if r[col]]
        emoji, sig_text, _ = _signal_badge(int(r['total']))
        rows.append({
            'date': d.strftime('%Y-%m-%d'),
            'close': float(r['close']),
            'ma_above': '·'.join(above) if above else '—',
            'kd': _kd_cell(r['k_dir'], r['d_dir']),
            'ma_score': int(r['ma_score']),
            'di_vals': f"{r['plus_di']:.1f} / {r['minus_di']:.1f}",
            'di_score': int(r['di_score']),
            'subtotal': int(r['subtotal']),
            'dif': _arrow(r['dif_dir'], r['dif_score']),
            'macd': _arrow(r['macd_dir'], r['macd_score']),
            'breadth_diff': None if pd.isna(r['breadth_diff']) else int(r['breadth_diff']),
            'breadth_score': int(r['breadth_score']),
            'total': int(r['total']),
            'pcr_near': _pct_or_none(r['pcr_near']),
            'pcr_far': _pct_or_none(r['pcr_far']),
            'options_score': int(r['options_score']),
            'signal': f'{emoji} {sig_text}',
        })
    return rows


def _detail_table(comp: pd.DataFrame) -> dash_table.DataTable:
    colored = []
    for col in _SIGNED_NUMERIC:
        colored += [
            {'if': {'filter_query': f'{{{col}}} > 0', 'column_id': col},
             'color': _SCORE_COLOR['score_long'], 'fontWeight': '600'},
            {'if': {'filter_query': f'{{{col}}} < 0', 'column_id': col},
             'color': _SCORE_COLOR['score_short'], 'fontWeight': '600'},
        ]
    for col in _ARROW_COLUMNS:
        colored += [
            {'if': {'filter_query': f'{{{col}}} contains "+"', 'column_id': col},
             'color': _SCORE_COLOR['score_long'], 'fontWeight': '600'},
            {'if': {'filter_query': f'{{{col}}} contains "-"', 'column_id': col},
             'color': _SCORE_COLOR['score_short'], 'fontWeight': '600'},
        ]
    # KD 欄：K、D 同步上升／下降時著色，不同向（K↑ D↓）維持灰字
    colored += [
        {'if': {'filter_query': '{kd} = "↑"', 'column_id': 'kd'},
         'color': _SCORE_COLOR['score_long'], 'fontWeight': '600'},
        {'if': {'filter_query': '{kd} = "↓"', 'column_id': 'kd'},
         'color': _SCORE_COLOR['score_short'], 'fontWeight': '600'},
    ]
    columns = []
    for cid, name, ctype in _DETAIL_COLUMNS:
        col = {'name': name, 'id': cid, 'type': ctype}
        if cid in _SIGNED_NUMERIC:
            col['format'] = _FMT_SIGNED
        elif cid == 'close':
            col['format'] = _FMT_INDEX
        elif cid in _PCT_COLUMNS:
            col['format'] = _FMT_PCT
        columns.append(col)
    return dash_table.DataTable(
        data=_detail_rows(comp),
        columns=columns,
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
                    '均線分：三條均線都站上 +2、都未站上 −2；站上 MA5·10 未站上 MA20、或未站上 MA5 但站上 MA10·20，'
                    '要 K、D 同步上升才 +1；僅站上 MA20 要 K、D 同步下降才 −1（KD 欄為 K、D 方向）。'
                    '均線分 + DMI分 = 小計；小計 < 5 時 DIF／MACD 下彎各 −1，小計 > −5 時上彎各 +1'
                    '（箭頭為方向、數字為實際計分）；'
                    f'現貨 = 台灣50 + 中型100（市值前 {TOP_N} 檔模擬）中 5／10／20 日均線同時向上減同時向下的檔數'
                    f'（還原價），> +{BREADTH_THRESHOLD} 為 +1、< −{BREADTH_THRESHOLD} 為 −1。'
                    '選擇權 = 台指選擇權未平倉量 P/C（賣權 ÷ 買權；當月 = 最近未到期月契約、'
                    f'遠月 = 其餘月契約合計，週契約不計）：當月 > {PCR_NEAR_LONG}% 且 > 遠月 +1、< {PCR_NEAR_SHORT}% −1、'
                    f'> {PCR_NEAR_STRONG}% 再 +1；遠月 > {PCR_FAR_SHORT}% 且 > 當月 −1。'
                    '+DI／−DI 與 P/C 以未四捨五入的原值計分，顯示值剛好等於門檻時以原值為準。',
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
        customdata=comp[['ma_score', 'di_score', 'dif_score', 'macd_score', 'breadth_score', 'options_score']]
                   .assign(breadth_diff=comp['breadth_diff'].map(lambda v: '—' if pd.isna(v) else _signed(v)),
                           kd=[_kd_cell(k, d) for k, d in zip(comp['k_dir'], comp['d_dir'])]).to_numpy(),
        hovertemplate=(
            '總分 %{y}<br>均線 %{customdata[0]}（KD %{customdata[7]}）｜DMI %{customdata[1]}｜'
            'DIF %{customdata[2]}｜MACD %{customdata[3]}｜'
            '現貨 %{customdata[4]}（檔數差 %{customdata[6]}）｜選擇權 %{customdata[5]}<extra></extra>'
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
                           subtitle='範圍 -11 ~ +11（均線（含 KD 確認）+ DMI + MACD + 現貨 + 選擇權）')
    kpi_signal  = kpi_card('訊號', f'{emoji} {sig_text}',
                           subtitle='> 0 做多 ｜ -1 ≤ 分數 ≤ 0 觀望 ｜ < -1 出場')

    return fig, _detail_table(comp), kpi_date, kpi_score, kpi_signal
