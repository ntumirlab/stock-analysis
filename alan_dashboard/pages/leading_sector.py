"""
領先潛力族群頁面
================
依六項條件篩選個股，以證交所產業分類呈現「族群家數分布」，
並提供 date picker 瀏覽歷史分布。

篩選條件（1、4、5、6 為 AND；2 與 3 之間為 OR）：
    1. 收盤價 >= 480 天新高 * 90%
    2. 營益率增 12%  + 買超排行前 40 檔
    3. 營益率增 0.1% + 買超排行前 20 檔
    4. 60 日均線乖離 < 28%
    5. 120 日均線乖離 < 45%
    6. 收盤價 <= 近 15 天最低價 * 132%

價格一律使用還原價格（etl:adj_close / etl:adj_low），
買超排行與營益率增邏輯與 strategy_class.alan_tw_strategy_base 一致：
    - 買超排行：外資 / 投信 / 自營商 / 主力（top15 分點）四者擇一（OR 聯集），
      各以「買賣超股數 ÷ 發行股數」之 1 日 / 2 日累計 / 3 日累計
      全市場排名進前 N 檔（主力另須達絕對比例門檻）
    - 營益率增：本季營益率 > 前一季營益率 * 門檻
"""

from datetime import datetime

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from finlab import data

from alan_dashboard.theme import TZ, COLOR, FONT, CARD_STYLE, kpi_card

dash.register_page(__name__, path='/leading-sector', name='領先潛力族群',
                   title='領先潛力族群', order=1)

# ── Constants ─────────────────────────────────────────────────────────────────

# 類股色盤（依類股名稱固定指派，超過 8 個類股時循環使用；
# 類股身分以座標軸標籤為準，顏色僅輔助區隔）
_PALETTE = [
    '#2a78d6',  # blue
    '#eb6834',  # orange
    '#1baf7a',  # aqua
    '#eda100',  # yellow
    '#e87ba4',  # magenta
    '#008300',  # green
    '#4a3aa7',  # violet
    '#e34948',  # red
]

# 條件參數
_NEW_HIGH_DAYS = 480
_NEW_HIGH_PCT = 0.90
_BIAS60_MAX = 0.28
_BIAS120_MAX = 0.45
_LOW_RATIO_DAYS = 15
_LOW_RATIO_MAX = 1.32
_COND2 = {'op_growth': 1.12,  'top_n': 40, 'label': '營益率增12%+買超前40'}
_COND3 = {'op_growth': 1.001, 'top_n': 20, 'label': '營益率增0.1%+買超前20'}

# 主力（top15 分點）買超比例絕對門檻，與 Alan 策略一致
_MAIN_FORCE_MIN = {1: 0.0008, 2: 0.0015, 3: 0.0025}

# 資料起算日：需保留 480 交易日 rolling 視窗的暖身期
_COMPUTE_START = '2020-06-01'
# 儀表板可瀏覽的歷史起點
_HISTORY_START = '2023-01-01'

_CHIP_SOURCES = ('外資', '投信', '自營商', '主力')

_CONDITION_NOTE = (
    '條件：收盤 ≥ 480日高×90%｜（營益率增12%+買超前40 或 營益率增0.1%+買超前20）'
    '｜60日乖離 < 28%｜120日乖離 < 45%｜收盤 ≤ 15日低×132%（還原價）'
)


# ── Screen computation ─────────────────────────────────────────────────────────

def _chip_rank_masks(top_ns):
    """計算各買超來源進前 N 檔的條件。

    Returns:
        dict: {top_n: {source: bool DataFrame}}
    """
    with data.universe(market='TSE_OTC'):
        foreign = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
        trust = data.get('institutional_investors_trading_summary:投信買賣超股數')
        dealer = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
        shares_outstanding = data.get('internal_equity_changes:發行股數')
        # finlab 2.x 將分點資料載為 nullable Int64，rank() 會 TypeError，統一轉 float64
        top15_buy = data.get('etl:broker_transactions:top15_buy').astype('float64')
        top15_sell = data.get('etl:broker_transactions:top15_sell').astype('float64')

    main_force_shares = (top15_buy - top15_sell) * 1000

    ratios = {
        '外資': foreign / shares_outstanding,
        '投信': trust / shares_outstanding,
        '自營商': dealer / shares_outstanding,
        '主力': main_force_shares / shares_outstanding,
    }

    masks = {n: {} for n in top_ns}
    for source, ratio in ratios.items():
        ratio = ratio.loc[_COMPUTE_START:]
        windows = {1: ratio, 2: ratio.rolling(2).sum(), 3: ratio.rolling(3).sum()}
        ranks = {d: w.rank(axis=1, ascending=False) for d, w in windows.items()}

        for top_n in top_ns:
            conds = []
            for d in (1, 2, 3):
                cond = ranks[d] <= top_n
                if source == '主力':
                    cond = cond & (windows[d] > _MAIN_FORCE_MIN[d])
                conds.append(cond)
            masks[top_n][source] = conds[0] | conds[1] | conds[2]

    return masks


def compute_screen() -> dict:
    """計算六項條件的篩選結果，回傳儀表板所需的快取資料。"""
    with data.universe(market='TSE_OTC'):
        close = data.get('price:收盤價')
        adj_close = data.get('etl:adj_close')
        adj_low = data.get('etl:adj_low')
        operating_margin = data.get('fundamental_features:營業利益率')

    # 產業分類與名稱：security_categories 保留已下市股票的列（company_basic_info
    # 為當前快照、下市即消失），且現存股票的分類與簡稱經比對與 company_basic_info 完全一致
    sec_cat = data.get('security_categories').set_index('stock_id')

    def _valid(series):
        # 過濾 NaN 與字串 'nan'（finlab 部分欄位以文字儲存缺值）
        return {k: v for k, v in series.items() if pd.notna(v) and str(v) != 'nan'}

    industry_map = _valid(sec_cat['category'])
    name_map = _valid(sec_cat['name'])

    close = close.loc[_COMPUTE_START:]
    adj_close = adj_close.loc[_COMPUTE_START:]
    adj_low = adj_low.loc[_COMPUTE_START:]

    # 條件 1：收盤價 >= 480 天新高 * 90%（還原價）
    high_480 = adj_close.rolling(_NEW_HIGH_DAYS).max()
    high_ratio = adj_close / high_480
    cond1 = high_ratio >= _NEW_HIGH_PCT

    # 條件 2 / 3：營益率增 + 買超排行（營益率為季頻，與日頻條件以 FinlabDataFrame 自動對齊）
    fund12 = operating_margin > (operating_margin.shift(1) * _COND2['op_growth'])
    fund001 = operating_margin > (operating_margin.shift(1) * _COND3['op_growth'])

    chip_masks = _chip_rank_masks(top_ns=(_COND2['top_n'], _COND3['top_n']))
    src40 = chip_masks[_COND2['top_n']]
    src20 = chip_masks[_COND3['top_n']]
    chip40 = src40['外資'] | src40['投信'] | src40['自營商'] | src40['主力']
    chip20 = src20['外資'] | src20['投信'] | src20['自營商'] | src20['主力']

    cond2 = fund12 & chip40
    cond3 = fund001 & chip20

    # 條件 4 / 5：均線乖離上限（還原價）
    ma60 = adj_close.rolling(60).mean()
    ma120 = adj_close.rolling(120).mean()
    bias60 = (adj_close - ma60) / ma60
    bias120 = (adj_close - ma120) / ma120
    cond4 = bias60 < _BIAS60_MAX
    cond5 = bias120 < _BIAS120_MAX

    # 條件 6：收盤 ÷ 近 15 日最低價 <= 1.32（還原價）
    low_ratio = adj_close / adj_low.rolling(_LOW_RATIO_DAYS).min()
    cond6 = low_ratio <= _LOW_RATIO_MAX

    signal = cond1 & (cond2 | cond3) & cond4 & cond5 & cond6
    # 發行股數為事件型資料（日期不限交易日），條件對齊時索引會混入非交易日，
    # 且訊號值被 ffill 帶入而價格欄為 NaN；一律鎖回實際交易日（adj_close 的索引）
    signal = signal.reindex(adj_close.index.intersection(signal.index))
    signal = signal.fillna(False).astype(bool).loc[_HISTORY_START:]

    def _align(df, fill=None):
        if fill is None:
            return df.reindex(index=signal.index, columns=signal.columns)
        return df.reindex(index=signal.index, columns=signal.columns, fill_value=fill)

    return {
        'signal': signal,
        'dates': [d.strftime('%Y-%m-%d') for d in signal.index],
        'cond2': _align(cond2, False),
        'cond3': _align(cond3, False),
        'src40': {s: _align(m, False) for s, m in src40.items()},
        'src20': {s: _align(m, False) for s, m in src20.items()},
        'detail': {
            'close': _align(close),
            'high_ratio': _align(high_ratio),
            'bias60': _align(bias60),
            'bias120': _align(bias120),
            'low_ratio': _align(low_ratio),
        },
        'industry': industry_map,
        'names': name_map,
        'updated': datetime.now(TZ).strftime('%Y-%m-%d %H:%M'),
    }


# 首次計算（每個 worker process 只執行一次；FinLab 登入由 alan_dashboards.py 負責）
_CACHE = compute_screen()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _snap_date(date_str: str) -> str | None:
    """把任意日期貼齊到最近一個（<= 該日）的交易日；早於歷史起點回傳 None。"""
    dates = _CACHE['dates']
    candidates = [d for d in dates if d <= date_str]
    return candidates[-1] if candidates else None


def _category_color(category: str) -> str:
    """類股顏色：依全部類股名稱排序後固定指派，跨日期一致。"""
    all_cats = sorted(set(_CACHE['industry'].values()) | {'未分類'})
    return _PALETTE[all_cats.index(category) % len(_PALETTE)]


def _stock_rows(date_str: str) -> list[dict]:
    """整理指定交易日入選個股的明細（依類股家數多寡排序）。"""
    signal = _CACHE['signal']
    row = signal.loc[date_str]
    stock_ids = [sid for sid, v in row.items() if bool(v)]

    rows = []
    for sid in stock_ids:
        category = _CACHE['industry'].get(sid) or '未分類'
        matched = []
        sources = set()
        if bool(_CACHE['cond2'].loc[date_str, sid]):
            matched.append(_COND2['label'])
            sources |= {s for s in _CHIP_SOURCES if bool(_CACHE['src40'][s].loc[date_str, sid])}
        if bool(_CACHE['cond3'].loc[date_str, sid]):
            matched.append(_COND3['label'])
            sources |= {s for s in _CHIP_SOURCES if bool(_CACHE['src20'][s].loc[date_str, sid])}

        detail = _CACHE['detail']
        rows.append({
            'stock_id': sid,
            'name': _CACHE['names'].get(sid, sid),
            'category': category,
            'close': detail['close'].loc[date_str, sid],
            'high_ratio': detail['high_ratio'].loc[date_str, sid],
            'bias60': detail['bias60'].loc[date_str, sid],
            'bias120': detail['bias120'].loc[date_str, sid],
            'low_ratio': detail['low_ratio'].loc[date_str, sid],
            'matched': '、'.join(matched),
            'sources': '、'.join(s for s in _CHIP_SOURCES if s in sources),
        })

    # 類股依家數排序（多→少），同類股內依代號排序
    counts = pd.Series([r['category'] for r in rows]).value_counts()
    rows.sort(key=lambda r: (-counts[r['category']], r['category'], r['stock_id']))
    return rows


def _build_figure(date_str: str, rows: list[dict]) -> go.Figure:
    """族群家數分布圖：x 為類股（依家數排序），每檔個股一個方塊堆疊。"""
    fig = go.Figure()

    if not rows:
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', height=420,
            annotations=[{'text': f'{date_str} 無符合條件的個股', 'showarrow': False,
                          'font': {'size': 16, 'color': COLOR['text_muted']}}],
            xaxis={'visible': False}, yaxis={'visible': False},
        )
        return fig

    counts = pd.Series([r['category'] for r in rows]).value_counts()
    categories = list(counts.index)  # 已依家數排序
    tick_labels = {c: f'{c}<br>({counts[c]} 檔)' for c in categories}

    for r in rows:
        cat = r['category']
        fig.add_trace(go.Bar(
            x=[tick_labels[cat]],
            y=[1],
            text=f"{r['name']}({r['stock_id']})",
            textposition='inside',
            insidetextanchor='middle',
            textfont={'color': 'white', 'size': 13, 'family': FONT},
            marker={'color': _category_color(cat),
                    'line': {'color': 'white', 'width': 2}},
            customdata=[[r['name'], r['stock_id'], f"{r['close']:.2f}",
                         f"{r['bias60']:+.1%}", f"{r['bias120']:+.1%}",
                         f"{r['high_ratio']:.1%}", f"{r['low_ratio']:.2f}",
                         r['matched'], r['sources']]],
            hovertemplate=(
                '<b>%{customdata[0]}（%{customdata[1]}）</b><br>'
                '收盤價(未還原)：%{customdata[2]}<br>'
                '60日乖離：%{customdata[3]}｜120日乖離：%{customdata[4]}<br>'
                '收盤/480日高：%{customdata[5]}｜收盤/15日低：%{customdata[6]}<br>'
                '符合條件：%{customdata[7]}<br>'
                '買超來源：%{customdata[8]}<extra></extra>'
            ),
            showlegend=False,
        ))

    max_count = int(counts.max())
    for cat in categories:
        fig.add_annotation(
            x=tick_labels[cat], y=counts[cat], yshift=10,
            text=f'<b>{counts[cat]} 檔</b>', showarrow=False,
            font={'size': 12, 'color': COLOR['text_secondary']},
        )

    fig.update_layout(
        barmode='stack',
        bargap=0.3,
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': FONT, 'color': COLOR['text_secondary'], 'size': 12},
        margin={'l': 20, 'r': 20, 't': 30, 'b': 20},
        height=max(420, 64 * max_count + 140),
        yaxis={'visible': False, 'range': [0, max_count * 1.12], 'fixedrange': True},
        xaxis={
            'tickfont': {'size': 12, 'color': COLOR['text_secondary']},
            'linecolor': COLOR['border'],
            'fixedrange': True,
        },
        hoverlabel={'bgcolor': 'white', 'font_size': 12},
        dragmode=False,  # 鎖定圖表：禁止拖曳縮放，避免誤觸
    )
    return fig


# ── Page layout ────────────────────────────────────────────────────────────────

def layout():
    """以函式回傳 layout，讓 date picker 邊界每次進入頁面時取自最新快取。"""
    return html.Div([
        # ── Control bar ──────────────────────────────────────────────────────
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.Button('‹', id='ls-prev-day-btn', color='light', size='sm',
                                   style={'border': f"1px solid {COLOR['border']}"}),
                        dcc.DatePickerSingle(
                            id='ls-date-picker',
                            date=_CACHE['dates'][-1],
                            min_date_allowed=_CACHE['dates'][0],
                            max_date_allowed=_CACHE['dates'][-1],
                            display_format='YYYY-MM-DD',
                            style={'margin': '0 6px'},
                        ),
                        dbc.Button('›', id='ls-next-day-btn', color='light', size='sm',
                                   style={'border': f"1px solid {COLOR['border']}"}),
                    ], width='auto', className='d-flex align-items-center'),
                    dbc.Col(
                        dbc.Button(
                            '↻ 重新整理', id='ls-refresh-btn',
                            color='light', size='sm',
                            style={'border': f"1px solid {COLOR['border']}", 'fontSize': '13px'},
                        ),
                        width='auto', className='d-flex align-items-center',
                    ),
                    dbc.Col(
                        html.Div(id='ls-loading-indicator', style={
                            'fontSize': '12px', 'color': COLOR['text_muted'],
                        }),
                        className='d-flex align-items-center ms-auto',
                    ),
                ], className='g-2 align-items-center'),
                html.Div(_CONDITION_NOTE, style={
                    'fontSize': '11px', 'color': COLOR['text_muted'], 'marginTop': '6px',
                }),
            ], fluid=True),
        ], style={
            'backgroundColor': 'white',
            'borderBottom': f"1px solid {COLOR['border']}",
            'padding': '12px 0',
            'position': 'sticky', 'top': 0, 'zIndex': 100,
        }),

        # ── Main content ─────────────────────────────────────────────────────
        dbc.Container([
            dbc.Row([
                dbc.Col(html.Div(id='ls-kpi-date'), width=4),
                dbc.Col(html.Div(id='ls-kpi-stocks'), width=4),
                dbc.Col(html.Div(id='ls-kpi-groups'), width=4),
            ], className='g-2 mt-2'),

            dbc.Card([
                dbc.CardBody([
                    html.Div('族群家數分布', style={
                        'fontSize': '14px', 'fontWeight': '600',
                        'color': COLOR['text_heading'], 'margin': '4px 0 0 8px',
                    }),
                    dcc.Graph(id='ls-sector-chart', config={'displayModeBar': False}),
                ], style={'padding': '8px'}),
            ], style={**CARD_STYLE, 'marginTop': '12px'}),

            dbc.Card([
                dbc.CardBody([
                    html.Div('入選個股明細', style={
                        'fontSize': '14px', 'fontWeight': '600',
                        'color': COLOR['text_heading'], 'margin': '4px 0 2px 8px',
                    }),
                    html.Div('條件與各比率皆以還原價計算；收盤價欄為當日實際（未還原）收盤價',
                             style={'fontSize': '11px', 'color': COLOR['text_muted'],
                                    'margin': '0 0 8px 8px'}),
                    html.Div(id='ls-stock-table'),
                ], style={'padding': '8px'}),
            ], style={**CARD_STYLE, 'margin': '12px 0 24px'}),
        ], fluid=True),
    ])


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output('ls-date-picker', 'date'),
    Input('ls-prev-day-btn', 'n_clicks'),
    Input('ls-next-day-btn', 'n_clicks'),
    State('ls-date-picker', 'date'),
    prevent_initial_call=True,
)
def shift_date(_prev, _next, current):
    """‹ › 按鈕沿交易日移動。"""
    dates = _CACHE['dates']
    snapped = _snap_date(current[:10]) or dates[0]
    idx = dates.index(snapped)
    if ctx.triggered_id == 'ls-prev-day-btn':
        idx = max(0, idx - 1)
    else:
        idx = min(len(dates) - 1, idx + 1)
    return dates[idx]


@callback(
    Output('ls-date-picker', 'min_date_allowed'),
    Output('ls-date-picker', 'max_date_allowed'),
    Output('ls-date-picker', 'date', allow_duplicate=True),
    Output('ls-loading-indicator', 'children'),
    Input('ls-refresh-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def refresh_data(_n_clicks):
    """重新拉取 FinLab 資料並重算篩選結果。"""
    global _CACHE
    try:
        _CACHE = compute_screen()
        # 更新時間統一顯示於「訊號日」卡片，這裡僅在失敗時顯示訊息
        return _CACHE['dates'][0], _CACHE['dates'][-1], _CACHE['dates'][-1], ''
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, f'載入失敗：{e}'


@callback(
    Output('ls-sector-chart', 'figure'),
    Output('ls-stock-table', 'children'),
    Output('ls-kpi-date', 'children'),
    Output('ls-kpi-stocks', 'children'),
    Output('ls-kpi-groups', 'children'),
    Input('ls-date-picker', 'date'),
)
def update_view(picked):
    """依選定日期重繪族群分布與個股明細。"""
    date_str = _snap_date(picked[:10]) or _CACHE['dates'][0]
    rows = _stock_rows(date_str)
    fig = _build_figure(date_str, rows)

    n_groups = len({r['category'] for r in rows})
    kpi_date = kpi_card('訊號日（收盤資料）', date_str,
                        subtitle=f"供下一交易日操作參考｜資料更新：{_CACHE['updated']}")
    kpi_stocks = kpi_card('入選檔數', f'{len(rows)} 檔')
    kpi_groups = kpi_card('類股族群數', f'{n_groups} 個',
                          subtitle='依證交所產業分類')

    table = dash_table.DataTable(
        data=[{
            '代號': r['stock_id'],
            '名稱': r['name'],
            '類股': r['category'],
            '收盤價(未還原)': f"{r['close']:.2f}",
            '收盤/480日高': f"{r['high_ratio']:.1%}",
            '60日乖離': f"{r['bias60']:+.1%}",
            '120日乖離': f"{r['bias120']:+.1%}",
            '收盤/15日低': f"{r['low_ratio']:.2f}",
            '符合條件': r['matched'],
            '買超來源': r['sources'],
        } for r in rows],
        columns=[{'name': c, 'id': c} for c in (
            '代號', '名稱', '類股', '收盤價(未還原)', '收盤/480日高',
            '60日乖離', '120日乖離', '收盤/15日低', '符合條件', '買超來源')],
        sort_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={
            'fontFamily': FONT, 'fontSize': '13px', 'padding': '6px 10px',
            'textAlign': 'center', 'color': COLOR['text_secondary'],
        },
        style_header={
            'fontWeight': '600', 'backgroundColor': '#f9fafb',
            'color': COLOR['text_heading'],
            'borderBottom': f"1px solid {COLOR['border']}",
        },
        style_data={'borderBottom': f"1px solid {COLOR['border']}"},
    )

    return fig, table, kpi_date, kpi_stocks, kpi_groups
