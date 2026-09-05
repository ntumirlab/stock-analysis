"""
Alan 儀表板共用樣式與元件
=========================
供 alan_dashboards.py 與 pages/ 下各頁面共用，統一視覺風格。
"""

from zoneinfo import ZoneInfo

import dash_bootstrap_components as dbc
from dash import html

TZ = ZoneInfo('Asia/Taipei')

COLOR = {
    'text_heading':   '#1a202c',
    'text_secondary': '#374151',
    'text_muted':     '#6b7280',
    'accent':         '#1d4ed8',
    'border':         '#e5e7eb',
    'grid_zero':      '#d1d5db',
    'bg_page':        '#f0f2f5',
}

FONT = 'system-ui, -apple-system, sans-serif'

CARD_STYLE = {
    'border': f"1px solid {COLOR['border']}",
    'borderRadius': '8px',
    'boxShadow': '0 1px 4px rgba(0,0,0,0.06)',
    'backgroundColor': 'white',
}


def kpi_card(label: str, value, subtitle: str = '') -> dbc.Card:
    """統一樣式的 KPI 卡片。"""
    children = [
        html.Div(label, style={
            'fontSize': '12px', 'color': COLOR['text_muted'],
            'marginBottom': '4px', 'fontWeight': '500',
        }),
        html.Div(str(value), style={
            'fontSize': '22px', 'fontWeight': '700',
            'color': COLOR['text_heading'],
        }),
    ]
    if subtitle:
        children.append(html.Div(subtitle, style={
            'fontSize': '11px', 'color': COLOR['text_muted'],
            'marginTop': '6px', 'lineHeight': '1.4',
        }))
    return dbc.Card(dbc.CardBody(children, style={'padding': '14px 18px'}),
                    style=CARD_STYLE)


def section_card(children, **style_overrides) -> dbc.Card:
    """統一樣式的內容卡片（圖表、表格外框）。"""
    style = {**CARD_STYLE, **style_overrides}
    return dbc.Card(dbc.CardBody(children, style={'padding': '8px'}), style=style)
