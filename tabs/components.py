"""各分頁共用的版面元件。

抽出來的理由：帳戶資金與損益兩頁各自有一份「摘要卡片列」，同名不同介面，
排版機制也不同（一邊 CSS Grid、一邊 Bootstrap 欄位），加卡片時很容易只改到一邊。
"""

from dash import html
import dash_bootstrap_components as dbc


def summary_card(title, value, value_class=None, value_style=None, hint=None):
    """摘要卡片：標題＋數值，可選一行小字說明。

    Args:
        title (str): 卡片標題
        value (str): 已格式化的數值文字
        value_class (str, optional): 數值的 CSS class（如 Bootstrap 的 text-primary）
        value_style (dict, optional): 數值的行內樣式（如台股紅綠配色）
        hint (str, optional): 數值下方的小字

    Returns:
        dbc.Card

    Note:
        同一列卡片的 hint 要嘛都給、要嘛都不給——只有一張多一行會讓整排高度參差。
    """
    body = [
        html.H5(title, className="card-title"),
        html.H3(value, className=value_class or "card-text", style=value_style or {}),
    ]
    if hint is not None:
        body.append(html.Small(hint, className="text-muted"))

    return dbc.Card([dbc.CardBody(body)])


def card_grid(cards, min_width="200px"):
    """把卡片排成自動換行的等寬格線。

    用 CSS Grid 而非 dbc.Row/Col：1fr 的定義是「扣掉 gap 後平分剩餘空間」，
    因此卡片寬度加間距恆等於容器寬度。Bootstrap 欄位在欄數不整除 12 時，
    負邊距與 min-width:auto 會讓整列超出容器、把頁面推出左右捲軸。

    Args:
        cards (list): summary_card 產生的卡片
        min_width (str): 單張卡片的最小寬度，窄於此值就換行

    Returns:
        html.Div
    """
    return html.Div(cards, style={
        'display': 'grid',
        'gridTemplateColumns': f'repeat(auto-fit, minmax({min_width}, 1fr))',
        'gap': '1rem',
    })
