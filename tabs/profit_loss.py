from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import datetime

# 台股慣例：紅漲綠跌（與美股相反）
PROFIT_COLOR = '#d62728'
LOSS_COLOR = '#2ca02c'
NEUTRAL_COLOR = '#6c757d'


def _pnl_color(value):
    if value is None or value == 0:
        return NEUTRAL_COLOR
    return PROFIT_COLOR if value > 0 else LOSS_COLOR


def _format_money(value):
    if value is None:
        return "—"
    return f"${value:,.0f}"


def _format_ratio(value):
    """成本為 0 時 service 會回 None，顯示「—」而非假的 0%。"""
    if value is None:
        return "—"
    return f"{value:+.2f}%"


def _format_pnl_with_ratio(pnl, ratio):
    """金額與報酬率併成一行，例：$-1,618 (-1.74%)。

    報酬率為 None（成本為 0，例如尚未有任何平倉）時只留金額，
    不顯示空括號。
    """
    money = _format_money(pnl)
    if ratio is None:
        return money
    return f"{money} ({_format_ratio(ratio)})"


def _summary_card(title, block, hint):
    pnl = block['pnl']
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H5(title, className="card-title"),
                html.H3(_format_pnl_with_ratio(pnl, block['ratio']),
                        style={'color': _pnl_color(pnl)}),
                html.Small(hint, className="text-muted"),
            ])
        ])
    ], width=4)


class ProfitLossTab:
    def __init__(self, profit_loss_service):
        self.profit_loss_service = profit_loss_service

    def get_layout(self):
        """返回損益標籤的版面配置"""
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=90)

        pnl_style = [
            {'if': {'filter_query': '{pnl} > 0', 'column_id': ['pnl', 'ratio']},
             'color': PROFIT_COLOR},
            {'if': {'filter_query': '{pnl} < 0', 'column_id': ['pnl', 'ratio']},
             'color': LOSS_COLOR},
        ]

        return html.Div([
            html.Div([
                html.Label("選擇日期範圍（已實現損益）："),
                dcc.DatePickerRange(
                    id='pnl-date-range',
                    start_date=start_date,
                    end_date=end_date,
                    display_format='YYYY-MM-DD'
                )
            ], style={'width': '50%', 'margin': '10px'}),

            html.Div([
                html.Div(id='pnl-summary'),
                html.Small(
                    "報酬率分母為實際投入成本，不受出入金影響；"
                    "未實現損益取最近一次庫存快照，與日期範圍無關。",
                    className="text-muted"
                ),
            ], style={'margin': '20px'}),

            html.Div([
                html.H3("累積已實現損益", className='mb-3'),
                dcc.Graph(id='pnl-cumulative-graph')
            ], style={'margin': '20px'}),

            html.Div([
                html.H3("已實現損益明細", className='mb-3'),
                dash_table.DataTable(
                    id='pnl-realized-table',
                    columns=[
                        {'name': '成交日', 'id': 'trade_date'},
                        {'name': '股票 ID', 'id': 'stock_id'},
                        {'name': '股票名稱', 'id': 'stock_name'},
                        {'name': '數量(張)', 'id': 'quantity'},
                        {'name': '成交價', 'id': 'price'},
                        {'name': '損益', 'id': 'pnl'},
                        {'name': '報酬率(%)', 'id': 'ratio'},
                    ],
                    data=[],
                    page_size=15,
                    sort_action='native',
                    style_table={'overflowX': 'auto'},
                    style_data_conditional=pnl_style,
                )
            ], style={'margin': '20px'}),

            html.Div([
                html.H3("未實現損益（現有持股）", className='mb-3'),
                dash_table.DataTable(
                    id='pnl-unrealized-table',
                    columns=[
                        {'name': '股票 ID', 'id': 'stock_id'},
                        {'name': '股票名稱', 'id': 'stock_name'},
                        {'name': '數量(張)', 'id': 'quantity'},
                        {'name': '成本均價', 'id': 'cost_price'},
                        {'name': '最新價格', 'id': 'last_price'},
                        {'name': '損益', 'id': 'pnl'},
                        {'name': '報酬率(%)', 'id': 'ratio'},
                    ],
                    data=[],
                    page_size=15,
                    sort_action='native',
                    style_table={'overflowX': 'auto'},
                    style_data_conditional=pnl_style,
                )
            ], style={'margin': '20px'}),
        ])

    def register_callbacks(self, app):
        """註冊該標籤所需的回調函數"""

        @app.callback(
            Output('pnl-summary', 'children'),
            Input('account-dropdown', 'value'),
            Input('pnl-date-range', 'start_date'),
            Input('pnl-date-range', 'end_date'),
        )
        def update_summary(selected_account, start_date, end_date):
            date_range = _parse_range(start_date, end_date)
            if not selected_account or date_range is None:
                return html.Div("請選擇帳戶與日期範圍", className='text-muted')

            summary = self.profit_loss_service.get_summary(
                selected_account, date_range[0], date_range[1]
            )

            return dbc.Row([
                _summary_card("已實現損益", summary['realized'],
                              f"{summary['realized']['count']} 筆平倉"),
                _summary_card("未實現損益", summary['unrealized'],
                              f"{summary['unrealized']['count']} 檔持股"),
                _summary_card("合計", summary['total'],
                              f"投入成本 {_format_money(summary['total']['cost'])}"),
            ])

        @app.callback(
            Output('pnl-cumulative-graph', 'figure'),
            Input('account-dropdown', 'value'),
            Input('pnl-date-range', 'start_date'),
            Input('pnl-date-range', 'end_date'),
        )
        def update_cumulative_graph(selected_account, start_date, end_date):
            date_range = _parse_range(start_date, end_date)
            if not selected_account or date_range is None:
                return _empty_figure("請選擇帳戶與日期範圍")

            series = self.profit_loss_service.get_cumulative_realized(
                selected_account, date_range[0], date_range[1]
            )
            if not series:
                return _empty_figure("此期間沒有已實現損益")

            dates = [point['date'] for point in series]
            cumulative = [point['cumulative_pnl'] for point in series]
            line_color = _pnl_color(cumulative[-1])

            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=dates,
                y=cumulative,
                mode='lines+markers',
                name='累積已實現損益',
                line={'color': line_color},
                hovertemplate='%{x}<br>累積：$%{y:,.0f}<extra></extra>',
            ))
            figure.add_hline(y=0, line_dash='dot', line_color=NEUTRAL_COLOR)
            figure.update_layout(
                xaxis_title='成交日',
                yaxis_title='累積損益 (元)',
                margin={'l': 60, 'r': 20, 't': 20, 'b': 40},
                hovermode='x unified',
            )
            return figure

        @app.callback(
            Output('pnl-realized-table', 'data'),
            Output('pnl-unrealized-table', 'data'),
            Input('account-dropdown', 'value'),
            Input('pnl-date-range', 'start_date'),
            Input('pnl-date-range', 'end_date'),
        )
        def update_tables(selected_account, start_date, end_date):
            date_range = _parse_range(start_date, end_date)
            if not selected_account or date_range is None:
                return [], []

            realized = self.profit_loss_service.get_realized_records(
                selected_account, date_range[0], date_range[1]
            )
            # 不帶日期＝取最新快照；未實現沒有區間概念，不受上方日期選擇影響
            unrealized = self.profit_loss_service.get_unrealized_records(selected_account)

            return [_round_row(row) for row in realized], [_round_row(row) for row in unrealized]


def _parse_range(start_date, end_date):
    """把 DatePickerRange 的字串轉成 date；任一端缺失或格式異常時回 None。"""
    if not start_date or not end_date:
        return None
    try:
        start = datetime.datetime.fromisoformat(start_date).date()
        end = datetime.datetime.fromisoformat(end_date).date()
    except (ValueError, TypeError):
        return None
    return start, end


def _round_row(row):
    """表格顯示用的四捨五入；保留數值型別讓條件式配色與排序可用。"""
    rounded = dict(row)
    for key in ('quantity', 'price', 'cost_price', 'last_price'):
        if rounded.get(key) is not None:
            rounded[key] = round(float(rounded[key]), 3)
    for key in ('pnl', 'ratio'):
        if rounded.get(key) is not None:
            rounded[key] = round(float(rounded[key]), 2)
    return rounded


def _empty_figure(message):
    figure = go.Figure()
    figure.update_layout(
        annotations=[{
            'text': message,
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': 0.5, 'showarrow': False,
            'font': {'color': NEUTRAL_COLOR},
        }],
        xaxis={'visible': False},
        yaxis={'visible': False},
        margin={'l': 20, 'r': 20, 't': 20, 'b': 20},
    )
    return figure
