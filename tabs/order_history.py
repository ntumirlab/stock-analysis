from dash import dcc, html, dash_table, Input, Output, callback
import datetime


class OrderHistoryTab:
    def __init__(self, order_service):
        self.order_service = order_service

    def get_layout(self):
        """返回下單歷史標籤的版面配置"""
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("年："),
                                dcc.Dropdown(
                                    id='order-year-dropdown',
                                    options=[],  # 將在回調中動態更新
                                    value=None,
                                ),
                            ],
                            style={'width': '30%', 'display': 'inline-block'},
                        ),
                        html.Div(
                            [
                                html.Label("月："),
                                dcc.Dropdown(
                                    id='order-month-dropdown',
                                    options=[],  # 將在回調中動態更新
                                    value=None,
                                ),
                            ],
                            style={'width': '30%', 'display': 'inline-block'},
                        ),
                        html.Div(
                            [
                                html.Label("日："),
                                dcc.Dropdown(
                                    id='order-day-dropdown',
                                    options=[],  # 將在回調中動態更新
                                    value=None,
                                ),
                            ],
                            style={'width': '30%', 'display': 'inline-block'},
                        ),
                    ],
                    style={'margin': '10px'},
                ),
                html.Div(
                    [
                        dash_table.DataTable(
                            id='order-history-table',
                            columns=[
                                {'name': '下單時間戳記', 'id': 'order_timestamp'},
                                {'name': '買/賣', 'id': 'action'},
                                {'name': '股票 ID', 'id': 'stock_id'},
                                {'name': '股票名稱', 'id': 'stock_name'},
                                {'name': '數量(張)', 'id': 'quantity'},
                                {'name': '委託價', 'id': 'limit_price'},
                                {'name': '追價', 'id': 'extra_bid_pct'},
                                {'name': '資金類型', 'id': 'order_condition'},
                                {'name': '模擬', 'id': 'view_only'},
                            ],
                            data=[],
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                        )
                    ],
                    style={'margin': '20px'},
                ),
            ]
        )

    def register_callbacks(self, app):
        """註冊該標籤所需的回調函數"""

        # 1. 更新年份下拉選單
        @app.callback(
            Output('order-year-dropdown', 'options'),
            Output('order-year-dropdown', 'value'),
            Input('account-dropdown', 'value'),
        )
        def update_order_years(selected_account):
            if not selected_account:
                return [], None

            years = self.order_service.get_available_years(selected_account)
            default_year = years[0]['value'] if years else None
            return years, default_year

        # 2. 更新月份下拉選單
        @app.callback(
            Output('order-month-dropdown', 'options'),
            Output('order-month-dropdown', 'value'),
            Input('account-dropdown', 'value'),
            Input('order-year-dropdown', 'value'),
        )
        def update_order_months(selected_account, selected_year):
            if not selected_account or not selected_year:
                return [], None

            months = self.order_service.get_available_months(selected_account, selected_year)
            default_month = months[0]['value'] if months else None
            return months, default_month

        # 3. 更新日期下拉選單
        @app.callback(
            Output('order-day-dropdown', 'options'),
            Output('order-day-dropdown', 'value'),
            Input('account-dropdown', 'value'),
            Input('order-year-dropdown', 'value'),
            Input('order-month-dropdown', 'value'),
        )
        def update_order_days(selected_account, selected_year, selected_month):
            if not selected_account or not selected_year or not selected_month:
                return [], None

            days = self.order_service.get_available_days(selected_account, selected_year, selected_month)
            default_day = days[0]['value'] if days else None
            return days, default_day

        # 4. 更新訂單歷史表格
        @app.callback(
            Output('order-history-table', 'data'),
            Input('account-dropdown', 'value'),
            Input('order-year-dropdown', 'value'),
            Input('order-month-dropdown', 'value'),
            Input('order-day-dropdown', 'value'),
        )
        def update_order_history(selected_account, selected_year, selected_month, selected_day):
            if not selected_account or not selected_year or not selected_month or not selected_day:
                return []

            try:
                date_str = f"{selected_year}-{selected_month}-{selected_day}"
                query_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception as e:
                return []

            orders = self.order_service.get_orders_by_account_and_date(selected_account, query_date)
            return orders
