from dash import dcc, html, dash_table, Input, Output, callback
import datetime


class InventoryHistoryTab:
    def __init__(self, inventory_service):
        self.inventory_service = inventory_service

    def get_layout(self):
        """返回庫存歷史標籤的版面配置"""
        return html.Div(
            [
                html.Div(
                    [
                        html.Label("選擇日期："),
                        dcc.DatePickerSingle(
                            id='inventory-date-picker', date=datetime.date.today(), display_format='YYYY-MM-DD'
                        ),
                    ],
                    style={'width': '30%', 'margin': '10px'},
                ),
                html.Div(
                    [
                        dash_table.DataTable(
                            id='inventory-history-table',
                            columns=[
                                {'name': '股票 ID', 'id': 'stock_id'},
                                {'name': '股票名稱', 'id': 'stock_name'},
                                {'name': '數量(張)', 'id': 'quantity'},
                                {'name': '最新價格', 'id': 'last_price'},
                                {'name': '損益', 'id': 'pnl'},
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

        # 更新庫存歷史表格
        @app.callback(
            Output('inventory-history-table', 'data'),
            Input('account-dropdown', 'value'),
            Input('inventory-date-picker', 'date'),
        )
        def update_inventory_history(selected_account, selected_date):
            if not selected_account or not selected_date:
                return []

            try:
                query_date = datetime.datetime.strptime(selected_date, "%Y-%m-%d").date()
            except Exception as e:
                # 如果日期字符串已經是 datetime 格式
                if isinstance(selected_date, str):
                    try:
                        query_date = datetime.datetime.fromisoformat(selected_date).date()
                    except:
                        return []
                else:
                    return []

            inventories = self.inventory_service.get_inventories_by_account_and_date(selected_account, query_date)
            return inventories
