"""GoldenAI Lite 的簡易 dashboard：下單歷史／庫存／帳戶資金／推薦清單。

與主 dashboard.py 的差異：不掛 AutoIndex（Lite 無報表檔案、避免檔案系統
瀏覽面），另加推薦清單頁。tabs 直接複用主 dashboard 的模組。
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from dao.recommendation_dao import RecommendationDAO
from service.account_service import AccountService
from service.inventory_service import InventoryService
from service.order_service import OrderService
from service.balance_service import BalanceService
from tabs.order_history import OrderHistoryTab
from tabs.inventory_history import InventoryHistoryTab
from tabs.balance_history import BalanceHistoryTab
from tabs.recommendation_list import RecommendationListTab


def create_app():
    account_service = AccountService()

    order_history_tab = OrderHistoryTab(OrderService())
    inventory_history_tab = InventoryHistoryTab(InventoryService())
    balance_history_tab = BalanceHistoryTab(BalanceService())
    recommendation_tab = RecommendationListTab(RecommendationDAO(frequency="weekly"))

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    def serve_layout():
        # 每次頁面加載時獲取最新帳戶列表
        accounts = account_service.get_all_accounts()
        account_options = [{'label': acc['account_name'], 'value': acc['account_id']} for acc in accounts]

        return html.Div([
            html.Div([
                html.H1("GoldenAI Lite Dashboard"),
            ], style={'margin': '10px 0'}),

            html.Div([
                html.Label("選擇帳戶："),
                dcc.Dropdown(
                    id='account-dropdown',
                    options=account_options,
                    value=account_options[0]['value'] if account_options else None
                )
            ], style={'width': '30%', 'margin': '10px'}),

            dcc.Tabs(id="tabs", value='tab-order-history', children=[
                dcc.Tab(label='下單歷史', value='tab-order-history', children=[
                    order_history_tab.get_layout()
                ]),
                dcc.Tab(label='庫存', value='tab-inventory', children=[
                    inventory_history_tab.get_layout()
                ]),
                dcc.Tab(label='帳戶資金', value='tab-balance-history', children=[
                    balance_history_tab.get_layout()
                ]),
                dcc.Tab(label='推薦清單', value='tab-recommendations', children=[
                    recommendation_tab.get_layout()
                ]),
            ])
        ])

    app.layout = serve_layout

    order_history_tab.register_callbacks(app)
    inventory_history_tab.register_callbacks(app)
    balance_history_tab.register_callbacks(app)
    recommendation_tab.register_callbacks(app)

    return app


app = create_app()
server = app.server

if __name__ == '__main__':
    app.run(debug=True, port=8060)
