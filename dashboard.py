import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from flask import Flask
from flask_autoindex import AutoIndex

from service.account_service import AccountService
from service.inventory_service import InventoryService
from service.order_service import OrderService
from service.balance_service import BalanceService
from tabs.order_history import OrderHistoryTab
from tabs.inventory_history import InventoryHistoryTab
from tabs.balance_history import BalanceHistoryTab


def create_app():
    # 初始化服務
    account_service = AccountService()
    order_service = OrderService()
    inventory_service = InventoryService()
    balance_service = BalanceService()

    # 初始化標籤
    order_history_tab = OrderHistoryTab(order_service)
    inventory_history_tab = InventoryHistoryTab(inventory_service)
    balance_history_tab = BalanceHistoryTab(balance_service)

    # 獲取根目錄路徑
    root_dir = os.path.dirname(os.path.abspath(__file__))
    assets_path = os.path.join(root_dir, 'assets')

    # 創建Flask server
    server = Flask(__name__)

    # 創建AutoIndex實例並指向assets資料夾
    auto_index = AutoIndex(server, browse_root=assets_path, add_url_rules=False, sort_by='last_modified', order='desc')

    # 添加路由規則
    @server.route('/assets/')
    @server.route('/assets/<path:path>')
    def autoindex(path='.'):
        return auto_index.render_autoindex(path)

    # 創建 Dash 應用
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        server=server,  # 使用我們自定義的Flask server
        suppress_callback_exceptions=True,
    )

    # 定義應用布局為函數
    def serve_layout():
        # 每次頁面加載時獲取最新帳戶列表
        accounts = account_service.get_all_accounts()
        account_options = [{'label': acc['account_name'], 'value': acc['account_id']} for acc in accounts]

        return html.Div(
            [
                # 標題區域
                html.Div(
                    [
                        html.H1("MirLab Dashboard", style={'display': 'inline-block', 'marginRight': '20px'}),
                        html.A(
                            "查看報表文件",
                            href="/assets/",
                            target="_blank",
                            className="btn btn-outline-primary",
                            style={'verticalAlign': 'middle'},
                        ),
                    ],
                    style={'margin': '10px 0'},
                ),
                # 帳戶選擇區域
                html.Div(
                    [
                        html.Label("選擇帳戶："),
                        dcc.Dropdown(
                            id='account-dropdown',
                            options=account_options,
                            value=account_options[0]['value'] if account_options else None,
                        ),
                    ],
                    style={'width': '30%', 'margin': '10px'},
                ),
                dcc.Tabs(
                    id="tabs",
                    value='tab-order-history',
                    children=[
                        dcc.Tab(label='下單歷史', value='tab-order-history', children=[order_history_tab.get_layout()]),
                        dcc.Tab(label='庫存', value='tab-inventory', children=[inventory_history_tab.get_layout()]),
                        dcc.Tab(
                            label='帳戶資金', value='tab-balance-history', children=[balance_history_tab.get_layout()]
                        ),
                    ],
                ),
            ]
        )

    # 使用函數式布局
    app.layout = serve_layout

    # 註冊所有標籤的回調函數
    order_history_tab.register_callbacks(app)
    inventory_history_tab.register_callbacks(app)
    balance_history_tab.register_callbacks(app)

    return app


app = create_app()
server = app.server

if __name__ == '__main__':
    app.run(debug=True)
