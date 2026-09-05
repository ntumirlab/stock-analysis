"""
Alan 台股儀表板
===============
以 Dash Pages 整合多個儀表板頁面，共用同一端口與視覺風格：

    /                多空轉折模型（TAIEX K 線 + S4 分數）
    /leading-sector  領先潛力族群（六條件篩選 + 族群家數分布）

啟動：python -m alan_dashboard.app（於 repo 根目錄執行）
URL : http://localhost:8052
"""

from dotenv import load_dotenv
load_dotenv('.env')

import dash
from dash import html
import dash_bootstrap_components as dbc

from utils.finlab_auth import login_finlab
from alan_dashboard.theme import COLOR

# FinLab 登入須在 app 建立前完成：use_pages 會在建立 app 時載入 pages/
#（相對於本檔所在目錄），而 leading_sector 於 import 階段即需拉取資料計算
login_finlab()

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='Alan 台股儀表板',
)
app.config.suppress_callback_exceptions = True
server = app.server  # gunicorn 進入點

app.layout = html.Div([
    # ── 全域導覽列 ────────────────────────────────────────────────────────────
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Span('Alan 台股儀表板', style={
                        'fontWeight': '700', 'color': COLOR['text_heading'],
                        'fontSize': 'clamp(15px, 3vw, 18px)',
                    }),
                    width='auto', className='d-flex align-items-center',
                ),
                dbc.Col(
                    dbc.Nav(
                        [
                            dbc.NavLink(
                                page['name'], href=page['relative_path'],
                                active='exact', style={'fontSize': '14px'},
                            )
                            for page in sorted(dash.page_registry.values(),
                                               key=lambda p: p.get('order') or 0)
                        ],
                        pills=True,
                    ),
                    width='auto', className='d-flex align-items-center',
                ),
            ], className='g-3 align-items-center'),
        ], fluid=True),
    ], style={
        'backgroundColor': 'white',
        'borderBottom': f"1px solid {COLOR['border']}",
        'padding': '10px 0',
    }),

    dash.page_container,
], style={'backgroundColor': COLOR['bg_page'], 'minHeight': '100vh'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8052)
