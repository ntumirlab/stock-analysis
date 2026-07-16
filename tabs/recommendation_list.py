from dash import html
import dash_bootstrap_components as dbc


class RecommendationListTab:
    """最新一期推薦清單（Lite dashboard 用）。

    資料在 get_layout() 內讀取：dashboard 的 serve_layout 每次頁面載入都會
    重新呼叫 get_layout，所以重新整理即可看到 fetcher 剛同步的清單，
    不需要 callback。
    """

    def __init__(self, sections):
        """sections: [(區塊標題, RecommendationDAO), ...]，每組各渲染最新一期。"""
        self.sections = sections

    def get_layout(self):
        return html.Div([
            self._render_section(label, dao.get_latest())
            for label, dao in self.sections
        ])

    @staticmethod
    def _render_section(label, record):
        if record is None:
            return html.Div([
                html.H3(label, className='mb-3'),
                html.P("尚無推薦清單資料（等待每日 07:50 的清單同步）", className="text-muted"),
            ], style={'margin': '20px'})

        def fmt_price(value):
            return f"{value:g}" if value is not None else "—"

        header = html.Thead(html.Tr([
            html.Th("順位"), html.Th("代號"), html.Th("名稱"),
            html.Th("觀點"), html.Th("目標價"), html.Th("停損價"),
        ]))
        body = html.Tbody([
            html.Tr([
                html.Td(idx + 1),
                html.Td(stock.id),
                html.Td(stock.name or "—"),
                html.Td(stock.sentiment),
                html.Td(fmt_price(stock.TP)),
                html.Td(fmt_price(stock.SL)),
            ])
            for idx, stock in enumerate(record.stocks)
        ])

        return html.Div([
            html.H3(f"{label}（{record.date}，共 {len(record.stocks)} 檔）", className='mb-3'),
            dbc.Table([header, body], bordered=True, hover=True, striped=True),
        ], style={'margin': '20px'})

    def register_callbacks(self, app):
        """本頁無互動元件，保留介面與其他 tab 一致。"""
