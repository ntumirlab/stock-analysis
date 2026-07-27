from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import datetime
import pandas as pd

from tabs.components import card_grid, summary_card

class BalanceHistoryTab:
    def __init__(self, balance_service):
        self.balance_service = balance_service

    def get_layout(self):
        """返回帳戶資金標籤的版面配置"""
        # 計算默認日期範圍：過去一年
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365)
        
        return html.Div([
            # 日期選擇和概要摘要
            html.Div([
                html.Div([
                    html.Label("選擇日期範圍："),
                    dcc.DatePickerRange(
                        id='balance-date-range',
                        start_date=start_date,
                        end_date=end_date,
                        display_format='YYYY-MM-DD'
                    )
                ], style={'width': '50%', 'margin': '10px'}),
                
                html.Div([
                    html.H3("帳戶資金摘要", className='mb-3'),
                    html.Div(id='balance-summary', className='summary-box')
                ], style={'margin': '20px'})
            ]),
            # 此處原為 className='row'，但兩個子元素都不是 col。Bootstrap 的 .row 是
            # display:flex，子元素會變成寬度依內容而定的 flex item——摘要區塊因此撐成
            # 卡片的自然總寬而非容器寬度，把整頁推出左右捲軸。改回一般區塊排版
            
            # 資金水位趨勢圖
            html.Div([
                html.H3("資金水位趨勢", className='mb-3'),
                dcc.Graph(id='balance-trend-graph')
            ], style={'margin': '20px'}),
            
            # 月度資產變化率熱力圖
            # 分母是月初總資產、分子是月末減月初，出入金會被算進去，故不能稱為報酬率。
            # 真報酬率需以「已實現＋未實現損益 ÷ 投入成本」計算，待累積足夠損益資料後再做
            html.Div([
                html.H3("月度資產變化率", className='mb-3'),
                html.Small("含出入金影響，非投資報酬率", className="text-muted"),
                dcc.Graph(id='monthly-return-heatmap')
            ], style={'margin': '20px'})
        ])
        
    def register_callbacks(self, app):
        """註冊該標籤所需的回調函數"""
        
        # 帳戶資金摘要回調
        @app.callback(
            Output('balance-summary', 'children'),
            Input('account-dropdown', 'value')
        )
        def update_balance_summary(selected_account):
            if not selected_account:
                return html.Div("請選擇帳戶", className='text-muted')
                
            # 獲取最新餘額數據
            latest_balance = self.balance_service.get_latest_balance(selected_account)
            
            if not latest_balance:
                return html.Div("無資金數據", className='text-muted')
                
            # 解析日期
            try:
                fetch_date = datetime.datetime.strptime(
                    latest_balance['fetch_timestamp'], 
                    "%Y-%m-%d %H:%M:%S"
                ).strftime("%Y-%m-%d")
            except:
                fetch_date = "未知日期"
                
            # 創建摘要卡片
            # 未交割款獨立成卡，銀行餘額 ＋ 未交割款 ＝ 可動用現金 的關係因此
            # 看得出來，不必靠文字說明（也就不會有某張卡多一行導致整排參差）
            summary_cards = card_grid([
                summary_card("銀行餘額", f"${latest_balance['bank_balance']:,.2f}",
                             value_class="card-text text-primary"),
                summary_card("未交割款", f"${latest_balance['settlements']:,.2f}",
                             value_class="card-text text-secondary"),
                summary_card("可動用現金", f"${latest_balance['adjusted_bank_balance']:,.2f}",
                             value_class="card-text text-info"),
                summary_card("持股市值", f"${latest_balance['market_value']:,.2f}",
                             value_class="card-text text-success"),
                summary_card("總資產", f"${latest_balance['total_assets']:,.2f}",
                             value_class="card-text text-dark"),
            ])
            
            return html.Div([
                html.P(f"最新資料日期：{fetch_date}", className="text-muted mb-3"),
                summary_cards
            ])
            
        # 資金水位趨勢圖回調
        @app.callback(
            Output('balance-trend-graph', 'figure'),
            [Input('account-dropdown', 'value'),
             Input('balance-date-range', 'start_date'),
             Input('balance-date-range', 'end_date')]
        )
        def update_balance_trend(selected_account, start_date, end_date):
            if not selected_account or not start_date or not end_date:
                return go.Figure()
                
            try:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
            except:
                # 處理已經是日期格式的情況
                if isinstance(start_date, str) and isinstance(end_date, str):
                    try:
                        start_date = datetime.datetime.fromisoformat(start_date).date()
                        end_date = datetime.datetime.fromisoformat(end_date).date()
                    except:
                        return go.Figure()
                else:
                    return go.Figure()
            
            # 獲取趨勢數據
            trend_data = self.balance_service.get_balance_trend_data(
                selected_account, 
                start_date,
                end_date
            )
            
            if not trend_data:
                fig = go.Figure()
                fig.update_layout(
                    title="無資金水位數據",
                    xaxis_title="日期",
                    yaxis_title="金額"
                )
                return fig
                
            # 轉換為DataFrame便於處理
            df = pd.DataFrame(trend_data)
            
            # 創建趨勢圖
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['date'], 
                y=df['adjusted_bank_balance'],
                mode='lines+markers',
                name='可動用現金',
                line=dict(color='#36a2eb', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df['date'], 
                y=df['market_value'],
                mode='lines+markers',
                name='持股市值',
                line=dict(color='#4bc0c0', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df['date'], 
                y=df['total_assets'],
                mode='lines+markers',
                name='總資產',
                line=dict(color='#ff6384', width=2)
            ))
            
            # 更新佈局
            fig.update_layout(
                title="資金水位趨勢",
                xaxis_title="日期",
                yaxis_title="金額",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                template="plotly_white"
            )
            
            return fig
            
        # 月度回報率熱力圖回調
        @app.callback(
            Output('monthly-return-heatmap', 'figure'),
            [Input('account-dropdown', 'value')]
        )
        def update_monthly_return_heatmap(selected_account):
            if not selected_account:
                return go.Figure()
                
            # 獲取熱力圖數據
            heatmap_data, years, max_return, min_return = self.balance_service.get_monthly_return_data(
                selected_account
            )

            if not heatmap_data:
                fig = go.Figure()
                fig.update_layout(
                    title="無足夠數據計算月度資產變化率",
                    xaxis_title="月份",
                    yaxis_title="年份"
                )
                return fig
            
            # 所有月份名稱和顯示順序
            all_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            month_names = {
                '01': '一月', '02': '二月', '03': '三月', '04': '四月',
                '05': '五月', '06': '六月', '07': '七月', '08': '八月',
                '09': '九月', '10': '十月', '11': '十一月', '12': '十二月'
            }
            all_month_labels = [month_names[m] for m in all_months]
            
            # 處理數據，確保只有有效資料被考慮
            all_years = sorted(set([item['year'] for item in heatmap_data]))
            
            # 創建數據矩陣
            z_matrix = []
            annotations = []
            
            # 為每個年份創建一行資料，包含所有12個月
            for year in all_years:
                row_data = []
                for month in all_months:
                    # 查找對應的變化率
                    value = None
                    for item in heatmap_data:
                        if item['year'] == year and item['month'] == month:
                            value = item['return']
                            break
                    
                    row_data.append(value)
                    
                    # 如果有值，添加註釋
                    if value is not None:
                        month_idx = all_months.index(month)
                        year_idx = all_years.index(year)
                        
                        # 確定文字顏色
                        text_color = "black" if min_return <= value <= (min_return + max_return)/2 else "white"
                        
                        annotations.append({
                            "x": month_idx,
                            "y": year_idx,
                            "text": f"{value:.2f}%",
                            "showarrow": False,
                            "font": {"color": text_color, "size": 10}
                        })
                
                z_matrix.append(row_data)
            
            # 創建熱力圖
            fig = go.Figure(data=go.Heatmap(
                z=z_matrix,
                x=all_month_labels,
                y=all_years,
                colorscale='RdBu_r',
                showscale=True,
                colorbar={"title": "變化率 (%)"},
                # 確保數據範圍適當
                zmid=0,  # 中間點設為0
                zmin=min(min_return, -0.1),  # 確保有一定範圍
                zmax=max(max_return, 0.1),
                # 設置缺失值顏色
                hoverongaps=False
            ))
            
            # 添加註釋
            fig.update_layout(annotations=annotations)
            
            # 更新佈局
            fig.update_layout(
                title="月度資產變化率（含出入金）",
                xaxis_title="月份",
                yaxis_title="年份",
                xaxis=dict(
                    # 固定顯示所有月份
                    tickmode='array',
                    tickvals=list(range(len(all_month_labels))),
                    ticktext=all_month_labels,
                    showgrid=True
                ),
                yaxis=dict(
                    # 固定年份順序
                    categoryorder="array",
                    categoryarray=all_years,
                    showgrid=True
                ),
                plot_bgcolor='white',
                margin=dict(l=50, r=50, t=80, b=50),
                template="plotly_white",
                width=900
            )
            
            # 確保有標示網格
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
            
            return fig



