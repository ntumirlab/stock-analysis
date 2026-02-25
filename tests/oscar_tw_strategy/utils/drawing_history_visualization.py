#!/usr/bin/env python3
"""
Drawing History Visualization Utility

建立專業的 K 線圖與指標視覺化，類似 TradingView
使用 Plotly 建立互動式圖表，包含：
- 蠟燭圖 (Candlestick)
- SAR 指標
- MACD 指標
- 買賣訊號標記
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional


def create_trading_visualization(
    stock_id: str,
    price_data: pd.DataFrame,
    sar_values: pd.Series,
    macd_dif: pd.Series,
    macd_dea: pd.Series,
    macd_histogram: pd.Series,
    buy_signals: pd.Series,
    sell_signals: pd.Series,
    position: pd.Series,
    trade_price: pd.Series,
    foreign_buy: pd.Series,
    trust_buy: pd.Series,
    dealer_buy: pd.Series,
    output_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    建立交易策略視覺化圖表
    
    Args:
        stock_id: 股票代碼
        price_data: 包含 open, high, low, close 的 DataFrame
        sar_values: SAR 指標值
        macd_dif: MACD DIF 線 (快線)
        macd_dea: MACD DEA 線 (慢線/訊號線)
        macd_histogram: MACD 柱狀圖
        buy_signals: 買入訊號 (布林值)
        sell_signals: 賣出訊號 (布林值)
        position: 持倉狀態 (布林值)
        trade_price: 交易價格 (開盤價)
        foreign_buy: 外資買超 (布林值)
        trust_buy: 投信買超 (布林值)
        dealer_buy: 自營商買超 (布林值)
        output_path: HTML 輸出路徑
        start_date: 起始日期 (可選)
        end_date: 結束日期 (可選)
        
    Returns:
        str: 輸出檔案路徑
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 日期範圍篩選
    if start_date:
        price_data = price_data.loc[start_date:]
        sar_values = sar_values.loc[start_date:]
        macd_dif = macd_dif.loc[start_date:]
        macd_dea = macd_dea.loc[start_date:]
        macd_histogram = macd_histogram.loc[start_date:]
        buy_signals = buy_signals.loc[start_date:]
        sell_signals = sell_signals.loc[start_date:]
        position = position.loc[start_date:]
        trade_price = trade_price.loc[start_date:]
        foreign_buy = foreign_buy.loc[start_date:]
        trust_buy = trust_buy.loc[start_date:]
        dealer_buy = dealer_buy.loc[start_date:]
    
    if end_date:
        price_data = price_data.loc[:end_date]
        sar_values = sar_values.loc[:end_date]
        macd_dif = macd_dif.loc[:end_date]
        macd_dea = macd_dea.loc[:end_date]
        macd_histogram = macd_histogram.loc[:end_date]
        buy_signals = buy_signals.loc[:end_date]
        sell_signals = sell_signals.loc[:end_date]
        position = position.loc[:end_date]
        trade_price = trade_price.loc[:end_date]
        foreign_buy = foreign_buy.loc[:end_date]
        trust_buy = trust_buy.loc[:end_date]
        dealer_buy = dealer_buy.loc[:end_date]
    
    # 移除無資料的日期 (非交易日) - 同時對齊所有數據
    valid_mask = price_data.notna().all(axis=1)
    valid_dates = price_data.index[valid_mask]
    
    # 確保所有序列都對齊到相同的交易日
    price_data = price_data.loc[valid_dates]
    sar_values = sar_values.reindex(valid_dates)
    macd_dif = macd_dif.reindex(valid_dates)
    macd_dea = macd_dea.reindex(valid_dates)
    macd_histogram = macd_histogram.reindex(valid_dates)
    buy_signals = buy_signals.reindex(valid_dates, fill_value=False)
    sell_signals = sell_signals.reindex(valid_dates, fill_value=False)
    trade_price = trade_price.reindex(valid_dates)
    foreign_buy = foreign_buy.reindex(valid_dates, fill_value=False)
    trust_buy = trust_buy.reindex(valid_dates, fill_value=False)
    dealer_buy = dealer_buy.reindex(valid_dates, fill_value=False)
    
    # ⚠️ 重要：這裡的 shift(1) 僅用於視覺化對齊，並不修正策略本身可能存在的 look-ahead bias
    # 假設 position 代表「今天收盤後決定，明天該持有」，而實際交易在「明天以開盤價」執行
    # 因此圖上的綠色背景應該從「明天」開始顯示，所以在此將 position 向後移動一天
    position_shifted = position.shift(1).reindex(valid_dates, fill_value=False).fillna(False)
    
    # 建立子圖：主圖(價格+SAR)、MACD圖
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,  # 共享 X 軸，確保拖曳和縮放時兩個圖表同步
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{stock_id} - Price & SAR', 'MACD'),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # === 主圖：蠟燭圖 ===
    # 綠色代表上漲 (收盤價 > 開盤價)，紅色代表下跌 (收盤價 < 開盤價)
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name='Price',
            increasing_line_color='#00c853',  # 綠色 - 上漲
            increasing_fillcolor='#00c853',
            decreasing_line_color='#ff1744',  # 紅色 - 下跌
            decreasing_fillcolor='#ff1744'
        ),
        row=1, col=1
    )
    
    # === 法人買超資訊（隱藏的散點圖用於顯示 hover 資訊）===
    institutional_text = []
    for date in price_data.index:
        foreign = '✓' if foreign_buy.loc[date] else '✗'
        trust = '✓' if trust_buy.loc[date] else '✗'
        dealer = '✓' if dealer_buy.loc[date] else '✗'
        institutional_text.append(f'外資:{foreign} 投信:{trust} 自營:{dealer}')
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['close'],
            mode='markers',
            marker=dict(size=0.1, opacity=0),  # 幾乎看不見的標記
            text=institutional_text,
            hovertemplate='<b>法人:</b> %{text}<extra></extra>',
            name='Institutional',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # === SAR 指標 ===
    fig.add_trace(
        go.Scatter(
            x=sar_values.index,
            y=sar_values,
            mode='markers',
            marker=dict(
                size=3,
                color='#ff9800',
                symbol='circle'
            ),
            name='SAR',
            hovertemplate='SAR: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # === 買入訊號標記 ===
    buy_dates = buy_signals[buy_signals].index
    if len(buy_dates) > 0:
        # 使用實際交易價格（開盤價）
        buy_prices = trade_price.loc[buy_dates]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='#4caf50',
                    symbol='triangle-up',
                    line=dict(width=2, color='white')
                ),
                text=[f'${p:.2f}' for p in buy_prices],
                textposition='bottom center',
                textfont=dict(size=9, color='#4caf50'),
                name='Buy Signal',
                hovertemplate='Buy<br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # === 賣出訊號標記 ===
    sell_dates = sell_signals[sell_signals].index
    if len(sell_dates) > 0:
        # 使用實際交易價格（開盤價）
        sell_prices = trade_price.loc[sell_dates]
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='#f44336',
                    symbol='triangle-down',
                    line=dict(width=2, color='white')
                ),
                text=[f'${p:.2f}' for p in sell_prices],
                textposition='top center',
                textfont=dict(size=9, color='#f44336'),
                name='Sell Signal',
                hovertemplate='Sell<br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # === 持倉期間背景色 ===
    # 找出持倉期間的連續區間（使用 shifted position）
    position_changes = position_shifted.astype(int).diff()
    hold_starts = position_changes[position_changes == 1].index
    hold_ends = position_changes[position_changes == -1].index
    
    # 處理開頭和結尾的特殊情況
    if position_shifted.iloc[0]:
        hold_starts = pd.DatetimeIndex([position_shifted.index[0]]).append(hold_starts)
    if position_shifted.iloc[-1]:
        hold_ends = hold_ends.append(pd.DatetimeIndex([position_shifted.index[-1]]))
    
    for start, end in zip(hold_starts, hold_ends):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor='rgba(76, 175, 80, 0.1)',
            layer='below',
            line_width=0,
            row=1, col=1
        )
    
    # === MACD 圖 ===
    # MACD 柱狀圖 - 綠色為正，紅色為負
    colors = ['#00c853' if val >= 0 else '#ff1744' for val in macd_histogram]
    fig.add_trace(
        go.Bar(
            x=macd_histogram.index,
            y=macd_histogram,
            name='MACD Histogram',
            marker_color=colors,
            showlegend=True,
            hovertemplate='Histogram: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # MACD DIF 線 (快線)
    fig.add_trace(
        go.Scatter(
            x=macd_dif.index,
            y=macd_dif,
            mode='lines',
            line=dict(color='#2196f3', width=1.5),
            name='MACD DIF',
            hovertemplate='DIF: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # MACD DEA 線 (慢線/訊號線)
    fig.add_trace(
        go.Scatter(
            x=macd_dea.index,
            y=macd_dea,
            mode='lines',
            line=dict(color='#ff9800', width=1.5),
            name='MACD DEA',
            hovertemplate='DEA: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # MACD 零軸線
    fig.add_hline(
        y=0,
        line=dict(color='gray', width=1, dash='dash'),
        row=2, col=1
    )
    
    # === 圖表佈局設定 ===
    fig.update_layout(
        title={
            'text': f'<b>{stock_id} - Trading Strategy Visualization</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_rangeslider_visible=False,
        hovermode='x unified',  # 顯示統一的垂直虛線和所有數據
        height=900,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        # 設定拖曳為主要互動方式
        dragmode='pan'
    )
    
    # X軸設定 - 兩個子圖都要設定相同的 X 軸屬性以確保同步
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#e0e0e0',
        type='category',  # 使用 category 類型來避免顯示非交易日
        rangeslider_visible=False,
        matches='x',  # 確保 X 軸匹配，實現同步縮放和拖曳
        showspikes=True,  # 顯示垂直虛線
        spikemode='across',  # 虛線橫跨所有子圖
        spikesnap='cursor',  # 虛線跟隨游標
        spikecolor='rgba(150, 150, 150, 0.8)',  # 虛線顏色（半透明灰色）
        spikethickness=1,  # 虛線粗細
        spikedash='dash'  # 虛線樣式
    )
    
    # 只在底部子圖顯示 X 軸標題
    fig.update_xaxes(
        title_text='Date',
        row=2, col=1
    )
    
    # Y軸設定
    fig.update_yaxes(
        title_text='Price',
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='#e0e0e0'
    )
    
    fig.update_yaxes(
        title_text='MACD',
        row=2, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='#e0e0e0'
    )
    
    # 儲存為 HTML
    config = {
        'scrollZoom': True,  # 滾輪縮放
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],  # 移除選取工具
        'doubleClick': 'reset',  # 雙擊重置視圖
        'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape']  # 新增繪圖工具
    }
    
    fig.write_html(
        str(output_path),
        config=config,
        include_plotlyjs='cdn'
    )
    
    return str(output_path)


def prepare_price_data(
    stock_id: str,
    market_data: dict,
    start_date: Optional[str] = None
) -> pd.DataFrame:
    """
    準備價格數據供視覺化使用
    
    Args:
        stock_id: 股票代碼
        market_data: 市場數據字典
        start_date: 起始日期
        
    Returns:
        pd.DataFrame: 包含 open, high, low, close 的價格數據
    """
    # 從市場數據中取得該股票的價格資料（使用原始價格）
    # 注意：這裡需要根據實際的資料結構調整
    
    price_df = pd.DataFrame({
        'open': market_data['open'][stock_id],
        'high': market_data['high'][stock_id],
        'low': market_data['low'][stock_id],
        'close': market_data['close'][stock_id]
    })
    
    if start_date:
        price_df = price_df.loc[start_date:]
    
    return price_df.dropna()
