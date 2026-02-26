from finlab import data
from finlab import backtest

with data.universe(market='TSE_OTC'):
    market_value = data.get("etl:market_value")
    adj_close = data.get('etl:adj_close')
    eps = data.get('financial_statement:每股盈餘')
    revenue_growth_yoy = data.get('monthly_revenue:去年同月增減(%)')


# 計算季線（60日移動平均）並判斷季線是否上升
ma60 = adj_close.average(60)
# 使用rise函數檢查ma60是否在過去nwindow天連續上升
ma60_rising = ma60.rise(1)

# 股價是否大於季線
above_ma60 = adj_close > ma60
# 股價突破三個月高點
high_3m = adj_close.rolling(60).max()
price_break_high_3m = adj_close >= high_3m


# 過去四個季度的盈餘總和大於2元，且連續兩年都滿足這個條件
cumulative_eps_last_year = eps.rolling(4).sum() > 2
cumulative_eps_year_before_last = eps.shift(4).rolling(4).sum() > 2
eps_condition = cumulative_eps_last_year & cumulative_eps_year_before_last

# 設定營業額成長的條件
revenue_growth_condition = revenue_growth_yoy > 30


# 買入條件
buy_condition = (
    # 技術面
    above_ma60
    & ma60_rising
    & price_break_high_3m
    &
    # 基本面
    eps_condition
    & revenue_growth_condition
)

below_ma60 = adj_close < ma60
not_recover_in_5_days = below_ma60.sustain(5)
ma60_falling = ma60 < ma60.shift(1)
# 計算每天股票價格相比前一天收盤價的變動百分比
price_change_percent = adj_close.pct_change()
# 設定停板條件，即價格跌幅小於或等於-10%
hit_drop_limit = price_change_percent <= -0.095


# 賣出條件
sell_condition = not_recover_in_5_days | ma60_falling | hit_drop_limit

position = buy_condition.hold_until(sell_condition)

# 設定起始日期
start_date = '2015-12-31'
position = position.loc[start_date:]

# 現在將市值數據整合進來，每個重平衡周期選出市值前10大的股票
position = market_value[position].is_largest(10)

# 使用 sim 函數進行模擬
report = backtest.sim(position, resample='M', name="吳Peter策略選股_10檔", upload=False)
