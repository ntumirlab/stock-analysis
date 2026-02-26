from finlab.plot import StrategyReturnStats

# 回測起始時間
start_date = '2022-12-31'
end_date = '2023-07-31'

# 選定策略範圍
strategy_names = ['藏獒_2016', '吳Peter策略選股']

report = StrategyReturnStats(start_date, end_date, strategy_names)
# 繪製策略報酬率近期報酬率長條圖
report.plot_strategy_last_return().show()
# 繪製策略累積報酬率時間序列
report.plot_strategy_creturn().show()
