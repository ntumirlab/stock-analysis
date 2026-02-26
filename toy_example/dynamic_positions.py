# import pandas as pd

# buy_signal = pd.Series([True, False, True, False, False, True, False])
# sell_signal_1 = pd.Series([False, False, False, True, False, True, True])
# sell_signal_2 = pd.Series([True, False, True, False, False, True, True])

# # 初始化持倉表，初始值設為 0.0
# position = pd.Series(0.0, index=buy_signal.index)

# # 初始化部分賣出標誌
# partial_sell_triggered = False

# # 先處理買入訊號
# position[buy_signal] = 1.0

# # 遍歷數據，填充持倉訊號
# for i in range(1, len(position)):
#     # 先檢查 sell_signal_2，如果發生，且之前 sell_signal_1 已觸發，將持倉改為 0.0
#     if sell_signal_2[i] and partial_sell_triggered:
#         position[i] = 0.0
#     # 檢查 sell_signal_1，如果發生，將持倉改為 0.5 並設置部分賣出標誌
#     elif sell_signal_1[i]:
#         position[i] = 0.5
#         partial_sell_triggered = True
#     # 否則保持前一個持倉
#     else:
#         position[i] = position[i - 1]

# print(position)

from finlab import data, backtest
from finlab.online.order_executor import Position
import pandas as pd

with data.universe(market='TSE_OTC'):
    position = pd.DataFrame((data.get('price:收盤價') < 0), dtype='float')

position.loc['2024-10-27':'2024-10-30', '1101'] = 1
position.loc['2024-10-27':'2024-10-30', '1102'] = 0.5

report = backtest.sim(position, resample=None, upload=False, name="test")

position_today = Position.from_report(report, 1000, odd_lot=True)  # 零股
print(position_today)
