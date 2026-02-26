import pandas as pd
import numpy as np
from numba import njit

# 初始化信號數據
buy_signal = pd.DataFrame([True, False, True, False, False, True, False, True], columns=['stock'])
sell_condition_1 = pd.DataFrame([False, False, False, True, False, True, True, False], columns=['stock'])
sell_condition_2 = pd.DataFrame([True, False, True, False, False, True, True, True], columns=['stock'])

# 將信號轉換為 NumPy 陣列
buy_signal_np = buy_signal.to_numpy()
sell_condition_1_np = sell_condition_1.to_numpy()
sell_condition_2_np = sell_condition_2.to_numpy()

# 初始化持倉表（轉換為 NumPy 格式），初始值設為 0.0
position_np = np.zeros_like(buy_signal_np, dtype=np.float64)


@njit
def process_positions(buy_signal, sell_condition_1, sell_condition_2, position):
    # 先處理買入訊號，將持倉設為 1.0
    for i in range(len(buy_signal)):
        if buy_signal[i, 0]:
            position[i, 0] = 1.0

    # 遍歷數據，處理持倉變化
    for day_idx in range(1, len(position)):
        if position[day_idx - 1, 0] == 0.0 and buy_signal[day_idx, 0]:  # 如果當天有買入訊號，設為 1.0
            position[day_idx, 0] = 1.0
        elif position[day_idx - 1, 0] == 1.0 and sell_condition_1[day_idx, 0]:  # 判斷賣出條件1
            position[day_idx, 0] = 0.5
        elif position[day_idx - 1, 0] == 0.5 and sell_condition_2[day_idx, 0]:  # 判斷賣出條件2
            position[day_idx, 0] = 0.0
        else:
            position[day_idx, 0] = position[day_idx - 1, 0]  # 保持前一天的持倉狀態

    return position


# 先將 buy_signal 為 True 的位置設為 1.0
position_np = process_positions(buy_signal_np, sell_condition_1_np, sell_condition_2_np, position_np)


# 將結果轉換回 pandas DataFrame
position = pd.DataFrame(position_np, index=buy_signal.index, columns=buy_signal.columns)

# 輸出結果
print(position)
