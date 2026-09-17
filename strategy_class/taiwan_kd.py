"""
台灣標準 KD 指標計算模組
完全匹配台灣看盤軟體（XQ、Yahoo Finance、券商軟體）的 KD 計算
"""

import pandas as pd
import numpy as np

def taiwan_kd_fast(high_df, low_df, close_df, fastk_period=9, alpha=1/3, verbose=True):
    """
    快速版台灣標準KD計算 - 完全向量化

    verbose=False 時不 print 進度（儀表板等非互動環境使用）
    """
    if verbose:
        print(f"⚡ 開始計算台灣標準KD指標 ({len(close_df.columns)} 檔股票)...")
    
    import time
    start_time = time.time()
    
    # 1. 向量化計算 RSV
    rolling_high = high_df.rolling(window=fastk_period, min_periods=fastk_period).max()
    rolling_low = low_df.rolling(window=fastk_period, min_periods=fastk_period).min()
    
    # 避免除零錯誤
    denominator = rolling_high - rolling_low
    # 當最高價=最低價時，設定RSV為50（中性值）
    rsv = np.where(
        denominator != 0,
        ((close_df - rolling_low) / denominator) * 100,
        50
    )
    rsv = pd.DataFrame(rsv, index=close_df.index, columns=close_df.columns)
    
    # 2. 計算 K 值 (EMA 平滑 RSV)
    k_df = rsv.ewm(alpha=alpha, adjust=False).mean()
    
    # 3. 計算 D 值 (EMA 平滑 K 值)
    d_df = k_df.ewm(alpha=alpha, adjust=False).mean()
    
    calc_time = time.time() - start_time
    if verbose:
        print(f"✅ KD計算完成！耗時: {calc_time:.2f} 秒")
        print(f"📊 K值有效數據: {k_df.count().sum()} 個")
        print(f"📊 D值有效數據: {d_df.count().sum()} 個")
    
    return k_df, d_df