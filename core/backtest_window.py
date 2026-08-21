"""回測視窗起點的對齊運算（純運算，不 import finlab，CI 可測）。

`lookback_months` 原本是直接拿日期切 position。切點落在持倉中間時，finlab 會把
當時進行中的部位當成一筆新進場，產出策略根本不存在的短天期交易 —— 實測 2026-08-13
那次切在週三，生出一筆「隔天買、再隔天賣」的一日單、七檔平均 -3.02%，整段三個月的
年化因此從 +11.2% 變成 -0.86%。同一份資料在 08-15 與 08-16 兩次執行也因此差了
20 個百分點（+15.17% vs +35.97%）。

修法是把切點退到「空手的交易日」。**兩個條件缺一不可**：
- 空手：切在持倉中間就會製造上述假交易。
- 交易日：position 經過 `.shift(-1)`，訊號日被訂在前一個交易日。視窗若從週六起跑，
  frame 裡第一個交易日是週一、而週一的值已經是 True，finlab 就把週一當成訊號日、
  隔天才進場（實測首筆會變成 period 3）。週五休市時同理，要再往前退到週四。
"""

import pandas as pd


def flat_trading_days(position, trading_days=None) -> pd.DatetimeIndex:
    """position 中「當日完全空手」且「屬於交易日」的日期。

    trading_days 傳 None 時不做交易日過濾，結果會包含週末 —— 只在呼叫端拿不到
    交易日曆時當退路，一般情況都應該傳入。
    """
    if position is None or len(position.index) == 0:
        return pd.DatetimeIndex([])

    idx = pd.DatetimeIndex(position.index)[~position.any(axis=1).to_numpy()]
    if trading_days is not None:
        idx = idx[idx.isin(pd.DatetimeIndex(trading_days))]
    return idx


def snap_cutoff_to_flat_trading_day(position, cutoff, trading_days=None) -> pd.Timestamp:
    """把 cutoff 往回退到最近一個空手交易日。

    退不到（cutoff 之前整段都在持倉，或 position 沒有交易日）時回傳 position 起點，
    也就是不裁切 —— 寧可視窗長一點，也不要留下被截斷的假交易。
    """
    if position is None or len(position.index) == 0:
        return pd.Timestamp(cutoff)

    candidates = flat_trading_days(position, trading_days)
    candidates = candidates[candidates <= pd.Timestamp(cutoff)]
    if len(candidates) == 0:
        return pd.Timestamp(position.index[0])
    return pd.Timestamp(candidates[-1])
