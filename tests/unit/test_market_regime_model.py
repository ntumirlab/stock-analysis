"""多空轉折模型：市值前 150 檔成分、審核時程、現貨檔數差（純 pandas，不需 finlab／ta-lib）。

指標計算與總分組合需要 ta-lib 與 strategy_class，放在 tests/integration/test_market_regime_components.py。
"""

import numpy as np
import pandas as pd

from alan_dashboard.market_regime_model import (
    breadth_score, listed_common_stocks, ma_direction_breadth, review_schedule, top_n_membership,
)


def test_listed_common_stocks_filters_market_and_ids():
    cat = pd.DataFrame({
        'stock_id': ['2330', '0050', '2881A', '910861', '9103', '9151', '7610', '6488', '00631L'],
        'name':     ['台積電', '元大台灣50', '富邦特', '神隆', '美德醫療-DR', '旺旺', '聯友金屬-創', '環球晶',
                     '元大台灣50正2'],
        'category': ['半導體業', 'domestic_etf', '金融保險業', '存託憑證', '存託憑證', '存託憑證', '其他',
                     '半導體業', 'domestic_etf'],
        'market':   ['sii', 'etf', 'sii', 'sii', 'sii', 'sii', 'sii', 'otc', 'etf'],
    })
    # 排除 ETF、特別股、存託憑證（含 4 碼與名稱沒有 -DR 的舊 TDR）、創新板、上櫃
    assert listed_common_stocks(cat) == ['2330']


def test_review_schedule_cutoff_and_effective_dates():
    days = pd.bdate_range('2025-01-01', '2025-12-31')  # 不含假日，僅驗證時程規則
    sched = [(c.strftime('%m-%d'), e.strftime('%m-%d')) for c, e in review_schedule(days)]
    # 3/6/9/12 月第三個星期五（3/21、6/20、9/19、12/19）後的下一個交易日生效；
    # 資料截止 = 生效星期一往前四週（2/24、5/26、8/25、11/24）
    assert sched == [('02-24', '03-24'), ('05-26', '06-23'), ('08-25', '09-22'), ('11-24', '12-22')]

    # 資料只到 9/17：9 月的審核 9/22 才生效，不列入；資料太短、連截止日都沒有時為空
    assert review_schedule(pd.bdate_range('2025-01-01', '2025-09-17'))[-1][1].strftime('%m-%d') == '06-23'
    assert review_schedule(pd.bdate_range('2025-09-01', '2025-09-30')) == []

    # 第三個星期五後的星期一放假：生效日順延到下一個交易日，截止日仍以名義星期一往前四週計
    days = pd.bdate_range('2025-05-01', '2025-07-31').drop(pd.Timestamp('2025-06-23'))
    assert [(c.strftime('%m-%d'), e.strftime('%m-%d')) for c, e in review_schedule(days)] == [('05-26', '06-24')]


def test_top_n_membership_follows_review_schedule_and_buffer_rules():
    days = pd.bdate_range('2025-05-01', '2025-12-31')
    mv = pd.DataFrame(index=days, columns=list('ABCD'), dtype=float)
    mv[:] = [[4, 3, 2, 1]] * len(days)          # 5/26 截止：A > B > C > D → 首次取前 2：{A, B}
    mv.loc['2025-08-01':] = [4, 2, 3, 1]        # 8/25 截止：A > C > B > D
    mv.loc['2025-11-01':] = [3, 2, 4, 1]        # 11/24 截止：C > A > B > D
    # n=2、納入門檻第 1 名、剔除門檻第 4 名
    member = top_n_membership(mv, list('ABCDZ'), n=2, entry_rank=1, exit_rank=4)

    assert not member.loc[:'2025-06-20'].any().any()                       # 首次生效日前無名單
    assert member.loc['2025-06-23'].tolist() == [True, True, False, False]  # 5/26 排名，6/23 生效
    # 9/22：C 第 2 名未達納入門檻、B 第 3 名未跌破剔除門檻 → 名單不變（緩衝）
    assert member.loc['2025-09-22'].tolist() == [True, True, False, False]
    # 12/22：C 升到第 1 名納入；A、B 未跌破第 4 名都保留 → 超出 2 檔，剔除排名最低的 B
    assert member.loc['2025-12-22'].tolist() == [True, False, True, False]
    assert member.loc['2025-12-19'].tolist() == [True, True, False, False]  # 生效前一交易日仍是舊名單
    assert list(member.columns) == list('ABCD')  # 不在市值資料裡的代號被忽略


def test_top_n_membership_refills_when_member_drops_below_exit_rank():
    days = pd.bdate_range('2025-05-01', '2025-09-30')
    mv = pd.DataFrame(index=days, columns=list('ABCD'), dtype=float)
    mv[:] = [[4, 3, 2, 1]] * len(days)
    mv.loc['2025-08-01':] = [4, 1, 3, 2]        # 8/25 截止：A > C > D > B，B 跌到第 4 名
    member = top_n_membership(mv, list('ABCD'), n=2, entry_rank=1, exit_rank=4)
    # B 剔除；C 第 2 名雖未達納入門檻，但為維持 2 檔由排名最高的非成分股遞補
    assert member.loc['2025-09-22'].tolist() == [True, False, True, False]


def test_ma_direction_breadth_counts_members_only():
    dates = pd.bdate_range('2025-01-01', periods=30)
    n = len(dates)
    close = pd.DataFrame({
        'UP':   np.arange(1, n + 1, dtype=float),          # 三條均線持續向上
        'DOWN': np.arange(n, 0, -1, dtype=float),          # 三條均線持續向下
        'FLAT': np.full(n, 10.0),                          # 均線持平：不計
        'OUT':  np.arange(1, n + 1, dtype=float),          # 向上但不是成分股
    }, index=dates)
    member = pd.DataFrame(True, index=dates, columns=['UP', 'DOWN', 'FLAT'])
    member['OUT'] = False
    diff = ma_direction_breadth(close, member)

    assert diff.iloc[-1] == 0            # UP(+1) 與 DOWN(−1) 抵銷，FLAT 與 OUT 不計
    member['DOWN'] = False
    assert ma_direction_breadth(close, member).iloc[-1] == 1
    assert ma_direction_breadth(close, member).iloc[:20].eq(0).all()  # MA20 暖身期間無值
    # 成分名單資料日比收盤價早（市值晚更新）：最後幾天沿用最近一次名單
    assert ma_direction_breadth(close, member.iloc[:-3]).iloc[-1] == 1
    # 當日個股收盤價尚未更新（全為 NaN）或只更新一部分（成分股 UP、FLAT 只剩一檔有價）：無值，而不是 0
    close.loc[dates[-1], 'UP'] = np.nan
    assert np.isnan(ma_direction_breadth(close, member).iloc[-1])
    close.iloc[-1] = np.nan
    assert np.isnan(ma_direction_breadth(close, member).iloc[-1])


def test_ma_direction_breadth_ignores_floating_point_noise():
    dates = pd.bdate_range('2025-01-01', periods=30)
    close = pd.DataFrame({'FLAT': np.full(len(dates), 100.0)}, index=dates)
    close.iloc[-1, 0] = 100.0 + 1e-11  # 均線變動 1e-12：浮點雜訊等級，應視為持平
    member = pd.DataFrame(True, index=dates, columns=['FLAT'])
    assert ma_direction_breadth(close, member).iloc[-1] == 0
    close.iloc[-1, 0] = 100.01  # 真的漲一檔：三條均線都向上
    assert ma_direction_breadth(close, member).iloc[-1] == 1


def test_breadth_score_threshold():
    diff = pd.Series([26, 25, 0, -25, -26])
    assert breadth_score(diff).tolist() == [1, 0, 0, 0, -1]
