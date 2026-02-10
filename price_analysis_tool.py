"""
價格分析工具

此工具用於分析股票的原始收盤價和還原收盤價之間的差異，
特別是針對判斷創新高時可能產生的不同結果進行分析。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from finlab import data
import seaborn as sns
from typing import List, Union, Tuple, Dict, Optional


def analyze_price_differences(stock_id: str, start_date: str, days: int = 30, window: int = 120) -> None:
    """
    分析原始收盤價和還原收盤價的差異，以及對創新高判斷的影響

    參數:
    - stock_id: 股票代碼
    - start_date: 起始日期 (YYYY-MM-DD)
    - days: 分析天數
    - window: 創新高的窗口期(天數)
    """
    print(f"🔍 分析 {stock_id} 從 {start_date} 開始的 {days} 天數據")
    print(f"使用 {window} 天作為創新高窗口期")

    try:
        # 載入價格資料
        adj_close = data.get('etl:adj_close')
        close = data.get('price:收盤價')
    except Exception as e:
        print(f"❌ 載入價格數據失敗: {e}")
        return

    # 檢查股票是否在數據中
    if stock_id not in adj_close.columns or stock_id not in close.columns:
        print(f"❌ 股票 {stock_id} 不在價格數據中")
        return

    # 轉換日期格式並篩選日期範圍
    start_date = pd.to_datetime(start_date)
    all_dates = adj_close.index

    # 找到最接近的日期
    valid_dates = all_dates[all_dates >= start_date]
    if len(valid_dates) == 0:
        print(f"❌ 指定的起始日期 {start_date} 超出數據範圍")
        return

    closest_date = valid_dates[0]
    if closest_date != start_date:
        print(
            f"⚠️ 指定日期 {start_date.strftime('%Y-%m-%d')} 不是交易日，使用最近的交易日 {closest_date.strftime('%Y-%m-%d')}"
        )

    start_idx = all_dates.get_loc(closest_date)
    end_idx = min(start_idx + days, len(all_dates))
    date_range = all_dates[start_idx:end_idx]

    # 截取指定日期範圍的價格數據
    stock_adj_close = adj_close[stock_id].loc[date_range]
    stock_close = close[stock_id].loc[date_range]

    # 計算移動窗口最高價
    adj_high = adj_close[stock_id].rolling(window=window).max()
    close_high = close[stock_id].rolling(window=window).max()

    # 取得分析期間內的移動窗口高點
    adj_high_period = adj_high.loc[date_range]
    close_high_period = close_high.loc[date_range]

    # 判斷創新高
    adj_new_highs = stock_adj_close >= adj_high_period
    close_new_highs = stock_close >= close_high_period

    # 找出兩種價格判斷結果不同的日期
    diff_days = adj_new_highs != close_new_highs
    diff_dates = date_range[diff_days]

    # 繪製對比圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # 子圖1: 還原收盤價
    ax1.plot(date_range, stock_adj_close, label='Adjusted Close', color='blue', linewidth=2)
    ax1.plot(date_range, adj_high_period, label=f'{window}-Day High (Adj)', color='red', linestyle='--')

    # 標記還原價創新高的點
    if adj_new_highs.sum() > 0:
        ax1.scatter(
            date_range[adj_new_highs],
            stock_adj_close[adj_new_highs],
            color='orange',
            s=80,
            zorder=5,
            label=f'New Highs ({adj_new_highs.sum()} points)',
        )

    # 子圖2: 原始收盤價
    ax2.plot(date_range, stock_close, label='Original Close', color='green', linewidth=2)
    ax2.plot(date_range, close_high_period, label=f'{window}-Day High (Orig)', color='purple', linestyle='--')

    # 標記原始價創新高的點
    if close_new_highs.sum() > 0:
        ax2.scatter(
            date_range[close_new_highs],
            stock_close[close_new_highs],
            color='lime',
            s=80,
            zorder=5,
            label=f'New Highs ({close_new_highs.sum()} points)',
        )

    # 標記差異日期
    if len(diff_dates) > 0:
        for date in diff_dates:
            ax1.axvline(x=date, color='black', linestyle=':', alpha=0.5)
            ax2.axvline(x=date, color='black', linestyle=':', alpha=0.5)

        # 添加淺黃色背景標示差異區域
        for date in diff_dates:
            ax1.axvspan(date - pd.Timedelta(days=0.5), date + pd.Timedelta(days=0.5), color='yellow', alpha=0.2)
            ax2.axvspan(date - pd.Timedelta(days=0.5), date + pd.Timedelta(days=0.5), color='yellow', alpha=0.2)

    # 設置子圖標題和軸標籤
    ax1.set_title(f'{stock_id} - Adjusted Close Analysis', fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax2.set_title(f'{stock_id} - Original Close Analysis', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # 設定x軸刻度
    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.setp(ax2.get_xticklabels(), rotation=45)

    # 設定圖表總標題
    plt.suptitle(
        f'{stock_id} Price Comparison Analysis - Original vs Adjusted\n'
        f'Period: {date_range[0].strftime("%Y-%m-%d")} to {date_range[-1].strftime("%Y-%m-%d")}',
        fontsize=16,
        fontweight='bold',
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, top=0.9)
    plt.show()

    # 顯示統計資訊
    print("\n📊 創新高統計:")
    print(f"- 分析天數: {len(date_range)} 天")
    print(f"- 還原收盤價創新高天數: {adj_new_highs.sum()} 天")
    print(f"- 原始收盤價創新高天數: {close_new_highs.sum()} 天")
    print(f"- 判斷不同的天數: {diff_days.sum()} 天")

    # 分析判斷不同的原因
    if diff_days.sum() > 0:
        print("\n🔍 詳細分析判斷不同的日期:")
        analyze_diff_reasons(stock_id, diff_dates, stock_adj_close, stock_close, adj_high_period, close_high_period)

        # 繪製價格調整因子
        plot_adjustment_factor(stock_id, date_range, stock_adj_close, stock_close)


def analyze_diff_reasons(
    stock_id: str,
    diff_dates: pd.DatetimeIndex,
    adj_close: pd.Series,
    close: pd.Series,
    adj_high: pd.Series,
    close_high: pd.Series,
) -> None:
    """
    分析創新高判斷不同的原因
    """
    # 計算調整因子
    adjust_factor = adj_close / close

    # 分析每個差異日期
    for date in diff_dates:
        date_str = date.strftime('%Y-%m-%d')
        adj_price = adj_close.loc[date]
        close_price = close.loc[date]
        adj_high_val = adj_high.loc[date]
        close_high_val = close_high.loc[date]
        factor = adjust_factor.loc[date]

        # 判斷哪種價格創新高
        adj_is_high = adj_price >= adj_high_val
        close_is_high = close_price >= close_high_val

        # 計算與高點的百分比差距
        adj_high_diff = (adj_price / adj_high_val - 1) * 100
        close_high_diff = (close_price / close_high_val - 1) * 100

        # 輸出分析結果
        print(f"\n日期: {date_str}")
        print(
            f"  還原收盤價: {adj_price:.2f}, 120天高點: {adj_high_val:.2f}, 與高點差距: {adj_high_diff:+.2f}%, 創新高: {'是' if adj_is_high else '否'}"
        )
        print(
            f"  原始收盤價: {close_price:.2f}, 120天高點: {close_high_val:.2f}, 與高點差距: {close_high_diff:+.2f}%, 創新高: {'是' if close_is_high else '否'}"
        )
        print(f"  調整因子: {factor:.4f} (還原收盤價/原始收盤價)")

        # 分析可能原因
        if adj_is_high and not close_is_high:
            # 還原價創新高，原始價沒有
            if factor < 1:
                print("  📌 可能原因: 還原收盤價因除息調整後變低，使其相對於歷史高點更容易突破")
            else:
                print("  📌 可能原因: 還原收盤價高於原始收盤價，可能是反向股票分割或其他調整所致")
        elif close_is_high and not adj_is_high:
            # 原始價創新高，還原價沒有
            if factor < 1:
                print("  📌 可能原因: 原始收盤價沒有經過調整，而還原價的歷史高點較高")
            else:
                print("  📌 可能原因: 原始收盤價突破高點，但還原價的歷史高點因調整而更高")
        else:
            print("  📌 錯誤: 預期兩種價格判斷結果不同")


def plot_adjustment_factor(stock_id: str, date_range: pd.DatetimeIndex, adj_close: pd.Series, close: pd.Series) -> None:
    """
    繪製調整因子圖表
    """
    # 計算分析期間的調整因子
    adjust_factor = adj_close / close

    # 檢查是否有明顯的調整
    factor_change = adjust_factor.pct_change().abs()
    significant_changes = factor_change[factor_change > 0.01]  # 變化超過1%視為顯著

    plt.figure(figsize=(12, 6))
    plt.plot(date_range, adjust_factor, color='blue', linewidth=2)

    # 標記顯著變化點
    if len(significant_changes) > 0:
        change_dates = significant_changes.index.intersection(date_range)
        if len(change_dates) > 0:
            plt.scatter(change_dates, adjust_factor.loc[change_dates], color='red', s=100, zorder=5)

            # 添加垂直線標示變化點
            for date in change_dates:
                plt.axvline(x=date, color='red', linestyle='--', alpha=0.5)

                # 添加標籤
                idx = date_range.get_loc(date)
                if idx > 0:
                    prev_date = date_range[idx - 1]
                    prev_factor = adjust_factor.loc[prev_date]
                    curr_factor = adjust_factor.loc[date]
                    change_pct = (curr_factor / prev_factor - 1) * 100

                    plt.annotate(
                        f"{change_pct:+.2f}%",
                        xy=(date, curr_factor),
                        xytext=(10, 15),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", color='black'),
                    )

    # 設定圖表格式
    plt.title(f'{stock_id} - Price Adjustment Factor (Adj Close / Original Close)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Adjustment Factor', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 格式化 x 軸日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # 顯示調整因子統計
    print("\n📊 調整因子統計:")
    print(f"  平均值: {adjust_factor.mean():.4f}")
    print(f"  最大值: {adjust_factor.max():.4f}")
    print(f"  最小值: {adjust_factor.min():.4f}")

    if len(significant_changes) > 0:
        change_dates = significant_changes.index.intersection(date_range)
        if len(change_dates) > 0:
            print("\n  顯著變化點:")
            for date in change_dates:
                idx = date_range.get_loc(date)
                if idx > 0:
                    prev_date = date_range[idx - 1]
                    prev_factor = adjust_factor.loc[prev_date]
                    curr_factor = adjust_factor.loc[date]
                    change_pct = (curr_factor / prev_factor - 1) * 100

                    print(f"  {date.strftime('%Y-%m-%d')}: {prev_factor:.4f} → {curr_factor:.4f} ({change_pct:+.2f}%)")

            # 猜測可能的除權息事件
            if any(adjust_factor.loc[change_dates] < adjust_factor.shift().loc[change_dates]):
                print("\n  💡 分析結果顯示可能有除權息事件，導致還原收盤價下調")


def find_divergence_stocks(start_date: str, days: int = 30, window: int = 120, top_n: int = 10) -> List[Dict]:
    """
    尋找市場中判斷創新高結果差異最大的股票

    參數:
    - start_date: 起始日期 (YYYY-MM-DD)
    - days: 分析天數
    - window: 創新高的窗口期(天數)
    - top_n: 顯示前N名差異最大的股票

    返回:
    - 包含差異股票資訊的列表
    """
    print(f"🔍 尋找判斷創新高結果差異最大的股票...")

    try:
        # 載入價格資料
        adj_close = data.get('etl:adj_close')
        close = data.get('etl:close')
    except Exception as e:
        print(f"❌ 載入價格數據失敗: {e}")
        return []

    # 處理日期範圍
    start_date = pd.to_datetime(start_date)
    all_dates = adj_close.index

    # 找到最接近的日期
    valid_dates = all_dates[all_dates >= start_date]
    if len(valid_dates) == 0:
        print(f"❌ 指定的起始日期 {start_date} 超出數據範圍")
        return []

    closest_date = valid_dates[0]
    start_idx = all_dates.get_loc(closest_date)
    end_idx = min(start_idx + days, len(all_dates))
    date_range = all_dates[start_idx:end_idx]

    # 找出兩個DataFrame中共有的股票
    common_stocks = sorted(list(set(adj_close.columns) & set(close.columns)))
    print(f"  共有 {len(common_stocks)} 支股票可供分析")

    # 儲存結果
    results = []

    # 分析每支股票
    for i, stock in enumerate(common_stocks):
        if (i + 1) % 100 == 0:
            print(f"  已分析 {i + 1}/{len(common_stocks)} 支股票...")

        try:
            # 取得價格數據
            stock_adj_close = adj_close[stock].loc[date_range]
            stock_close = close[stock].loc[date_range]

            # 如果數據中有NaN則跳過
            if stock_adj_close.isna().any() or stock_close.isna().any():
                continue

            # 計算移動窗口最高價
            adj_high = adj_close[stock].rolling(window=window).max().loc[date_range]
            close_high = close[stock].rolling(window=window).max().loc[date_range]

            # 判斷創新高
            adj_new_highs = stock_adj_close >= adj_high
            close_new_highs = stock_close >= close_high

            # 計算差異天數
            diff_days = adj_new_highs != close_new_highs
            diff_count = diff_days.sum()

            # 如果有差異，則記錄
            if diff_count > 0:
                # 計算調整因子
                adjust_factor = stock_adj_close / stock_close

                # 判斷價格變化
                factor_change = adjust_factor.pct_change().abs()
                max_change = factor_change.max()

                results.append(
                    {
                        'stock_id': stock,
                        'diff_count': diff_count,
                        'diff_ratio': diff_count / len(date_range),
                        'adj_highs': adj_new_highs.sum(),
                        'close_highs': close_new_highs.sum(),
                        'factor_mean': adjust_factor.mean(),
                        'factor_max_change': max_change,
                    }
                )
        except Exception as e:
            print(f"  分析 {stock} 時發生錯誤: {e}")

    # 按差異天數排序
    results.sort(key=lambda x: x['diff_count'], reverse=True)

    # 顯示結果
    print("\n📊 創新高判斷差異最大的股票:")
    print(
        f"{'股票':^8} {'差異天數':^10} {'差異比例':^10} {'還原高點':^10} {'原始高點':^10} {'調整因子':^10} {'最大變化':^10}"
    )
    print("-" * 70)

    for i, res in enumerate(results[:top_n]):
        print(
            f"{res['stock_id']:^8} {res['diff_count']:^10} {res['diff_ratio'] * 100:^10.2f}% "
            f"{res['adj_highs']:^10} {res['close_highs']:^10} {res['factor_mean']:^10.4f} "
            f"{res['factor_max_change'] * 100:^10.2f}%"
        )

    return results[:top_n]


def analyze_price_adjustment_history(stock_id: str, years: int = 3) -> None:
    """
    分析特定股票的長期價格調整歷史

    參數:
    - stock_id: 股票代碼
    - years: 分析的年數
    """
    print(f"🔍 分析 {stock_id} 最近 {years} 年的價格調整歷史")

    try:
        # 載入價格資料
        adj_close = data.get('etl:adj_close')
        close = data.get('etl:close')
    except Exception as e:
        print(f"❌ 載入價格數據失敗: {e}")
        return

    # 檢查股票是否存在
    if stock_id not in adj_close.columns or stock_id not in close.columns:
        print(f"❌ 股票 {stock_id} 不在價格數據中")
        return

    # 計算分析日期範圍
    end_date = adj_close.index[-1]
    start_date = end_date - pd.DateOffset(years=years)

    # 篩選日期範圍
    mask = (adj_close.index >= start_date) & (adj_close.index <= end_date)
    date_range = adj_close.index[mask]

    # 取得價格數據
    stock_adj_close = adj_close[stock_id].loc[date_range]
    stock_close = close[stock_id].loc[date_range]

    # 計算調整因子
    adjust_factor = stock_adj_close / stock_close

    # 分析調整因子的變化
    factor_change = adjust_factor.pct_change()
    significant_changes = factor_change[abs(factor_change) > 0.01]  # 變化超過1%

    # 繪製價格和調整因子
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # 子圖1: 價格走勢
    ax1.plot(date_range, stock_adj_close, label='Adjusted Close', color='blue', linewidth=1.5)
    ax1.plot(date_range, stock_close, label='Original Close', color='green', linewidth=1.5)

    # 標記價格顯著差異的點
    if len(significant_changes) > 0:
        for date in significant_changes.index:
            if date in date_range:
                ax1.axvline(x=date, color='red', linestyle=':', alpha=0.5)

    # 子圖2: 調整因子
    ax2.plot(date_range, adjust_factor, color='purple', linewidth=1.5)

    # 標記調整因子變化點
    if len(significant_changes) > 0:
        for date in significant_changes.index:
            if date in date_range:
                ax2.axvline(x=date, color='red', linestyle=':', alpha=0.5)

                # 添加變化標記
                idx = date_range.get_loc(date)
                if idx > 0:
                    prev_date = date_range[idx - 1]
                    prev_factor = adjust_factor.loc[prev_date]
                    curr_factor = adjust_factor.loc[date]
                    change_pct = (curr_factor / prev_factor - 1) * 100

                    # 只標記變化超過2%的點
                    if abs(change_pct) > 2:
                        ax2.annotate(
                            f"{change_pct:+.2f}%",
                            xy=(date, curr_factor),
                            xytext=(10, 10 if change_pct > 0 else -25),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", color='black'),
                        )

    # 設定子圖標題和軸標籤
    ax1.set_title(f'{stock_id} - Price Trend', fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f'{stock_id} - Adjustment Factor (Adj Close / Original Close)', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Adjustment Factor', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 設定整體標題
    plt.suptitle(
        f'{stock_id} - {years}-Year Price Adjustment History Analysis\n'
        f'Period: {date_range[0].strftime("%Y-%m-%d")} to {date_range[-1].strftime("%Y-%m-%d")}',
        fontsize=16,
        fontweight='bold',
    )

    # 格式化x軸
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, top=0.9)
    plt.show()

    # 顯示調整事件
    if len(significant_changes) > 0:
        print("\n📅 價格調整事件:")
        print(f"{'日期':^12} {'調整前因子':^12} {'調整後因子':^12} {'變化百分比':^12} {'推測事件':^20}")
        print("-" * 70)

        for date in sorted(significant_changes.index):
            if date in date_range:
                idx = date_range.get_loc(date)
                if idx > 0:
                    prev_date = date_range[idx - 1]
                    prev_factor = adjust_factor.loc[prev_date]
                    curr_factor = adjust_factor.loc[date]
                    change_pct = (curr_factor / prev_factor - 1) * 100

                    # 猜測可能的事件類型
                    event_type = "除息" if change_pct < 0 else "除權或其他調整"

                    print(
                        f"{date.strftime('%Y-%m-%d'):^12} {prev_factor:^12.4f} {curr_factor:^12.4f} "
                        f"{change_pct:^+12.2f}% {event_type:^20}"
                    )


def deep_analyze_adjustment_reason(stock_id: str, target_date: str, years_back: int = 2) -> None:
    """
    深度分析調整因子變化的具體原因

    參數:
    - stock_id: 股票代碼
    - target_date: 目標分析日期 (YYYY-MM-DD)
    - years_back: 往前分析的年數
    """
    print(f"🔍 深度分析 {stock_id} 在 {target_date} 前後的價格調整原因")

    try:
        # 載入價格資料
        adj_close = data.get('etl:adj_close')
        close = data.get('price:收盤價')
    except Exception as e:
        print(f"❌ 載入價格數據失敗: {e}")
        return

    # 檢查股票是否存在
    if stock_id not in adj_close.columns or stock_id not in close.columns:
        print(f"❌ 股票 {stock_id} 不在價格數據中")
        return

    # 設定分析日期範圍
    target_date = pd.to_datetime(target_date)
    start_date = target_date - pd.DateOffset(years=years_back)

    # 篩選日期範圍
    mask = (adj_close.index >= start_date) & (adj_close.index <= target_date)
    date_range = adj_close.index[mask]

    # 取得價格數據
    stock_adj_close = adj_close[stock_id].loc[date_range]
    stock_close = close[stock_id].loc[date_range]

    # 計算調整因子
    adjust_factor = stock_adj_close / stock_close

    # 計算調整因子的變化率
    factor_change = adjust_factor.pct_change()
    daily_change = adjust_factor.diff()

    # 找出所有顯著變化點（變化超過5%）
    significant_changes = factor_change[abs(factor_change) > 0.05]

    print(f"\n📊 調整因子分析 (期間: {start_date.strftime('%Y-%m-%d')} 到 {target_date.strftime('%Y-%m-%d')}):")
    print(f"  目標日期調整因子: {adjust_factor.loc[target_date]:.4f}")
    print(f"  最早調整因子: {adjust_factor.iloc[0]:.4f}")
    print(f"  因子變化倍數: {adjust_factor.loc[target_date] / adjust_factor.iloc[0]:.4f}")

    if len(significant_changes) > 0:
        print(f"\n📅 發現 {len(significant_changes)} 個重大調整事件:")
        print(f"{'日期':^12} {'調整前':^10} {'調整後':^10} {'變化%':^10} {'累積因子':^10} {'可能事件':^15}")
        print("-" * 80)

        cumulative_factor = 1.0
        for date in sorted(significant_changes.index):
            if date in date_range:
                idx = date_range.get_loc(date)
                if idx > 0:
                    prev_date = date_range[idx - 1]
                    prev_factor = adjust_factor.loc[prev_date]
                    curr_factor = adjust_factor.loc[date]
                    change_pct = (curr_factor / prev_factor - 1) * 100

                    # 累積調整因子變化
                    cumulative_factor *= curr_factor / prev_factor

                    # 判斷事件類型
                    if change_pct > 50:
                        event_type = "股票分割"
                    elif change_pct < -20:
                        event_type = "除息/除權"
                    elif change_pct > 10:
                        event_type = "小額分割"
                    else:
                        event_type = "調整"

                    print(
                        f"{date.strftime('%Y-%m-%d'):^12} {prev_factor:^10.4f} {curr_factor:^10.4f} "
                        f"{change_pct:^+10.2f} {cumulative_factor:^10.4f} {event_type:^15}"
                    )

        print(f"\n💡 關鍵發現:")
        print(f"  - 累積調整倍數: {cumulative_factor:.4f}")
        print(f"  - 這表示從最早到現在，還原收盤價相對於原始收盤價被調整了 {cumulative_factor:.2f} 倍")

        if cumulative_factor > 2:
            print(f"  - 這種大幅調整通常是由於股票分割造成的")
            print(f"  - 例如: 1股分割成3-4股的大額股票分割")
        elif cumulative_factor < 0.8:
            print(f"  - 這種調整通常是由於大額除息造成的")

    else:
        print("\n📊 在分析期間內未發現重大調整事件")
        print("   調整因子保持相對穩定")

    # 繪製詳細的調整因子變化圖
    plt.figure(figsize=(15, 10))

    # 子圖1: 調整因子變化
    plt.subplot(3, 1, 1)
    plt.plot(date_range, adjust_factor, color='blue', linewidth=2)
    plt.title(f'{stock_id} - Adjustment Factor History', fontsize=14, fontweight='bold')
    plt.ylabel('Adjustment Factor', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 標記重大變化點
    if len(significant_changes) > 0:
        for date in significant_changes.index:
            if date in date_range:
                plt.axvline(x=date, color='red', linestyle='--', alpha=0.7)
                plt.annotate(
                    f"{date.strftime('%m/%d')}",
                    xy=(date, adjust_factor.loc[date]),
                    xytext=(0, 20),
                    textcoords='offset points',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                )

    # 子圖2: 還原收盤價
    plt.subplot(3, 1, 2)
    plt.plot(date_range, stock_adj_close, color='blue', linewidth=2, label='Adjusted Close')
    plt.title(f'{stock_id} - Adjusted Close Price', fontsize=14)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子圖3: 原始收盤價
    plt.subplot(3, 1, 3)
    plt.plot(date_range, stock_close, color='green', linewidth=2, label='Original Close')
    plt.title(f'{stock_id} - Original Close Price', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 格式化x軸
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # 解釋創新高差異的具體原因
    print(f"\n🎯 創新高差異的具體解釋:")
    current_factor = adjust_factor.loc[target_date]
    print(f"  1. 在 {target_date.strftime('%Y-%m-%d')}，調整因子為 {current_factor:.4f}")
    print(f"  2. 這意味著還原收盤價是原始收盤價的 {current_factor:.2f} 倍")

    if current_factor > 3:
        print(f"  3. 這種大幅調整通常源於股票分割:")
        print(f"     - 例如：1股分割成3-4股")
        print(f"     - 分割後原始收盤價變低，但還原收盤價保持歷史連續性")
        print(f"     - 因此還原收盤價的120天高點也被相應調整")
        print(f"  4. 創新高判斷差異的原因:")
        print(f"     - 還原收盤價: 已考慮分割調整，所以更容易達到「調整後」的高點")
        print(f"     - 原始收盤價: 未考慮分割，需要達到「分割前」的高點才算創新高")


def main():
    """主程式入口"""
    print("=" * 50)
    print("Price Analysis Tool - Analyze Close Price New High Differences")
    print("=" * 50)

    # 顯示主選單
    print("\nPlease select function:")
    print("1. Analyze single stock price differences")
    print("2. Find stocks with highest new high judgment differences")
    print("3. Analyze long-term price adjustment history")
    print("4. Deep analyze adjustment reason for specific date")
    print("0. Exit")

    choice = input("\nPlease enter option (0-4): ")

    if choice == '1':
        stock_id = input("Please enter stock ID: ")
        start_date = input("Please enter start date (YYYY-MM-DD): ")
        days = int(input("Please enter analysis days (default 30): ") or "30")
        window = int(input("Please enter new high window days (default 120): ") or "120")

        analyze_price_differences(stock_id, start_date, days, window)

    elif choice == '2':
        start_date = input("Please enter start date (YYYY-MM-DD): ")
        days = int(input("Please enter analysis days (default 30): ") or "30")
        window = int(input("Please enter new high window days (default 120): ") or "120")
        top_n = int(input("Please enter top N stocks to display (default 10): ") or "10")

        results = find_divergence_stocks(start_date, days, window, top_n)

        # 詢問是否要深入分析某支股票
        if results:
            print("\nWould you like to analyze one stock in detail?")
            stock_id = input("Please enter stock ID (or press Enter to skip): ")

            if stock_id:
                analyze_price_differences(stock_id, start_date, days, window)

    elif choice == '3':
        stock_id = input("Please enter stock ID: ")
        years = int(input("Please enter analysis years (default 3): ") or "3")

        analyze_price_adjustment_history(stock_id, years)

    elif choice == '4':
        stock_id = input("Please enter stock ID: ")
        target_date = input("Please enter target date (YYYY-MM-DD): ")
        years_back = int(input("Please enter years to look back (default 2): ") or "2")

        deep_analyze_adjustment_reason(stock_id, target_date, years_back)

    elif choice == '0':
        print("Thank you for using, goodbye!")

    else:
        print("❌ Invalid option, please rerun the program and select a valid option")


if __name__ == "__main__":
    main()
