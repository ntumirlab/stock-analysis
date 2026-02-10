"""
BIAS 分析模組
用於分析乖離率與交易回報的關係
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple


class BiasAnalyzer:
    def __init__(self):
        pass

    def plot_bias_vs_return(self, bias: pd.DataFrame, trades: pd.DataFrame, bias_name: str) -> Dict[str, Any]:
        """
        繪製乖離率與回報的關係圖

        Parameters:
        -----------
        bias : pd.DataFrame
            乖離率數據
        trades : pd.DataFrame
            交易報告數據
        bias_name : str
            乖離率名稱（如 'Bias_5', 'Bias_10' 等）

        Returns:
        --------
        Dict[str, Any] : 包含分析結果的字典
        """
        bias_values = []
        trade_returns = []

        # 提取交易數據
        for date, stock_id, trade_return in zip(trades['entry_sig_date'], trades['stock_id'], trades['return']):
            stock_id = stock_id.split()[0]  # 提取股票代號
            if date in bias.index and stock_id in bias.columns:
                # 獲取該筆交易的 bias 和 return，並轉換為百分比
                bias_val = bias.loc[date, stock_id]
                return_val = trade_return * 100  # 回報轉換為百分比

                if not pd.isna(bias_val) and not pd.isna(return_val):
                    bias_values.append(bias_val)
                    trade_returns.append(return_val)

        # 將數據轉為 pandas Series
        bias_values = pd.Series(bias_values, name=bias_name)
        trade_returns = pd.Series(trade_returns, name="Return (%)")

        # 確認數據大小
        print(f"Number of {bias_name} values: {len(bias_values)}")
        print(f"Number of Return values: {len(trade_returns)}")

        if len(bias_values) == 0:
            print(f"警告: 沒有找到 {bias_name} 的有效數據")
            return {}

        # 繪製散點圖
        self._plot_scatter(bias_values, trade_returns, bias_name)

        # 繪製直方圖
        average_return_per_bin = self._plot_histogram(bias_values, trade_returns, bias_name)

        return {
            'bias_values': bias_values,
            'trade_returns': trade_returns,
            'average_return_per_bin': average_return_per_bin,
        }

    def _plot_scatter(self, bias_values: pd.Series, trade_returns: pd.Series, bias_name: str):
        """繪製散點圖"""
        plt.figure(figsize=(10, 6))
        plt.scatter(bias_values, trade_returns, alpha=0.7, color="blue", label=f"{bias_name} vs Return")
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8)  # 基準線
        plt.title(f"Scatter Plot of {bias_name} vs Return")
        plt.xlabel(bias_name)
        plt.ylabel("Return (%)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def _plot_histogram(self, bias_values: pd.Series, trade_returns: pd.Series, bias_name: str) -> pd.Series:
        """繪製直方圖並返回每個區間的平均回報"""
        # 動態確定區間
        min_bias = bias_values.min()
        max_bias = bias_values.max()

        # 先列印出極端高回報值，了解數據分佈
        high_returns = trade_returns[trade_returns > 50]  # 篩選超過50%的高回報
        if not high_returns.empty:
            print("\n高回報值統計:")
            print(f"高回報(>50%)的交易數量: {len(high_returns)}")
            print(f"最高回報值: {high_returns.max():.2f}%")
            print(f"高回報交易占總交易比例: {(len(high_returns) / len(trade_returns) * 100):.2f}%")

        # 創建適當的區間
        bins = np.arange(0, max_bias + 0.05, 0.05)  # 每5%一個區間

        bias_return_df = pd.DataFrame({bias_name: bias_values, "Return (%)": trade_returns})
        bias_return_df[f"{bias_name}_Bins"] = pd.cut(bias_return_df[bias_name], bins=bins)

        # 計算每個區間的平均回報和交易數量
        grouped = bias_return_df.groupby(f"{bias_name}_Bins")
        average_return_per_bin = grouped["Return (%)"].mean()
        counts_per_bin = grouped.size()

        # 繪製直方圖
        plt.figure(figsize=(12, 6))
        average_return_per_bin.plot(kind="bar", color="green", alpha=0.7)
        plt.title(f"Average Return per {bias_name} Interval")
        plt.xlabel(f"{bias_name} Interval")
        plt.ylabel("Average Return (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 輸出統計數據
        print(f"\nAverage Return per {bias_name} Interval:")
        print(average_return_per_bin.round(2))

        print(f"\nNumber of Trades per {bias_name} Interval:")
        print(counts_per_bin)

        # 添加一個圖表顯示每個區間的交易數量
        plt.figure(figsize=(12, 6))
        counts_per_bin.plot(kind="bar", color="blue", alpha=0.7)
        plt.title(f"Number of Trades per {bias_name} Interval")
        plt.xlabel(f"{bias_name} Interval")
        plt.ylabel("Number of Trades")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return average_return_per_bin

    def analyze_all_bias(self, bias_dict: Dict[str, pd.DataFrame], trades: pd.DataFrame) -> Dict[str, Any]:
        """
        分析所有乖離率指標

        Parameters:
        -----------
        bias_dict : Dict[str, pd.DataFrame]
            包含所有乖離率數據的字典
        trades : pd.DataFrame
            交易報告數據

        Returns:
        --------
        Dict[str, Any] : 包含所有分析結果的字典
        """
        all_results = {}

        print("=== BIAS 分析報告 ===\n")

        for bias_name, bias_data in bias_dict.items():
            print(f"\n{'=' * 50}")
            print(f"分析 {bias_name}")
            print(f"{'=' * 50}")

            result = self.plot_bias_vs_return(bias_data, trades, bias_name)
            all_results[bias_name] = result

        return all_results


def create_bias_analyzer():
    """創建BIAS分析器實例的工廠函數"""
    return BiasAnalyzer()
