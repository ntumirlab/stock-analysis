import pandas as pd


class SignalAnalyzer:
    def __init__(self, signal_df):
        self.signal_df = signal_df

    def remove_never_bought_stocks(self):
        # 使用字典先收集所有有效的列
        valid_columns = {
            stock: self.signal_df[stock] for stock in self.signal_df.columns if self.signal_df[stock].any()
        }

        # 一次性使用 pd.concat 建立一個新的 DataFrame
        clean_df = pd.concat(valid_columns, axis=1)

        # 重新設定索引，以保持與原始signal_df相同的索引
        clean_df = clean_df.reindex(self.signal_df.index)

        return clean_df
