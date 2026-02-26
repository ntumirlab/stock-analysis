import pandas as pd


class ReportAnalyzer:
    def __init__(self, report):
        self.report = report

    def analyze_trades_for_date(self, date):
        trades_df = self.report.get_trades()
        # 將字串日期轉換為pandas的日期時間對象
        analysis_date = pd.to_datetime(date)

        # 買入股票的情況
        buys = trades_df[trades_df['entry_date'] == analysis_date]['stock_id'].tolist()

        # 賣出股票的情況
        sells = trades_df[trades_df['exit_date'] == analysis_date]['stock_id'].tolist()

        still_holds = trades_df[
            (trades_df['entry_date'] < analysis_date)
            & ((trades_df['exit_date'] > analysis_date) | (trades_df['exit_date'].isna()))
        ]['stock_id'].tolist()

        return {'buys': buys, 'sells': sells, 'still_holds': still_holds}
