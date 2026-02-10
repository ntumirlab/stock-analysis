from dao.balance_dao import BalanceDAO
import datetime
from zoneinfo import ZoneInfo


class BalanceService:
    def __init__(self):
        self.balance_dao = BalanceDAO()

    def get_balance_history(self, account_id, start_date, end_date):
        """
        獲取指定日期範圍內的餘額歷史

        Args:
            account_id (int): 帳戶ID
            start_date (datetime.date): 開始日期
            end_date (datetime.date): 結束日期

        Returns:
            list: 餘額歷史記錄列表
        """
        return self.balance_dao.get_balance_history(account_id, start_date, end_date)

    def get_latest_balance(self, account_id):
        """
        獲取帳戶最新的餘額記錄

        Args:
            account_id (int): 帳戶ID

        Returns:
            dict: 最新餘額記錄
        """
        return self.balance_dao.get_latest_balance(account_id)

    def get_balance_trend_data(self, account_id, start_date, end_date):
        """
        獲取資金水位趨勢圖所需數據

        Args:
            account_id (int): 帳戶ID
            start_date (datetime.date): 開始日期
            end_date (datetime.date): 結束日期

        Returns:
            list: 資金水位趨勢數據
        """
        balance_history = self.balance_dao.get_balance_history(account_id, start_date, end_date)

        # 處理數據便於圖表顯示
        for record in balance_history:
            # 轉換時間戳格式為日期
            if 'fetch_timestamp' in record:
                try:
                    ts = record['fetch_timestamp']
                    if isinstance(ts, str):
                        dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        record['date'] = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    record['date'] = "未知日期"

        return balance_history

    def get_monthly_return_data(self, account_id, start_year=None, end_year=None):
        """
        計算月度回報率數據，用於熱力圖

        Args:
            account_id (int): 帳戶ID
            start_year (int, optional): 開始年份，默認為當前年份減3
            end_year (int, optional): 結束年份，默認為當前年份

        Returns:
            tuple: (熱力圖數據列表, 年份列表, 最大回報率, 最小回報率)
        """
        # 設置默認年份範圍
        current_year = datetime.datetime.now(ZoneInfo("Asia/Taipei")).year
        if start_year is None:
            start_year = current_year - 3
        if end_year is None:
            end_year = current_year

        # 獲取每月首日和末日的餘額數據
        monthly_data = self.balance_dao.get_monthly_balance_data(account_id, start_year, end_year)

        # 計算每月回報率
        heatmap_data = []
        years = sorted(monthly_data.keys())

        max_return = -float('inf')
        min_return = float('inf')

        for year in years:
            for month in sorted(monthly_data[year].keys()):
                month_data = monthly_data[year][month]

                # 檢查是否有起始和結束數據
                if 'start' in month_data and 'end' in month_data:
                    start_assets = month_data['start']
                    end_assets = month_data['end']

                    if start_assets > 0:  # 避免除以零
                        monthly_return = (end_assets - start_assets) / start_assets * 100
                    else:
                        monthly_return = 0

                    # 更新最大和最小回報率
                    max_return = max(max_return, monthly_return)
                    min_return = min(min_return, monthly_return)

                    # 添加到熱力圖數據
                    heatmap_data.append({'year': year, 'month': month, 'return': round(monthly_return, 2)})

        # 處理沒有數據的情況
        if not heatmap_data:
            return [], years, 0, 0

        # 如果最大最小值相等，設置一個合理的範圍
        if max_return == min_return:
            max_return += 5
            min_return -= 5

        return heatmap_data, years, max_return, min_return
