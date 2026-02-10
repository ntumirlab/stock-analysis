import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo


class ReportSaver:
    def __init__(self, report, output_dir='output'):
        self.report = report
        self.output_dir = output_dir

    def create_custom_filepath(self, prefix, middle_part=None, extension='xlsx'):
        if middle_part is None:
            middle_part = datetime.now(ZoneInfo("Asia/Taipei")).strftime('%Y%m%d_%H%M%S')
        filename = f'{prefix}_{middle_part}.{extension}'
        filepath = os.path.join(self.output_dir, filename)
        return filepath

    def ensure_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_report_html(self, middle_part=None):
        filepath = self.create_custom_filepath('report', middle_part, 'html')
        self.ensure_directory()
        self.report.display(save_report_path=filepath)
        print(f'Report saved to {filepath}')

    def save_trades_excel(self, middle_part=None):
        filepath = self.create_custom_filepath('trades_record', middle_part, 'xlsx')
        # 獲取交易記錄 DataFrame
        trades_df = self.report.get_trades()
        self.ensure_directory()
        trades_df.to_excel(filepath, index=True)
        print(f'Trades record saved to {filepath}')

    def save_dict_to_json(self, data_dict, prefix, middle_part=None):
        """將字典數據保存為JSON檔案，檔案名包含自定義部分或時間戳記。"""
        filepath = self.create_custom_filepath(prefix, middle_part, 'json')
        self.ensure_directory()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)
        print(f'Data saved to {filepath}')

    def save_report_stats_to_json(self, resample='1d', riskfree_rate=0.02, middle_part=None):
        """將報告的統計數據保存為JSON檔案。"""
        stats = self.report.get_stats(resample=resample, riskfree_rate=riskfree_rate)
        self.save_dict_to_json(stats, 'report_stats', middle_part)

    def save_position_info_to_json(self, middle_part=None):
        """將報告的持有部位與預期換股資訊保存為JSON檔案。"""
        position_info = self.report.position_info()
        self.save_dict_to_json(position_info, 'position_info', middle_part)
