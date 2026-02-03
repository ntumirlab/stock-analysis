"""
Max Stock Parameter Test Executor

測試不同持股數量限制 (max_stocks) 對策略績效的影響
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_class.oscar_tw_strategy import OscarTWStrategy

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaxStockTestExecutor:
    """執行 max_stocks 參數測試"""
    
    def __init__(
        self,
        min_stocks=5,
        max_stocks=20,
        start_date='2020-01-01',
        output_dir='assets/OscarTWStrategy/max_stock_arg_tests',
        pool=20
    ):
        """
        初始化測試執行器
        
        Args:
            min_stocks: 最小持股數
            max_stocks: 最大持股數
            start_date: 回測起始日期
            output_dir: 結果輸出目錄
            pool: 並行處理的worker數量
        """
        self.min_stocks = min_stocks
        self.max_stocks = max_stocks
        self.start_date = start_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pool = pool
        
        logger.info(f"初始化 MaxStockTestExecutor: 測試範圍 {min_stocks}-{max_stocks} 檔")
        logger.info(f"回測起始日期: {start_date}")
        logger.info(f"結果輸出目錄: {self.output_dir}")
        logger.info(f"並行處理數: {pool} workers")
    
    def run_test(self):
        """執行所有測試"""
        logger.info("=" * 60)
        logger.info("開始執行 Max Stocks 參數測試")
        logger.info("=" * 60)
        
        # 初始化策略（只需載入一次數據和生成訊號）
        logger.info("初始化策略並生成買賣訊號...")
        strategy = OscarTWStrategy(sar_max_dots=2, sar_reject_dots=3)
        
        # 取得 base_position 並套用起始日期
        base_position = strategy.base_position.loc[self.start_date:].copy()
        
        # 準備測試參數
        test_params = list(range(self.min_stocks, self.max_stocks + 1))
        
        # 使用並行處理測試不同的 max_stocks 值
        results = []
        
        logger.info(f"提交 {len(test_params)} 個測試任務到處理池...")
        
        with ProcessPoolExecutor(max_workers=self.pool) as executor:
            # 提交所有任務
            future_to_n_stocks = {
                executor.submit(self._test_single_config, base_position, n_stocks, 0.001425, 0.003): n_stocks
                for n_stocks in test_params
            }
            
            # 收集結果
            completed = 0
            total = len(test_params)
            for future in as_completed(future_to_n_stocks):
                n_stocks = future_to_n_stocks[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        raise ValueError("測試成功但未返回結果")
                except Exception as e:
                    logger.error(f"max_stocks={n_stocks} 時發生錯誤: {e}")
        
        # 將結果儲存為 CSV
        if results:
            self._save_results(results)
        else:
            logger.warning("沒有成功的測試結果")
        
        logger.info("\n" + "=" * 60)
        logger.info("測試完成")
        logger.info("=" * 60)
    
    @staticmethod
    def _test_single_config(base_position, n_stocks, fee_ratio=0.001425, tax_ratio=0.003):
        """測試單一配置"""
        from finlab.backtest import sim
        import traceback
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # 建立全新的 DataFrame 而非使用 apply
            # 轉換成 numpy 以避免 pandas index 問題
            final_position_values = base_position.values.copy()
            
            for i in range(len(base_position)):
                row_values = final_position_values[i]
                if row_values.any():
                    # 找到 True 的位置索引
                    true_indices = row_values.nonzero()[0]
                    if len(true_indices) > n_stocks:
                        # 重置所有為 False，只保留前 n_stocks 個
                        final_position_values[i] = False
                        final_position_values[i, true_indices[:n_stocks]] = True
            
            # 使用原始的 index 和 columns 重建 DataFrame
            final_position = pd.DataFrame(
                final_position_values,
                index=base_position.index.copy(),
                columns=base_position.columns.copy()
            )
            
            # 執行回測
            report = sim(
                position=final_position,
                resample='D',
                upload=False,
                fee_ratio=fee_ratio,
                tax_ratio=tax_ratio,
                position_limit=1.0 / n_stocks
            )
            
            # 提取績效指標
            metrics = report.get_metrics()
            
            # 整理結果 (根據 Finlab 文檔結構)
            result = {
                'max_stocks': n_stocks,
                'position_limit_pct': 100.0 / n_stocks,
                'annual_return': metrics['profitability']['annualReturn'],
                'max_drawdown': metrics['risk']['maxDrawdown'],
                'sharpe_ratio': metrics['ratio'].get('sharpeRatio', None),
                'sortino_ratio': metrics['ratio'].get('sortinoRatio', None),
                'calmar_ratio': metrics['ratio'].get('calmarRatio', None),
                'volatility': metrics['ratio'].get('volatility', None),
                'profit_factor': metrics['ratio'].get('profitFactor', None),
                'win_rate': metrics['winrate'].get('winRate', None),
                'total_trades': len(report.get_trades()),
                'avg_drawdown': metrics['risk'].get('avgDrawdown', None),
                'alpha': metrics['profitability'].get('alpha', None),
                'beta': metrics['profitability'].get('beta', None),
            }
            
            logger.info(f"成功完成 max_stocks={n_stocks}, return={result['annual_return']:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"測試 max_stocks={n_stocks} 失敗: {e}")
            logger.error(f"詳細錯誤:\n{traceback.format_exc()}")
            return None
    
    def _save_results(self, results):
        """儲存測試結果"""
        df = pd.DataFrame(results)
        
        # 依 max_stocks 排序
        df = df.sort_values('max_stocks').reset_index(drop=True)
        
        # 生成檔案名稱（包含時間戳記）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'max_stock_test_{self.min_stocks}to{self.max_stocks}_{timestamp}.csv'
        filepath = self.output_dir / filename
        
        # 儲存 CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"\n結果已儲存至: {filepath}")
        
        # 打印摘要表格
        logger.info("\n" + "=" * 60)
        logger.info("測試結果摘要")
        logger.info("=" * 60)
        
        # 格式化顯示
        summary_cols = ['max_stocks', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'total_trades']
        summary_df = df[summary_cols].copy()
        
        # 格式化百分比欄位
        summary_df['annual_return'] = summary_df['annual_return'].apply(lambda x: f"{x:.2%}")
        summary_df['max_drawdown'] = summary_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
        summary_df['sharpe_ratio'] = summary_df['sharpe_ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
        
        print(summary_df.to_string(index=False))
        
        # 找出最佳配置
        best_return_idx = df['annual_return'].idxmax()
        best_sharpe_idx = df['sharpe_ratio'].idxmax() if df['sharpe_ratio'].notna().any() else None
        
        logger.info("\n" + "=" * 60)
        logger.info("最佳配置")
        logger.info("=" * 60)
        logger.info(f"最高年化報酬率: max_stocks={df.loc[best_return_idx, 'max_stocks']}, "
                   f"報酬率={df.loc[best_return_idx, 'annual_return']:.2%}")
        
        if best_sharpe_idx is not None:
            logger.info(f"最高夏普比率: max_stocks={df.loc[best_sharpe_idx, 'max_stocks']}, "
                       f"夏普比率={df.loc[best_sharpe_idx, 'sharpe_ratio']:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Max Stock Parameter Test Executor')
    parser.add_argument('--min_stocks', type=int, default=5, help='最小持股數')
    parser.add_argument('--max_stocks', type=int, default=20, help='最大持股數')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='回測起始日期')
    parser.add_argument('--output_dir', type=str, default='assets/OscarTWStrategy/max_stock_tests', help='結果輸出目錄')
    parser.add_argument('--pool', type=int, default=20, help='並行處理的worker數量')
    
    args = parser.parse_args()
    
    executor = MaxStockTestExecutor(
        min_stocks=args.min_stocks,
        max_stocks=args.max_stocks,
        start_date=args.start_date,
        output_dir=args.output_dir,
        pool=args.pool
    )
    
    executor.run_test()
