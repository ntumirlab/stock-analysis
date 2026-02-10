"""
Single Stock Test Executor

測試策略在每一檔個股上的表現
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_class.oscar_tw_strategy import OscarTWStrategy, AdjustTWMarketInfo

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SingleStockTestExecutor:
    """執行單一股票測試"""

    def __init__(
        self,
        start_date='2020-01-01',
        output_dir='results/single_stock_tests',
        sar_max_dots=2,
        sar_reject_dots=3,
        pool=20,
    ):
        """
        初始化測試執行器

        Args:
            start_date: 回測起始日期
            output_dir: 結果輸出目錄
            sar_max_dots: SAR參數 - 最大買進點數
            sar_reject_dots: SAR參數 - 拒絕買進點數
            pool: 並行處理的worker數量
        """
        self.start_date = start_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sar_max_dots = sar_max_dots
        self.sar_reject_dots = sar_reject_dots
        self.pool = pool

        logger.info(f"初始化 SingleStockTestExecutor")
        logger.info(f"回測起始日期: {start_date}")
        logger.info(f"SAR參數: max_dots={sar_max_dots}, reject_dots={sar_reject_dots}")
        logger.info(f"結果輸出目錄: {self.output_dir}")
        logger.info(f"並行處理數: {pool} workers")

    def run_test(self):
        """執行測試"""
        logger.info("=" * 60)
        logger.info("開始執行單一股票測試")
        logger.info("=" * 60)

        # 初始化策略
        logger.info("初始化策略並生成買賣訊號...")
        strategy = OscarTWStrategy(sar_max_dots=self.sar_max_dots, sar_reject_dots=self.sar_reject_dots)

        # 生成檔案名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'oscar_single_stock_results_{timestamp}.csv'
        output_path = self.output_dir / filename

        # 取得 base_position
        base_position = strategy.base_position.loc[self.start_date :].copy()

        # 執行單一股票回測並儲存
        logger.info("開始對每一檔股票執行回測...")
        results_df = self._run_tests_and_save(base_position=base_position, output_path=str(output_path))

        # 打印統計摘要
        self._print_summary(results_df)

        # 找出表現最佳的股票
        self._print_top_performers(results_df)

        logger.info("\n" + "=" * 60)
        logger.info("測試完成")
        logger.info("=" * 60)

        return results_df

    def _run_tests_and_save(self, base_position, output_path):
        """
        對每一檔股票執行單獨回測並儲存結果

        Args:
            base_position: 基礎持倉訊號
            output_path: CSV輸出路徑

        Returns:
            pd.DataFrame: 包含所有股票回測結果的 DataFrame
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        # 獲取所有曾經出現訊號的股票
        all_stocks = base_position.columns[base_position.any(axis=0)].tolist()

        logger.info(f"開始測試 {len(all_stocks)} 檔股票的個別績效 (使用 {self.pool} workers)")

        results = []

        # 使用並行處理測試所有股票
        with ProcessPoolExecutor(max_workers=self.pool) as executor:
            # 提交所有任務
            future_to_stock = {
                executor.submit(self._test_single_stock, stock, base_position): stock for stock in all_stocks
            }

            # 使用 tqdm 追蹤進度
            with tqdm(total=len(all_stocks), desc="測試進度", unit="stock") as pbar:
                for future in as_completed(future_to_stock):
                    stock = future_to_stock[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            pbar.set_postfix({'stock': stock, 'return': f"{result['annual_return']:.2%}"})
                    except Exception as e:
                        logger.warning(f"股票 {stock} 回測失敗: {e}")
                    finally:
                        pbar.update(1)

        # 轉換為 DataFrame
        df = pd.DataFrame(results)

        # 依照年報酬率排序後儲存 CSV
        df = df.sort_values(by='annual_return', ascending=False)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"結果已儲存至: {output_path}")
        logger.info(f"成功測試 {len(df)} / {len(all_stocks)} 檔股票")

        return df

    @staticmethod
    def _test_single_stock(stock, base_position, fee_ratio=0.001425, tax_ratio=0.003):
        """
        測試單一股票 (靜態方法供並行處理使用)

        Args:
            stock: 股票代碼
            base_position: 基礎持倉訊號
            fee_ratio: 手續費率
            tax_ratio: 證交稅率

        Returns:
            dict: 單一股票的回測結果
        """
        from finlab.backtest import sim

        try:
            # 建立單一股票的持倉訊號
            single_stock_position = pd.DataFrame(False, index=base_position.index, columns=base_position.columns)
            single_stock_position[stock] = base_position[stock]

            # 執行回測
            report = sim(
                position=single_stock_position,
                resample='D',
                upload=False,
                market=AdjustTWMarketInfo(),
                fee_ratio=fee_ratio,
                tax_ratio=tax_ratio,
                position_limit=1.0,
            )
            report.display(save_report_path=f'assets/OscarTWStrategy/{stock}_report.html')

            # 提取績效指標
            metrics = report.get_metrics()
            trades = report.get_trades()

            # 整理結果 (根據 Finlab 文檔結構)
            result = {
                'stock_id': stock,
                'total_trades': len(trades),
                'annual_return': metrics['profitability']['annualReturn'],
                'max_drawdown': metrics['risk']['maxDrawdown'],
                'sharpe_ratio': metrics['ratio'].get('sharpeRatio', None),
                'sortino_ratio': metrics['ratio'].get('sortinoRatio', None),
                'calmar_ratio': metrics['ratio'].get('calmarRatio', None),
                'volatility': metrics['ratio'].get('volatility', None),
                'profit_factor': metrics['ratio'].get('profitFactor', None),
                'win_rate': metrics['winrate'].get('winRate', None),
                'expectancy': metrics['winrate'].get('expectancy', None),
                'mae': metrics['winrate'].get('mae', None),
                'mfe': metrics['winrate'].get('mfe', None),
                'avg_drawdown': metrics['risk'].get('avgDrawdown', None),
                'avg_drawdown_days': metrics['risk'].get('avgDrawdownDays', None),
                'alpha': metrics['profitability'].get('alpha', None),
                'beta': metrics['profitability'].get('beta', None),
                'total_days': len(base_position),
                'holding_days': single_stock_position[stock].sum(),
            }

            return result

        except Exception as e:
            # 靜默失敗，返回 None
            return None

    def _print_summary(self, df):
        """打印統計摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("整體統計摘要")
        logger.info("=" * 60)

        logger.info(f"總測試股票數: {len(df)}")
        logger.info(f"平均年化報酬率: {df['annual_return'].mean():.2%}")
        logger.info(f"平均最大回檔: {df['max_drawdown'].mean():.2%}")

        if df['sharpe_ratio'].notna().any():
            logger.info(f"平均夏普比率: {df['sharpe_ratio'].mean():.2f}")

        logger.info(f"平均交易次數: {df['total_trades'].mean():.1f}")

        # 統計正報酬股票數
        positive_returns = (df['annual_return'] > 0).sum()
        logger.info(f"正報酬股票數: {positive_returns} ({positive_returns / len(df) * 100:.1f}%)")

    def _print_top_performers(self, df, top_n=10):
        """打印表現最佳的股票"""
        logger.info("\n" + "=" * 60)
        logger.info(f"年化報酬率 Top {top_n}")
        logger.info("=" * 60)

        top_by_return = df.nlargest(top_n, 'annual_return')[
            ['stock_id', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'total_trades']
        ].copy()

        # 格式化顯示
        top_by_return['annual_return'] = top_by_return['annual_return'].apply(lambda x: f"{x:.2%}")
        top_by_return['max_drawdown'] = top_by_return['max_drawdown'].apply(lambda x: f"{x:.2%}")
        top_by_return['sharpe_ratio'] = top_by_return['sharpe_ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

        print(top_by_return.to_string(index=False))

        # 如果有夏普比率資料，也打印夏普比率 Top N
        if df['sharpe_ratio'].notna().any():
            logger.info("\n" + "=" * 60)
            logger.info(f"夏普比率 Top {top_n}")
            logger.info("=" * 60)

            top_by_sharpe = df.nlargest(top_n, 'sharpe_ratio')[
                ['stock_id', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'total_trades']
            ].copy()

            # 格式化顯示
            top_by_sharpe['annual_return'] = top_by_sharpe['annual_return'].apply(lambda x: f"{x:.2%}")
            top_by_sharpe['max_drawdown'] = top_by_sharpe['max_drawdown'].apply(lambda x: f"{x:.2%}")
            top_by_sharpe['sharpe_ratio'] = top_by_sharpe['sharpe_ratio'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

            print(top_by_sharpe.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Single Stock Test Executor')
    parser.add_argument('--start_date', type=str, default='1970-01-01', help='回測起始日期')
    parser.add_argument('--output_dir', type=str, default='assets/OscarTWStrategy', help='結果輸出目錄')
    parser.add_argument('--sar_max_dots', type=int, default=2, help='SAR最大買進點數')
    parser.add_argument('--sar_reject_dots', type=int, default=3, help='SAR拒絕買進點數')
    parser.add_argument('--pool', type=int, default=20, help='並行處理的worker數量')

    args = parser.parse_args()

    executor = SingleStockTestExecutor(
        start_date=args.start_date,
        output_dir=args.output_dir,
        sar_max_dots=args.sar_max_dots,
        sar_reject_dots=args.sar_reject_dots,
        pool=args.pool,
    )

    executor.run_test()
