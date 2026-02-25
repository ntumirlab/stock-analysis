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
from tests.oscar_tw_strategy.utils.drawing_overall_html import dataframe_to_sortable_html
from tests.oscar_tw_strategy.utils.drawing_history_visualization import create_trading_visualization, prepare_price_data
from tests.oscar_tw_strategy.utils.drawing_param_comparison import create_param_comparison_chart

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        stock_id=None,
        optimize_params=False
    ):
        """
        初始化測試執行器
        
        Args:
            start_date: 回測起始日期
            output_dir: 結果輸出目錄
            sar_max_dots: SAR參數 - 最大買進點數
            sar_reject_dots: SAR參數 - 拒絕買進點數
            pool: 並行處理的worker數量
            stock_id: 指定測試單一股票（可選）
            optimize_params: 是否進行參數優化（僅在指定stock_id時有效）
        """
        self.start_date = start_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sar_max_dots = sar_max_dots
        self.sar_reject_dots = sar_reject_dots
        self.pool = pool
        self.stock_id = stock_id
        self.optimize_params = optimize_params
        
        logger.info(f"初始化 SingleStockTestExecutor")
        logger.info(f"回測起始日期: {start_date}")
        logger.info(f"SAR參數: max_dots={sar_max_dots}, reject_dots={sar_reject_dots}")
        logger.info(f"結果輸出目錄: {self.output_dir}")
        if stock_id:
            logger.info(f"單一股票測試模式: {stock_id}")
            if optimize_params:
                logger.info(f"參數優化模式已啟用")
        else:
            logger.info(f"全市場測試模式，並行處理數: {pool} workers")
    
    @staticmethod
    def generate_param_grid():
        """
        生成參數網格（優化 SAR 和 MACD 參數）
        
        Returns:
            list: 參數組合列表
        """
        param_combinations = []
        
        # SAR 參數
        # max_dots: 1, 2, 3 (買進區間)
        # acceleration: 0.01, 0.02, 0.03 (加速因子)
        # maximum: 0.15, 0.2, 0.25 (最大加速因子)
        
        # MACD 參數
        # fast: 10, 12, 14
        # slow: 24, 26, 28  
        # signal: 8, 9, 10
        
        sar_max_dots_values = [1, 2, 3]
        sar_acceleration_values = [0.01, 0.02, 0.03]
        sar_maximum_values = [0.15, 0.2, 0.25]
        
        macd_fast_values = [10, 12, 14]
        macd_slow_values = [24, 26, 28]
        macd_signal_values = [8, 9, 10]
        
        for sar_max in sar_max_dots_values:
            for sar_accel in sar_acceleration_values:
                for sar_max_val in sar_maximum_values:
                    for macd_fast in macd_fast_values:
                        for macd_slow in macd_slow_values:
                            for macd_signal in macd_signal_values:
                                if macd_fast < macd_slow:  # 確保 fast < slow
                                    param_combinations.append({
                                        'sar_max_dots': sar_max,
                                        'sar_params': {
                                            'acceleration': sar_accel,
                                            'maximum': sar_max_val
                                        },
                                        'macd_params': {
                                            'fastperiod': macd_fast,
                                            'slowperiod': macd_slow,
                                            'signalperiod': macd_signal
                                        }
                                    })
        
        logger.info(f"生成了 {len(param_combinations)} 組參數組合")
        return param_combinations
    
    def run_test(self):
        """執行測試"""
        logger.info("=" * 60)
        logger.info("開始執行單一股票測試")
        logger.info("=" * 60)
        
        # 根據 optimize_params 和 stock_id 決定執行模式
        if self.optimize_params:
            # 參數優化模式
            if self.stock_id:
                # 單一股票參數優化
                logger.info(f"執行單一股票參數優化: {self.stock_id}")
                result = self._run_param_optimization(self.stock_id)
            else:
                # 全市場參數優化
                logger.info("執行全市場參數優化")
                result = self._run_all_stocks_param_optimization()
        else:
            # 非優化模式
            if self.stock_id:
                # 單一股票單一參數模式
                logger.info("初始化策略並生成買賣訊號...")
                strategy = OscarTWStrategy(
                    sar_max_dots=self.sar_max_dots,
                    sar_reject_dots=self.sar_reject_dots
                )
                base_position = strategy.base_position.loc[self.start_date:].copy()
                
                logger.info(f"執行單一股票回測: {self.stock_id}")
                result = self._run_single_stock_with_visualization(
                    stock_id=self.stock_id,
                    strategy=strategy,
                    base_position=base_position
                )
            else:
                # 全市場模式：對所有股票執行回測並生成 HTML 表格
                logger.info("初始化策略並生成買賣訊號...")
                strategy = OscarTWStrategy(
                    sar_max_dots=self.sar_max_dots,
                    sar_reject_dots=self.sar_reject_dots
                )
                base_position = strategy.base_position.loc[self.start_date:].copy()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_filename = f'oscar_single_stock_results_{timestamp}.csv'
                html_filename = f'oscar_single_stock_results_{timestamp}.html'
                csv_path = self.output_dir / csv_filename
                html_path = self.output_dir / html_filename
                
                logger.info("開始對每一檔股票執行回測...")
                results_df = self._run_tests_and_save(
                    base_position=base_position,
                    output_path=str(csv_path)
                )
                
                # 生成互動式 HTML 表格
                logger.info("生成互動式 HTML 表格...")
                dataframe_to_sortable_html(
                    df=results_df,
                    output_path=str(html_path),
                    title=f"Oscar Strategy - Single Stock Results ({timestamp})"
                )
                logger.info(f"HTML 表格已儲存至: {html_path}")
                
                # 打印統計摘要
                self._print_summary(results_df)
                
                # 找出表現最佳的股票
                self._print_top_performers(results_df)
                
                result = results_df
        
        logger.info("\n" + "=" * 60)
        logger.info("測試完成")
        logger.info("=" * 60)
        
        return result
    
    def _run_param_optimization(self, stock_id):
        """
        對單一股票執行參數優化（並行處理）
        
        Args:
            stock_id: 股票代碼
            
        Returns:
            dict: 最佳參數的回測結果
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        # 預先載入市場數據（只載入一次）
        logger.info("預先載入市場數據...")
        market_data = OscarTWStrategy.load_market_data()
        logger.info("市場數據載入完成")
        
        # 生成參數組合
        param_grid = self.generate_param_grid()
        logger.info(f"開始測試 {len(param_grid)} 組參數（使用 {self.pool} workers）...")
        
        param_results = []
        
        # 使用並行處理測試所有參數組合
        with ProcessPoolExecutor(max_workers=self.pool) as executor:
            # 提交所有任務（傳入預載數據）
            future_to_params = {
                executor.submit(
                    self._test_single_param,
                    stock_id,
                    params,
                    self.start_date,
                    self.sar_reject_dots,
                    market_data
                ): params
                for params in param_grid
            }
            
            # 使用 tqdm 追蹤進度
            with tqdm(total=len(param_grid), desc="參數優化進度", unit="param") as pbar:
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        result = future.result()
                        if result:
                            param_results.append(result)
                            pbar.set_postfix({
                                'SAR': params['sar_max_dots'],
                                'return': f"{result['annual_return']:.2%}"
                            })
                    except Exception as e:
                        logger.warning(f"參數組合測試失敗: {e}")
                    finally:
                        pbar.update(1)
        
        if not param_results:
            logger.error("所有參數組合測試均失敗")
            return None
        
        # 找出最佳結果
        best_result = max(param_results, key=lambda x: x['annual_return'])
        
        # 輸出最佳參數資訊
        logger.info("\n" + "=" * 60)
        logger.info("最佳參數組合")
        logger.info("=" * 60)
        logger.info(f"SAR max_dots: {best_result['sar_max_dots']}")
        logger.info(f"SAR accel: {best_result['sar_accel']:.2f}, maximum: {best_result['sar_maximum']:.2f}")
        logger.info(f"MACD: fast={best_result['macd_fast']}, slow={best_result['macd_slow']}, signal={best_result['macd_signal']}")
        logger.info(f"年化報酬率: {best_result['annual_return']:.2%}")
        logger.info(f"最大回檔: {best_result['max_drawdown']:.2%}")
        logger.info(f"夏普比率: {best_result['sharpe_ratio']:.2f}" if best_result['sharpe_ratio'] else "夏普比率: N/A")
        logger.info(f"交易次數: {best_result['total_trades']}")
        logger.info("=" * 60)
        
        # 使用最佳參數重新初始化策略並生成完整的視覺化
        logger.info("使用最佳參數重新初始化策略並生成視覺化...")
        best_params = best_result['params']
        best_strategy = OscarTWStrategy(
            sar_max_dots=best_params['sar_max_dots'],
            sar_reject_dots=self.sar_reject_dots,
            sar_params=best_params['sar_params'],
            macd_params=best_params['macd_params']
        )
        best_base_position = best_strategy.base_position.loc[self.start_date:].copy()
        
        final_result = self._run_single_stock_with_visualization(
            stock_id=stock_id,
            strategy=best_strategy,
            base_position=best_base_position
        )
        
        # 生成參數比較圖表
        param_comparison_path = self.output_dir / f'{stock_id}_param_comparison.html'
        create_param_comparison_chart(
            stock_id=stock_id,
            param_results=param_results,
            output_path=str(param_comparison_path)
        )
        logger.info(f"參數比較圖表已儲存至: {param_comparison_path}")
        
        # 添加最佳參數資訊到結果
        final_result['best_params'] = best_result['params']
        final_result['param_comparison_path'] = str(param_comparison_path)
        
        return final_result
    
    def _run_all_stocks_param_optimization(self):
        """
        對所有股票執行參數優化（並行處理）
        為每個股票找出最佳參數組合
        
        Returns:
            pd.DataFrame: 包含所有股票最佳參數的回測結果
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        # 預先載入市場數據（只載入一次）
        logger.info("預先載入市場數據...")
        market_data = OscarTWStrategy.load_market_data()
        logger.info("市場數據載入完成")
        
        # 先用默認參數初始化策略，獲取所有有訊號的股票列表
        logger.info("初始化策略以獲取股票列表...")
        temp_strategy = OscarTWStrategy(
            sar_max_dots=self.sar_max_dots,
            sar_reject_dots=self.sar_reject_dots,
            market_data=market_data
        )
        base_position = temp_strategy.base_position.loc[self.start_date:].copy()
        all_stocks = base_position.columns[base_position.any(axis=0)].tolist()
        del temp_strategy, base_position  # 釋放記憶體
        
        logger.info(f"開始對 {len(all_stocks)} 檔股票執行參數優化（使用 {self.pool} workers）...")
        
        # 生成參數組合
        param_grid = self.generate_param_grid()
        logger.info(f"每檔股票將測試 {len(param_grid)} 組參數")
        
        results = []
        
        # 為每檔股票執行參數優化
        with tqdm(total=len(all_stocks), desc="股票優化進度", unit="stock") as stock_pbar:
            for stock_id in all_stocks:
                logger.info(f"\n處理股票 {stock_id}...")
                
                # 並行測試該股票的所有參數組合（傳入預載數據）
                stock_param_results = []
                with ProcessPoolExecutor(max_workers=self.pool) as executor:
                    future_to_params = {
                        executor.submit(
                            self._test_single_param,
                            stock_id,
                            params,
                            self.start_date,
                            self.sar_reject_dots,
                            market_data
                        ): params
                        for params in param_grid
                    }
                    
                    for future in as_completed(future_to_params):
                        try:
                            result = future.result()
                            if result:
                                stock_param_results.append(result)
                        except Exception as e:
                            logger.warning(f"股票 {stock_id} 參數測試失敗: {e}")
                
                # 找出該股票的最佳參數
                if stock_param_results:
                    best_result = max(stock_param_results, key=lambda x: x['annual_return'])
                    best_result['stock_id'] = stock_id  # 添加股票代碼
                    results.append(best_result)
                    logger.info(f"  {stock_id} 最佳參數: SAR={best_result['sar_max_dots']} (accel={best_result['sar_accel']:.2f}, max={best_result['sar_maximum']:.2f}), "
                              f"MACD=({best_result['macd_fast']},{best_result['macd_slow']},{best_result['macd_signal']}), "
                              f"年化報酬: {best_result['annual_return']:.2%}")
                    
                    # 為該股票生成完整的報告和視覺化（使用最佳參數重新初始化策略）
                    logger.info(f"  生成 {stock_id} 的完整報告...")
                    best_params = best_result['params']
                    best_strategy = OscarTWStrategy(
                        sar_max_dots=best_params['sar_max_dots'],
                        sar_reject_dots=self.sar_reject_dots,
                        sar_params=best_params['sar_params'],
                        macd_params=best_params['macd_params'],
                        market_data=market_data
                    )
                    best_base_position = best_strategy.base_position.loc[self.start_date:].copy()
                    
                    # 生成報告和視覺化
                    self._run_single_stock_with_visualization(
                        stock_id=stock_id,
                        strategy=best_strategy,
                        base_position=best_base_position
                    )
                    
                    # 生成參數比較圖表
                    param_comparison_path = self.output_dir / f'{stock_id}_param_comparison.html'
                    create_param_comparison_chart(
                        stock_id=stock_id,
                        param_results=stock_param_results,
                        output_path=str(param_comparison_path)
                    )
                    logger.info(f"  {stock_id} 參數比較圖表已儲存")
                    
                    # 清理策略物件釋放記憶體
                    del best_strategy, best_base_position
                else:
                    logger.warning(f"  {stock_id} 所有參數組合測試均失敗")
                
                stock_pbar.update(1)
        
        if not results:
            logger.error("所有股票的參數優化均失敗")
            return None
        
        # 轉換為 DataFrame
        df = pd.DataFrame(results)
        
        # 重新排列欄位順序
        column_order = ['stock_id', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'total_trades',
                       'sar_max_dots', 'sar_accel', 'sar_maximum', 
                       'macd_fast', 'macd_slow', 'macd_signal']
        df = df[column_order]
        
        # 依照年報酬率排序
        df = df.sort_values(by='annual_return', ascending=False)
        
        # 儲存結果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'oscar_optimized_results.csv'
        html_filename = f'oscar_optimized_results.html'
        csv_path = self.output_dir / csv_filename
        html_path = self.output_dir / html_filename
        
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"優化結果已儲存至: {csv_path}")
        
        # 生成互動式 HTML 表格
        logger.info("生成互動式 HTML 表格...")
        dataframe_to_sortable_html(
            df=df,
            output_path=str(html_path),
            title=f"Oscar Strategy - Optimized Results ({timestamp})"
        )
        logger.info(f"HTML 表格已儲存至: {html_path}")
        
        # 打印統計摘要
        self._print_summary(df)
        
        # 找出表現最佳的股票
        self._print_top_performers(df)
        
        return df
    
    def _run_single_stock_with_visualization(self, stock_id, strategy, base_position):
        """
        執行單一股票回測並生成視覺化
        
        Args:
            stock_id: 股票代碼
            strategy: 策略實例
            base_position: 基礎持倉訊號
            
        Returns:
            dict: 回測結果
        """
        from finlab.backtest import sim
        
        # 檢查股票是否在資料中
        if stock_id not in base_position.columns:
            logger.error(f"股票 {stock_id} 不在市場資料中")
            return None
        
        # 建立單一股票的持倉訊號
        single_stock_position = pd.DataFrame(
            False, 
            index=base_position.index, 
            columns=base_position.columns
        )
        single_stock_position[stock_id] = base_position[stock_id]
        
        # 執行回測
        logger.info(f"執行 {stock_id} 回測...")
        report = sim(
            position=single_stock_position,
            resample=None,
            upload=False,
            market=AdjustTWMarketInfo(),
            fee_ratio=0.001425,
            tax_ratio=0.003,
            position_limit=1.0
        )
        
        # 儲存回測報告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'{stock_id}.html'
        report.display(save_report_path=str(report_path))
        logger.info(f"回測報告已儲存至: {report_path}")
        
        # 提取績效指標
        metrics = report.get_metrics()
        trades = report.get_trades()
        
        # 打印績效摘要
        logger.info("\n" + "=" * 60)
        logger.info(f"股票 {stock_id} 績效摘要")
        logger.info("=" * 60)
        logger.info(f"交易次數: {len(trades)}")
        logger.info(f"年化報酬率: {metrics['profitability']['annualReturn']:.2%}")
        logger.info(f"最大回檔: {metrics['risk']['maxDrawdown']:.2%}")
        if 'sharpeRatio' in metrics['ratio']:
            logger.info(f"夏普比率: {metrics['ratio']['sharpeRatio']:.2f}")
        logger.info("=" * 60)
        
        # 準備視覺化數據
        logger.info("生成交易視覺化圖表...")
        
        # 準備價格數據
        price_df = prepare_price_data(
            stock_id=stock_id,
            market_data=strategy.market_data,
            start_date=self.start_date
        )
        
        # 取得該股票的指標數據
        sar_stock = strategy.sar_values[stock_id].loc[self.start_date:]
        macd_dif_stock = strategy.macd_dif[stock_id].loc[self.start_date:]
        macd_dea_stock = strategy.macd_dea[stock_id].loc[self.start_date:]
        macd_hist_stock = strategy.macd_histogram[stock_id].loc[self.start_date:]
        
        # 取得該股票的交易價格（開盤價）
        trade_price_stock = strategy.trade_price[stock_id].loc[self.start_date:]
        
        # 取得該股票的法人買賣超數據
        foreign_buy_stock = strategy.institutional_condition['foreign_buy'][stock_id].loc[self.start_date:]
        trust_buy_stock = strategy.institutional_condition['trust_buy'][stock_id].loc[self.start_date:]
        dealer_buy_stock = strategy.institutional_condition['dealer_buy'][stock_id].loc[self.start_date:]
        
        # 取得策略持倉訊號
        position_stock = base_position[stock_id].loc[self.start_date:]
        
        # 計算實際交易訊號（從持倉變化計算）
        position_changes = position_stock.astype(int).diff()
        actual_buy_signals = position_changes == 1  # 0->1 表示買入
        actual_sell_signals = position_changes == -1  # 1->0 表示賣出
        
        # 生成視覺化圖表
        viz_path = self.output_dir / f'{stock_id}_visualization.html'
        create_trading_visualization(
            stock_id=stock_id,
            price_data=price_df,
            sar_values=sar_stock,
            macd_dif=macd_dif_stock,
            macd_dea=macd_dea_stock,
            macd_histogram=macd_hist_stock,
            buy_signals=actual_buy_signals,
            sell_signals=actual_sell_signals,
            position=position_stock,
            trade_price=trade_price_stock,
            foreign_buy=foreign_buy_stock,
            trust_buy=trust_buy_stock,
            dealer_buy=dealer_buy_stock,
            output_path=str(viz_path)
        )
        logger.info(f"視覺化圖表已儲存至: {viz_path}")
        
        # 返回結果
        result = {
            'stock_id': stock_id,
            'total_trades': len(trades),
            'annual_return': metrics['profitability']['annualReturn'],
            'max_drawdown': metrics['risk']['maxDrawdown'],
            'sharpe_ratio': metrics['ratio'].get('sharpeRatio', None),
            'sortino_ratio': metrics['ratio'].get('sortinoRatio', None),
            'calmar_ratio': metrics['ratio'].get('calmarRatio', None),
            'report_path': str(report_path),
            'visualization_path': str(viz_path)
        }
        
        return result
    
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
                executor.submit(
                    self._test_single_stock,
                    stock,
                    base_position
                ): stock
                for stock in all_stocks
            }
            
            # 使用 tqdm 追蹤進度
            with tqdm(total=len(all_stocks), desc="測試進度", unit="stock") as pbar:
                for future in as_completed(future_to_stock):
                    stock = future_to_stock[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            pbar.set_postfix({
                                'stock': stock,
                                'return': f"{result['annual_return']:.2%}"
                            })
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
    def _test_single_param(stock_id, params, start_date, sar_reject_dots=2, market_data=None):
        """
        測試單一參數組合 (靜態方法供並行處理使用)
        
        Args:
            stock_id: 股票代碼
            params: 參數字典
            start_date: 回測開始日期
            sar_reject_dots: SAR拒絕點數
            market_data: 預先載入的市場數據（避免重複載入）
            
        Returns:
            dict: 單一參數組合的回測結果
        """
        from finlab.backtest import sim
        
        try:
            # 初始化策略（傳入預載數據）
            strategy = OscarTWStrategy(
                sar_max_dots=params['sar_max_dots'],
                sar_reject_dots=sar_reject_dots,
                sar_params=params['sar_params'],
                macd_params=params['macd_params'],
                market_data=market_data
            )
            
            base_position = strategy.base_position.loc[start_date:].copy()
            
            # 檢查股票是否在資料中
            if stock_id not in base_position.columns:
                return None
            
            # 建立單一股票的持倉訊號
            single_stock_position = pd.DataFrame(
                False, 
                index=base_position.index, 
                columns=base_position.columns
            )
            single_stock_position[stock_id] = base_position[stock_id]
            
            # 執行回測
            report = sim(
                position=single_stock_position,
                resample=None,
                upload=False,
                market=AdjustTWMarketInfo(),
                fee_ratio=0.001425,
                tax_ratio=0.003,
                position_limit=1.0
            )
            
            # 提取績效指標
            metrics = report.get_metrics()
            trades = report.get_trades()
            
            # 記錄結果（只儲存數值,不儲存大物件）
            result = {
                'sar_max_dots': params['sar_max_dots'],
                'sar_accel': params['sar_params']['acceleration'],
                'sar_maximum': params['sar_params']['maximum'],
                'macd_fast': params['macd_params']['fastperiod'],
                'macd_slow': params['macd_params']['slowperiod'],
                'macd_signal': params['macd_params']['signalperiod'],
                'total_trades': len(trades),
                'annual_return': metrics['profitability']['annualReturn'],
                'max_drawdown': metrics['risk']['maxDrawdown'],
                'sharpe_ratio': metrics['ratio'].get('sharpeRatio', None),
                'params': params  # 只保留參數字典，不保留 strategy 和 base_position
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"參數組合測試失敗 ({params}): {e}")
            return None
    
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
            single_stock_position = pd.DataFrame(
                False, 
                index=base_position.index, 
                columns=base_position.columns
            )
            single_stock_position[stock] = base_position[stock]
            
            # 執行回測
            # 說明：
            #   - 我們刻意使用 resample=None，而不是預設的 'D' (日頻率)。
            #   - single_stock_position 已經依照 base_position.index 定義好持倉頻率
            #     （通常為每日收盤），讓回測引擎直接依照該索引時間點進行部位調整。
            #   - 如果改成 resample='D'，finlab.backtest.sim 會再做一次日頻率重取樣，
            #     可能改變實際交易執行時間與再平衡節奏，導致與策略原始訊號不一致，
            #     並且在不同資料頻率下可能產生額外的重複聚合或邏輯偏差。
            #   - 因此這裡設定 resample=None 是「刻意為之」：讓每次調整部位的時點
            #     完全跟 single_stock_position.index 對齊。未來若要修改此設定，
            #     請先確認策略訊號頻率與交易假設，並重新比對回測結果。
            report = sim(
                position=single_stock_position,
                resample=None,
                upload=False,
                market=AdjustTWMarketInfo(),
                fee_ratio=fee_ratio,
                tax_ratio=tax_ratio,
                position_limit=1.0
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
        logger.info(f"正報酬股票數: {positive_returns} ({positive_returns/len(df)*100:.1f}%)")
    
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
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='回測起始日期')
    parser.add_argument('--output_dir', type=str, default='assets/OscarTWStrategy/single_stock', help='結果輸出目錄')
    parser.add_argument('--sar_max_dots', type=int, default=2, help='SAR最大買進點數')
    parser.add_argument('--sar_reject_dots', type=int, default=3, help='SAR拒絕買進點數')
    parser.add_argument('--pool', type=int, default=20, help='並行處理的worker數量')
    parser.add_argument('--stock_id', type=str, default=None, help='指定測試單一股票（可選）')
    parser.add_argument('--optimize', action='store_true', help='啟用參數優化（僅在指定stock_id時有效）')
    
    args = parser.parse_args()
    
    executor = SingleStockTestExecutor(
        start_date=args.start_date,
        output_dir=args.output_dir,
        sar_max_dots=args.sar_max_dots,
        sar_reject_dots=args.sar_reject_dots,
        pool=args.pool,
        stock_id=args.stock_id,
        optimize_params=args.optimize
    )
    
    executor.run_test()
