"""
Single Stock Test Executor

測試策略在每一檔個股上的表現

Performance Optimizations (針對144核心機器):
1. Global worker initialization (_init_worker) - Market data loaded once per worker
2. Global base_position in workers - Eliminates 5-10x serialization overhead
3. Flattened task architecture - No nested ProcessPoolExecutor loops
4. Pre-filtering - Skip stocks with zero signals before submission
5. Removed HTML generation from parallel loop - Disk I/O bottleneck eliminated
6. Auto-detected optimal pool size - Prevents context switching (default: CPU cores - 2)

Expected Performance:
- Before: 25% CPU usage, week-long runtime with --pool 144
- After: 90-95% CPU usage, ~10-20x faster with --pool 100-110
"""

import os
import sys
import logging
import pandas as pd
import json
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

# Global variables for worker processes (to avoid expensive pickling)
_WORKER_MARKET_DATA = None
_WORKER_BASE_POSITION = None
_WORKER_POSITION_CACHE = {}  # Cache for pre-calculated positions by param hash

def _init_worker(market_data, base_position=None):
    """Initialize worker process with shared data (called once per worker)"""
    global _WORKER_MARKET_DATA, _WORKER_BASE_POSITION, _WORKER_POSITION_CACHE
    _WORKER_MARKET_DATA = market_data
    _WORKER_BASE_POSITION = base_position
    _WORKER_POSITION_CACHE = {}  # Each worker has its own cache


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
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sar_max_dots = sar_max_dots
        self.sar_reject_dots = sar_reject_dots
        self.pool = pool
        self.stock_id = stock_id
        self.optimize_params = optimize_params
        
        # Warn if pool size is too large
        import os
        cpu_count = os.cpu_count() or 4
        if self.pool > cpu_count:
            logger.warning(f"⚠️  並行處理數 ({self.pool}) 超過CPU核心數 ({cpu_count})，可能導致效能下降！")
            logger.warning(f"⚠️  建議設定 --pool {cpu_count} 或更小的值")
        
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
    
    def _save_checkpoint(self, checkpoint_name, data):
        """保存檢查點"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f)
        logger.info(f"檢查點已保存: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_name):
        """載入檢查點"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        return None
    
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
        對單一股票執行參數優化（優化版：預先計算所有策略位置）
        
        Args:
            stock_id: 股票代碼
            
        Returns:
            dict: 最佳參數的回測結果
        """
        from finlab.backtest import sim
        from tqdm import tqdm
        
        # 預先載入市場數據（只載入一次）
        logger.info("預先載入市場數據...")
        market_data = OscarTWStrategy.load_market_data()
        logger.info("市場數據載入完成")
        
        # 生成參數組合
        param_grid = self.generate_param_grid()
        logger.info(f"開始測試 {len(param_grid)} 組參數...")
        
        param_results = []
        
        # 🚀 優化策略：預先計算所有參數的策略位置（避免重複計算）
        logger.info("⚠️  階段1: 預先計算所有參數組合的策略位置（只計算一次）...")
        logger.info(f"⚠️  預估需要 2-5 分鐘計算 {len(param_grid)} 個策略...")
        sys.stdout.flush()
        
        param_positions = {}
        with tqdm(total=len(param_grid), desc="計算策略位置", unit="param", ncols=100, miniters=1) as pbar:
            for params in param_grid:
                try:
                    strategy = OscarTWStrategy(
                        sar_max_dots=params['sar_max_dots'],
                        sar_reject_dots=self.sar_reject_dots,
                        sar_params=params['sar_params'],
                        macd_params=params['macd_params'],
                        market_data=market_data
                    )
                    base_position = strategy.base_position.loc[self.start_date:].copy()
                    
                    # 只保存該股票的位置（節省記憶體）
                    if stock_id in base_position.columns and base_position[stock_id].any():
                        param_key = self._param_to_key(params)
                        param_positions[param_key] = {
                            'params': params,
                            'position': base_position[[stock_id]].copy()
                        }
                    
                    pbar.update(1)
                except Exception as e:
                    logger.warning(f"策略初始化失敗 {params}: {e}")
                    pbar.update(1)
        
        logger.info(f"✅ 策略位置計算完成，共 {len(param_positions)} 組有效參數")
        
        if not param_positions:
            logger.error(f"股票 {stock_id} 在所有參數下均無交易訊號")
            return None
        
        # 🚀 階段2: 並行執行回測（只做 sim，不做策略計算）
        logger.info(f"⚠️  階段2: 並行執行回測（{len(param_positions)} 個任務，使用 {self.pool} workers）...")
        logger.info("⚠️  預估首個結果 10-20 秒，總計 2-5 分鐘...")
        sys.stdout.flush()
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=self.pool) as executor:
            future_to_params = {
                executor.submit(
                    self._backtest_single_position,
                    stock_id,
                    param_data['position'],
                    param_data['params']
                ): param_data['params']
                for param_data in param_positions.values()
            }
            
            # 使用 tqdm 追蹤進度
            with tqdm(total=len(param_positions), desc="回測進度", unit="param", ncols=100, miniters=1) as pbar:
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
        對所有股票執行參數優化（並行處理，扁平化架構）
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
        
        # 先用默認參數初始化策略，獲取有訊號的股票列表（已通過成交量和法人條件篩選）
        # 使用極大的 SAR max_dots (500) 確保不會遺漏任何可能有訊號的股票
        logger.info("初始化策略以獲取股票列表（使用 SAR max_dots=500 確保涵蓋所有可能）...")
        temp_strategy = OscarTWStrategy(
            sar_max_dots=500,  # 使用極大值確保捕捉所有可能有訊號的股票
            sar_reject_dots=self.sar_reject_dots,
            market_data=market_data
        )
        base_position = temp_strategy.base_position.loc[self.start_date:].copy()
        # 只測試有訊號的股票（已通過成交量和法人條件，且至少有一次訊號）
        all_stocks = base_position.columns[base_position.any(axis=0)].tolist()
        del temp_strategy, base_position  # 釋放記憶體
        logger.info(f"找到 {len(all_stocks)} 檔股票（已通過成交量和三大法人條件篩選，使用寬鬆參數確保不遺漏）")
        
        logger.info(f"開始對 {len(all_stocks)} 檔股票執行參數優化（使用 {self.pool} workers）...")
        
        # 生成參數組合
        param_grid = self.generate_param_grid()
        logger.info(f"每檔股票將測試 {len(param_grid)} 組參數")
        
        # 扁平化：建立所有 (stock, param) 組合
        all_tasks = [(stock, params) for stock in all_stocks for params in param_grid]
        total_tasks = len(all_tasks)
        logger.info(f"總共 {total_tasks} 個任務 ({len(all_stocks)} 股票 × {len(param_grid)} 參數)")
        
        # 檢查是否有檢查點
        checkpoint_name = f"optimize_all_stocks_{datetime.now().strftime('%Y%m%d')}"
        checkpoint_data = self._load_checkpoint(checkpoint_name)
        completed_tasks = set()
        all_param_results = {}
        
        if checkpoint_data:
            logger.info(f"找到檢查點，已完成 {len(checkpoint_data.get('completed', []))} 個任務")
            completed_tasks = set(tuple(t) for t in checkpoint_data.get('completed', []))
            # 重建結果字典
            for stock_id, results in checkpoint_data.get('results', {}).items():
                all_param_results[stock_id] = results
        
        # 過濾掉已完成的任務
        remaining_tasks = [(s, p) for s, p in all_tasks if (s, self._param_to_key(p)) not in completed_tasks]
        logger.info(f"剩餘 {len(remaining_tasks)} 個任務需要完成")
        
        if not remaining_tasks:
            logger.info("所有任務已完成，跳過優化階段")
        else:
            # 🚀 階段1: 預先計算所有參數的策略位置（只計算一次）
            logger.info("=" * 80)
            logger.info("⚠️  階段1: 預先計算所有參數組合的策略位置")
            logger.info("=" * 80)
            
            # 找出需要計算的唯一參數組合
            unique_params = {}
            for stock, params in remaining_tasks:
                param_key = self._param_to_key(params)
                if param_key not in unique_params:
                    unique_params[param_key] = params
            
            logger.info(f"需要計算 {len(unique_params)} 組參數的策略位置")
            logger.info(f"預估需要 5-15 分鐘（取決於機器性能）...")
            sys.stdout.flush()
            
            # 預先計算所有策略位置（在主進程中，方便監控進度）
            param_positions = {}
            with tqdm(total=len(unique_params), desc="🔧 計算策略位置", unit="param", ncols=100, miniters=1) as pbar:
                for param_key, params in unique_params.items():
                    try:
                        strategy = OscarTWStrategy(
                            sar_max_dots=params['sar_max_dots'],
                            sar_reject_dots=self.sar_reject_dots,
                            sar_params=params['sar_params'],
                            macd_params=params['macd_params'],
                            market_data=market_data
                        )
                        base_position = strategy.base_position.loc[self.start_date:].copy()
                        
                        # 保存完整的 base_position（包含所有股票）
                        param_positions[param_key] = {
                            'params': params,
                            'position': base_position
                        }
                        
                        pbar.set_postfix({'SAR': params['sar_max_dots']})
                        pbar.update(1)
                        
                        # 清理策略物件
                        del strategy
                        
                    except Exception as e:
                        logger.warning(f"策略初始化失敗 {param_key}: {e}")
                        pbar.update(1)
            
            logger.info(f"✅ 階段1完成: {len(param_positions)} 組策略位置已計算")
            
            # 🚀 階段2: 並行執行回測（只做 sim，不做策略計算）
            logger.info("=" * 80)
            logger.info("⚠️  階段2: 並行執行回測（輕量級任務）")
            logger.info("=" * 80)
            
            # 建立所有 (stock, param_key) 回測任務
            backtest_tasks = []
            for stock, params in remaining_tasks:
                param_key = self._param_to_key(params)
                if param_key in param_positions:
                    stock_position = param_positions[param_key]['position']
                    # 檢查該股票是否有訊號
                    if stock in stock_position.columns and stock_position[stock].any():
                        backtest_tasks.append({
                            'stock': stock,
                            'param_key': param_key,
                            'params': params,
                            'position': stock_position[[stock]].copy()
                        })
            
            logger.info(f"階段2任務數: {len(backtest_tasks)} 個回測（{len(all_stocks)} 股票 × ~{len(unique_params)} 參數）")
            logger.info(f"使用 {self.pool} 個並行workers，預估 10-30 分鐘...")
            sys.stdout.flush()
            
            # 並行執行所有回測
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            with ProcessPoolExecutor(max_workers=self.pool) as executor:
                # 提交所有回測任務
                future_to_task = {
                    executor.submit(
                        self._backtest_single_position,
                        task['stock'],
                        task['position'],
                        task['params']
                    ): task
                    for task in backtest_tasks
                }
                
                # 使用 tqdm 追蹤進度
                checkpoint_interval = max(1, len(backtest_tasks) // 20)  # 每5%保存一次
                
                with tqdm(total=len(backtest_tasks), desc="🚀 執行回測", unit="test", ncols=100, miniters=1) as pbar:
                    for i, future in enumerate(as_completed(future_to_task)):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            if result:
                                stock_id = result['stock_id']
                                if stock_id not in all_param_results:
                                    all_param_results[stock_id] = []
                                all_param_results[stock_id].append(result)
                                completed_tasks.add((stock_id, task['param_key']))
                                
                                if result['annual_return'] > 0:
                                    pbar.set_postfix({
                                        'stock': stock_id,
                                        'return': f"{result['annual_return']:.2%}"
                                    })
                            
                            pbar.update(1)
                            
                            # 定期保存檢查點
                            if (i + 1) % checkpoint_interval == 0:
                                self._save_checkpoint(checkpoint_name, {
                                    'completed': [[s, p] for s, p in completed_tasks],
                                    'results': all_param_results,
                                    'timestamp': datetime.now().isoformat()
                                })
                                
                        except Exception as e:
                            logger.warning(f"回測失敗 {task['stock']}: {e}")
                            pbar.update(1)
                
                # 最終保存檢查點
                self._save_checkpoint(checkpoint_name, {
                    'completed': [[s, p] for s, p in completed_tasks],
                    'results': all_param_results,
                    'timestamp': datetime.now().isoformat()
                })
            
            logger.info(f"✅ 階段2完成: {len(backtest_tasks)} 個回測已執行")
        
        # 為每檔股票找出最佳參數並生成報告
        results = []
        logger.info("\n開始為每檔股票生成完整報告...")
        with tqdm(total=len(all_stocks), desc="生成報告", unit="stock", ncols=100, miniters=1) as report_pbar:
            for stock_id in all_stocks:
                stock_param_results = all_param_results.get(stock_id, [])
                
                if stock_param_results:
                    best_result = max(stock_param_results, key=lambda x: x['annual_return'])
                    best_result['stock_id'] = stock_id  # 添加股票代碼
                    results.append(best_result)
                    
                    logger.info(f"\n{stock_id} 最佳參數: SAR={best_result['sar_max_dots']} (accel={best_result['sar_accel']:.2f}, max={best_result['sar_maximum']:.2f}), "
                              f"MACD=({best_result['macd_fast']},{best_result['macd_slow']},{best_result['macd_signal']}), "
                              f"年化報酬: {best_result['annual_return']:.2%}")
                    
                    # 為該股票生成完整的報告和視覺化（使用最佳參數重新初始化策略）
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
                    
                    # 清理策略物件釋放記憶體
                    del best_strategy, best_base_position
                else:
                    logger.warning(f"{stock_id} 所有參數組合測試均失敗")
                
                report_pbar.update(1)
        
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
        對每一檔股票執行單獨回測並儲存結果（優化版：使用全局數據避免序列化）
        
        Args:
            base_position: 基礎持倉訊號
            output_path: CSV輸出路徑
            
        Returns:
            pd.DataFrame: 包含所有股票回測結果的 DataFrame
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        # 獲取所有股票
        all_stocks = base_position.columns.tolist()
        
        # 預先過濾：移除完全沒有訊號的股票（快速優化）
        stocks_with_signals = [stock for stock in all_stocks if base_position[stock].any()]
        logger.info(f"股票過濾：{len(all_stocks)} 總數 → {len(stocks_with_signals)} 有訊號")
        
        logger.info(f"開始測試 {len(stocks_with_signals)} 檔股票的個別績效 (使用 {self.pool} workers)")
        
        results = []
        
        # 使用並行處理測試所有股票（base_position傳入全局初始化器，不再序列化）
        logger.info(f"正在初始化 {self.pool} 個工作進程...")
        with ProcessPoolExecutor(max_workers=self.pool, initializer=_init_worker, initargs=(None, base_position)) as executor:
            logger.info("工作進程初始化完成，開始測試...")
            # 提交所有任務（不再傳遞base_position）
            future_to_stock = {
                executor.submit(self._test_single_stock_optimized, stock): stock
                for stock in stocks_with_signals
            }
            
            # 使用 tqdm 追蹤進度
            with tqdm(total=len(stocks_with_signals), desc="測試進度", unit="stock", ncols=100, miniters=1) as pbar:
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
    def _param_to_key(params):
        """將參數字典轉為可哈希的字符串鍵"""
        return f"sar_{params['sar_max_dots']}_{params['sar_params']['acceleration']:.3f}_{params['sar_params']['maximum']:.3f}_macd_{params['macd_params']['fastperiod']}_{params['macd_params']['slowperiod']}_{params['macd_params']['signalperiod']}"
    
    @staticmethod
    def _backtest_single_position(stock_id, position, params):
        """
        對預先計算好的位置執行回測（不重新計算策略）
        
        Args:
            stock_id: 股票代碼
            position: 預先計算好的持倉訊號 DataFrame
            params: 參數字典（用於記錄）
            
        Returns:
            dict: 回測結果
        """
        from finlab.backtest import sim
        
        try:
            # 直接執行回測（位置已預先計算）
            report = sim(
                position=position,
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
            
            # 記錄結果
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
                'params': params
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"回測失敗 ({params}): {e}")
            return None
    
    @staticmethod
    def _test_param_group(params, stock_list, start_date, sar_reject_dots=2):
        """
        測試單組參數下的多個股票（智能緩存：策略位置只計算一次）
        
        Args:
            params: 參數字典
            stock_list: 該參數下需要測試的股票列表
            start_date: 回測開始日期
            sar_reject_dots: SAR拒絕點數
            
        Returns:
            list: 該參數組合下所有股票的回測結果
        """
        from finlab.backtest import sim
        global _WORKER_MARKET_DATA
        
        results = []
        
        try:
            # 初始化策略（只計算一次！）
            strategy = OscarTWStrategy(
                sar_max_dots=params['sar_max_dots'],
                sar_reject_dots=sar_reject_dots,
                sar_params=params['sar_params'],
                macd_params=params['macd_params'],
                market_data=_WORKER_MARKET_DATA
            )
            
            base_position = strategy.base_position.loc[start_date:].copy()
            
            # 對該參數下的每個股票執行回測（使用同一個位置DataFrame）
            for stock_id in stock_list:
                try:
                    # 快速過濾：如果該股票完全沒有訊號，跳過
                    if stock_id not in base_position.columns or not base_position[stock_id].any():
                        continue
                    
                    # 建立單一股票的持倉訊號（只複製該列）
                    single_stock_position = base_position[[stock_id]].copy()
                    
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
                    
                    # 記錄結果
                    result = {
                        'stock_id': stock_id,
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
                        'params': params
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    # 單個股票失敗不影響其他股票
                    pass
            
            return results
            
        except Exception as e:
            logger.warning(f"參數組合測試失敗 ({params}): {e}")
            return []
    
    @staticmethod
    def _test_single_param(stock_id, params, start_date, sar_reject_dots=2):
        """
        測試單一參數組合 (靜態方法供並行處理使用)
        使用全局 _WORKER_MARKET_DATA 避免昂貴的序列化
        
        Args:
            stock_id: 股票代碼
            params: 參數字典
            start_date: 回測開始日期
            sar_reject_dots: SAR拒絕點數
            
        Returns:
            dict: 單一參數組合的回測結果
        """
        from finlab.backtest import sim
        global _WORKER_MARKET_DATA
        
        try:
            # 初始化策略（使用全局預載數據）
            strategy = OscarTWStrategy(
                sar_max_dots=params['sar_max_dots'],
                sar_reject_dots=sar_reject_dots,
                sar_params=params['sar_params'],
                macd_params=params['macd_params'],
                market_data=_WORKER_MARKET_DATA
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
    def _test_single_stock_optimized(stock, fee_ratio=0.001425, tax_ratio=0.003):
        """
        測試單一股票（優化版：使用全局base_position避免序列化）
        
        Args:
            stock: 股票代碼
            fee_ratio: 手續費率
            tax_ratio: 證交稅率
            
        Returns:
            dict: 單一股票的回測結果
        """
        from finlab.backtest import sim
        global _WORKER_BASE_POSITION
        
        try:
            # 使用全局變量避免每次任務都序列化整個DataFrame
            base_position = _WORKER_BASE_POSITION
            
            # 快速過濾：如果該股票完全沒有訊號，立即返回
            if not base_position[stock].any():
                return None
            
            # 建立單一股票的持倉訊號（只複製該列，不是整個DataFrame）
            single_stock_position = base_position[[stock]].copy()
            
            # 執行回測
            report = sim(
                position=single_stock_position,
                resample=None,  # 保持原始訊號時間對齊
                upload=False,
                market=AdjustTWMarketInfo(),
                fee_ratio=fee_ratio,
                tax_ratio=tax_ratio,
                position_limit=1.0
            )
            # 移除HTML生成（I/O瓶頸）- 只在需要時單獨生成
            
            # 提取績效指標
            metrics = report.get_metrics()
            trades = report.get_trades()
            
            # 整理結果 (只保留關鍵指標，減少內存占用)
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
                'holding_days': base_position[stock].sum(),
            }
            
            return result
            
        except Exception as e:
            # 靜默失敗，返回 None
            return None
    
    @staticmethod
    def _test_single_stock(stock, base_position, fee_ratio=0.001425, tax_ratio=0.003):
        """
        測試單一股票（舊版：保留用於向後兼容）
        新代碼請使用 _test_single_stock_optimized
        """
        return SingleStockTestExecutor._test_single_stock_optimized(stock, fee_ratio, tax_ratio)
    
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
    parser.add_argument('--pool', type=int, default=None, help='並行處理的worker數量（默認自動：<100核心用cores-2，>=100核心用75%%。建議144核機器用100-110）')
    parser.add_argument('--stock_id', type=str, default=None, help='指定測試單一股票（可選）')
    parser.add_argument('--optimize', action='store_true', help='啟用參數優化（僅在指定stock_id時有效）')
    
    args = parser.parse_args()
    
    # Auto-detect optimal pool size if not specified
    if args.pool is None:
        import os
        cpu_count = os.cpu_count() or 4
        # For high-core-count machines (>100 cores), use 70-75% to avoid context switching
        # For normal machines, use cores - 2
        if cpu_count > 100:
            args.pool = int(cpu_count * 0.75)  # e.g., 144 cores → 108 workers
            logger.info(f"自動偵測到高核心數機器 ({cpu_count} 核心)，設定並行處理數為 {args.pool} (~75%)")
        else:
            args.pool = max(1, cpu_count - 2)  # Leave 2 cores for system
            logger.info(f"自動偵測到 {cpu_count} 個CPU核心，設定並行處理數為 {args.pool}")
    
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
