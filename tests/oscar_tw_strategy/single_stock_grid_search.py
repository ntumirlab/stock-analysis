"""
Single Stock Grid Search Executor

測試策略在每一檔個股上的表現

Performance Optimizations (針對144核心機器):
1. Global worker initialization (_init_worker) - Market data loaded once per worker
2. Global base_position in workers - Eliminates 5-10x serialization overhead
3. Flattened task architecture - No nested ProcessPoolExecutor loops
4. Pre-filtering - Skip stocks with zero signals before submission
5. Removed HTML generation from parallel loop - Disk I/O bottleneck eliminated
6. Auto-detected optimal pool size - Prevents context switching (default: CPU cores - 2)
7. Two-stage optimization - Pre-calculate strategies once, run backtests in parallel
8. In-memory position caching - ~11GB for 729 params on 128-256GB machines
9. Smart task attribution - Uses task metadata to avoid KeyError on result processing

Expected Performance:
- Before: 25% CPU usage, week-long runtime with --pool 144
- After: 90-95% CPU usage, ~50-100x faster with --pool 100-110
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

from strategy_class.oscar.oscar_strategy_andor import (
    OscarAndOrStrategy,
    AdjustTWMarketInfo,
)
from tests.oscar_tw_strategy.utils.drawing_overall_html import (
    dataframe_to_sortable_html,
)
from tests.oscar_tw_strategy.utils.drawing_history_visualization import (
    create_trading_visualization,
    prepare_price_data,
)
from tests.oscar_tw_strategy.utils.drawing_param_comparison import (
    create_param_comparison_chart,
)
from tests.oscar_tw_strategy.utils.custom_report_metrics import (
    compute_total_reward_amount_from_creturn,
    get_metrics_with_fixed_annual_return,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for worker processes (to avoid expensive pickling)
_WORKER_MARKET_DATA = None
_WORKER_BASE_POSITION = None
_WORKER_POSITION_CACHE = {}  # Cache for pre-calculated positions by param hash
_WORKER_INITIAL_CAPITAL = 100_000.0


def _create_oscar_strategy_compat(
    sar_signal_lag_min=0,
    sar_signal_lag_max=2,
    sar_params=None,
    macd_params=None,
    market_data=None,
    **extra_kwargs,
):
    """Create Oscar AND/OR strategy instance."""
    shared_kwargs = {}
    if sar_params is not None:
        shared_kwargs["sar_params"] = sar_params
    if macd_params is not None:
        shared_kwargs["macd_params"] = macd_params
    if market_data is not None:
        shared_kwargs["market_data"] = market_data
    shared_kwargs.update(extra_kwargs)

    try:
        return OscarAndOrStrategy(
            sar_signal_lag_min=sar_signal_lag_min,
            sar_signal_lag_max=sar_signal_lag_max,
            **shared_kwargs,
        )
    except TypeError:
        raise


def _init_worker(market_data_path=None, base_position=None, initial_capital=100_000.0):
    """Initialize worker process with shared data (called once per worker)"""
    import pickle

    global _WORKER_MARKET_DATA, _WORKER_BASE_POSITION, _WORKER_POSITION_CACHE, _WORKER_INITIAL_CAPITAL

    # Load market_data from disk (avoids serialization overhead and redundant API calls)
    if market_data_path and os.path.exists(market_data_path):
        with open(market_data_path, "rb") as f:
            _WORKER_MARKET_DATA = pickle.load(f)
    else:
        _WORKER_MARKET_DATA = None

    _WORKER_BASE_POSITION = base_position
    _WORKER_POSITION_CACHE = {}  # Each worker has its own cache
    _WORKER_INITIAL_CAPITAL = float(initial_capital)


def _result_ranking_key(result):
    sharpe_ratio = result.get("sharpe_ratio")
    sharpe_value = float("-inf") if sharpe_ratio is None else float(sharpe_ratio)
    return (float(result["total_reward_amount"]), sharpe_value)


class SingleStockGridSearchExecutor:
    """執行單一股票網格搜尋"""

    def __init__(
        self,
        start_date="2020-01-01",
        end_date=None,
        output_dir="results/single_stock_tests",
        sar_signal_lag_min=0,
        sar_signal_lag_max=2,
        pool=20,
        stock_id=None,
        optimize_params=False,
        initial_capital=100_000,
    ):
        """
        初始化測試執行器

        Args:
            start_date: 回測起始日期
            end_date: 回測結束日期（可選，預設為最新價格日期）
            output_dir: 結果輸出目錄
            sar_signal_lag_min: SAR翻多與MACD黃金交叉的最小允許天數差
            sar_signal_lag_max: SAR翻多與MACD黃金交叉的最大允許天數差
            pool: 並行處理的worker數量
            stock_id: 指定測試單一股票（可選）
            optimize_params: 是否進行參數優化（僅在指定stock_id時有效）
        """
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.position_cache_dir = self.output_dir / "position_cache"
        self.position_cache_dir.mkdir(parents=True, exist_ok=True)

        # Stage 2 結果目錄（依日期分組）
        if end_date:
            self.stage2_dir = self.output_dir / f"stage_2_results_{end_date}"
        else:
            # 如果沒指定end_date，使用臨時標記（稍後在載入市場數據後會更新為實際日期）
            self.stage2_dir = self.output_dir / "stage_2_results_latest"
        self.stage2_dir.mkdir(parents=True, exist_ok=True)

        self.sar_signal_lag_min = sar_signal_lag_min
        self.sar_signal_lag_max = sar_signal_lag_max
        self.pool = pool
        self.stock_id = stock_id
        self.optimize_params = optimize_params
        self.initial_capital = float(initial_capital)

        # Warn if pool size is too large
        import os

        cpu_count = os.cpu_count() or 4
        if self.pool > cpu_count:
            logger.warning(
                f"⚠️  並行處理數 ({self.pool}) 超過CPU核心數 ({cpu_count})，可能導致效能下降！"
            )
            logger.warning(f"⚠️  建議設定 --pool {cpu_count} 或更小的值")

        logger.info(f"初始化 SingleStockGridSearchExecutor")
        logger.info(f"回測起始日期: {start_date}")
        logger.info(f"回測結束日期: {end_date if end_date else '最新價格日期（自動）'}")
        logger.info(
            f"SAR-MACD時間窗參數: lag_min={sar_signal_lag_min}, lag_max={sar_signal_lag_max}"
        )
        logger.info(f"Stage 1 結果目錄: {self.output_dir}")
        logger.info(f"Stage 2 結果目錄: {self.stage2_dir}")
        if stock_id:
            logger.info(f"單一股票網格搜尋模式: {stock_id}")
            if optimize_params:
                logger.info(f"參數優化模式已啟用")
        else:
            logger.info(f"全市場測試模式，並行處理數: {pool} workers")

    def _save_checkpoint(self, checkpoint_name, data):
        """保存檢查點"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(data, f)
        logger.info(f"檢查點已保存: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_name):
        """載入檢查點"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
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

        # SAR/MACD 對齊時間窗參數
        # lag_window: (min_lag, max_lag)
        # acceleration: 0.01, 0.02, 0.03 (加速因子)
        # maximum: 0.15, 0.2, 0.25 (最大加速因子)

        # MACD 參數
        # fast: 10, 12, 14
        # slow: 24, 26, 28
        # signal: 8, 9, 10

        # 保持與歷史相近的組合數量，避免任務量爆增
        sar_lag_windows = [(0, 2), (0, 3), (1, 3)]
        sar_acceleration_values = [0.01, 0.02, 0.03]
        sar_maximum_values = [0.15, 0.2, 0.25]

        macd_fast_values = [10, 12, 14]
        macd_slow_values = [24, 26, 28]
        macd_signal_values = [8, 9, 10]

        for lag_min, lag_max in sar_lag_windows:
            for sar_accel in sar_acceleration_values:
                for sar_max_val in sar_maximum_values:
                    for macd_fast in macd_fast_values:
                        for macd_slow in macd_slow_values:
                            for macd_signal in macd_signal_values:
                                if macd_fast < macd_slow:  # 確保 fast < slow
                                    param_combinations.append(
                                        {
                                            "sar_signal_lag_min": lag_min,
                                            "sar_signal_lag_max": lag_max,
                                            "sar_params": {
                                                "acceleration": sar_accel,
                                                "maximum": sar_max_val,
                                            },
                                            "macd_params": {
                                                "fastperiod": macd_fast,
                                                "slowperiod": macd_slow,
                                                "signalperiod": macd_signal,
                                            },
                                        }
                                    )

        logger.info(f"生成了 {len(param_combinations)} 組參數組合")
        return param_combinations

    def run_test(self):
        """執行測試"""
        logger.info("=" * 60)
        logger.info("開始執行單一股票網格搜尋")
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
                strategy = _create_oscar_strategy_compat(
                    sar_signal_lag_min=self.sar_signal_lag_min,
                    sar_signal_lag_max=self.sar_signal_lag_max,
                )
                # ⚠️ 重要：確保 position 從 start_date 開始
                base_position = strategy.base_position
                trading_days = (
                    strategy.market_data["close"].loc[self.start_date :].index
                )
                base_position = base_position.reindex(trading_days, fill_value=False)

                logger.info(f"執行單一股票回測: {self.stock_id}")
                result = self._run_single_stock_with_visualization(
                    stock_id=self.stock_id,
                    strategy=strategy,
                    base_position=base_position,
                )
            else:
                # 全市場模式：對所有股票執行回測並生成 HTML 表格
                logger.info("初始化策略並生成買賣訊號...")
                strategy = _create_oscar_strategy_compat(
                    sar_signal_lag_min=self.sar_signal_lag_min,
                    sar_signal_lag_max=self.sar_signal_lag_max,
                )
                # ⚠️ 重要：確保 position 從 start_date 開始
                base_position = strategy.base_position
                trading_days = (
                    strategy.market_data["close"].loc[self.start_date :].index
                )
                base_position = base_position.reindex(trading_days, fill_value=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"oscar_single_stock_results_{timestamp}.csv"
                html_filename = f"oscar_single_stock_results_{timestamp}.html"
                csv_path = self.output_dir / csv_filename
                html_path = self.output_dir / html_filename

                logger.info("開始對每一檔股票執行回測...")
                results_df = self._run_tests_and_save(
                    base_position=base_position, output_path=str(csv_path)
                )

                # 生成互動式 HTML 表格
                logger.info("生成互動式 HTML 表格...")
                dataframe_to_sortable_html(
                    df=results_df,
                    output_path=str(html_path),
                    title=f"Oscar Strategy - Single Stock Results ({timestamp})",
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
        market_data = OscarAndOrStrategy.load_market_data()
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
        with tqdm(
            total=len(param_grid),
            desc="計算策略位置",
            unit="param",
            ncols=100,
            miniters=1,
        ) as pbar:
            for params in param_grid:
                try:
                    strategy = _create_oscar_strategy_compat(
                        sar_signal_lag_min=params["sar_signal_lag_min"],
                        sar_signal_lag_max=params["sar_signal_lag_max"],
                        sar_params=params["sar_params"],
                        macd_params=params["macd_params"],
                        market_data=market_data,
                    )
                    # ⚠️ 重要：確保 position 從 start_date 開始
                    base_position = strategy.base_position
                    trading_days = market_data["close"].loc[self.start_date :].index
                    base_position = base_position.reindex(
                        trading_days, fill_value=False
                    )

                    # 只保存該股票的位置（節省記憶體）
                    if (
                        stock_id in base_position.columns
                        and base_position[stock_id].any()
                    ):
                        param_key = self._param_to_key(params)
                        param_positions[param_key] = {
                            "params": params,
                            "position": base_position[[stock_id]].copy(),
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
        logger.info(
            f"⚠️  階段2: 並行執行回測（{len(param_positions)} 個任務，使用 {self.pool} workers）..."
        )
        logger.info("⚠️  預估首個結果 10-20 秒，總計 2-5 分鐘...")
        sys.stdout.flush()

        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=self.pool) as executor:
            future_to_params = {
                executor.submit(
                    self._backtest_single_position,
                    stock_id,
                    param_data["position"],
                    param_data["params"],
                    self.initial_capital,
                ): param_data["params"]
                for param_data in param_positions.values()
            }

            # 使用 tqdm 追蹤進度
            with tqdm(
                total=len(param_positions),
                desc="回測進度",
                unit="param",
                ncols=100,
                miniters=1,
            ) as pbar:
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        result = future.result()
                        if result:
                            param_results.append(result)
                            pbar.set_postfix(
                                {
                                    "LagWin": f"{params['sar_signal_lag_min']}-{params['sar_signal_lag_max']}",
                                            "reward": f"{result['total_reward_amount']:.2f}",
                                }
                            )
                    except Exception as e:
                        logger.warning(f"參數組合測試失敗: {e}")
                    finally:
                        pbar.update(1)

        if not param_results:
            logger.error("所有參數組合測試均失敗")
            return None

        # 找出最佳結果
        best_result = max(param_results, key=_result_ranking_key)

        # 輸出最佳參數資訊
        logger.info("\n" + "=" * 60)
        logger.info("最佳參數組合")
        logger.info("=" * 60)
        logger.info(
            f"SAR lag window: {best_result['sar_signal_lag_min']} ~ {best_result['sar_signal_lag_max']}"
        )
        logger.info(
            f"SAR accel: {best_result['sar_accel']:.2f}, maximum: {best_result['sar_maximum']:.2f}"
        )
        logger.info(
            f"MACD: fast={best_result['macd_fast']}, slow={best_result['macd_slow']}, signal={best_result['macd_signal']}"
        )
        logger.info(f"總報酬金額: {best_result['total_reward_amount']:.2f}")
        logger.info(f"年化報酬率: {best_result['annual_return']:.2%}")
        logger.info(f"最大回檔: {best_result['max_drawdown']:.2%}")
        logger.info(
            f"夏普比率: {best_result['sharpe_ratio']:.2f}"
            if best_result["sharpe_ratio"]
            else "夏普比率: N/A"
        )
        logger.info(f"交易次數: {best_result['total_trades']}")
        logger.info("=" * 60)

        # 使用最佳參數重新初始化策略並生成完整的視覺化
        logger.info("使用最佳參數重新初始化策略並生成視覺化...")
        best_params = best_result["params"]
        best_strategy = _create_oscar_strategy_compat(
            sar_signal_lag_min=best_params["sar_signal_lag_min"],
            sar_signal_lag_max=best_params["sar_signal_lag_max"],
            sar_params=best_params["sar_params"],
            macd_params=best_params["macd_params"],
        )
        # ⚠️ 重要：確保 position 從 start_date 開始
        best_base_position = best_strategy.base_position
        trading_days = best_strategy.market_data["close"].loc[self.start_date :].index
        best_base_position = best_base_position.reindex(trading_days, fill_value=False)

        final_result = self._run_single_stock_with_visualization(
            stock_id=stock_id, strategy=best_strategy, base_position=best_base_position
        )

        # 生成參數比較圖表（保存到 Stage 2 目錄）
        param_comparison_path = self.stage2_dir / f"{stock_id}_param_comparison.html"
        create_param_comparison_chart(
            stock_id=stock_id,
            param_results=param_results,
            output_path=str(param_comparison_path),
        )
        logger.info(f"Stage 2 參數比較圖表已儲存至: {param_comparison_path}")

        # 添加最佳參數資訊到結果
        final_result["best_params"] = best_result["params"]
        final_result["param_comparison_path"] = str(param_comparison_path)

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
        market_data = OscarAndOrStrategy.load_market_data()
        logger.info("市場數據載入完成")

        # 獲取實際的最後交易日（如果未指定end_date）
        if not self.end_date:
            actual_end_date = market_data["close"].index.max().strftime("%Y-%m-%d")
            # 更新Stage 2目錄為實際日期
            self.stage2_dir = self.output_dir / f"stage_2_results_{actual_end_date}"
            self.stage2_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"未指定end_date，使用實際最後交易日: {actual_end_date}")
        else:
            actual_end_date = self.end_date

        # 先用寬鬆時間窗初始化策略，獲取有訊號的股票列表（已通過成交量和法人條件篩選）
        logger.info(
            "初始化策略以獲取股票列表（使用寬鬆 lag window 確保涵蓋所有可能）..."
        )
        temp_strategy = _create_oscar_strategy_compat(
            sar_signal_lag_min=0,
            sar_signal_lag_max=5,
            market_data=market_data,
        )
        # ⚠️ 重要：確保 position 從 start_date 開始
        base_position = temp_strategy.base_position
        trading_days = market_data["close"].loc[self.start_date :].index
        base_position = base_position.reindex(trading_days, fill_value=False)
        # 只測試有訊號的股票（已通過成交量和法人條件，且至少有一次訊號）
        all_stocks = base_position.columns[base_position.any(axis=0)].tolist()
        del temp_strategy, base_position  # 釋放記憶體
        logger.info(
            f"找到 {len(all_stocks)} 檔股票（已通過成交量和三大法人條件篩選，使用寬鬆參數確保不遺漏）"
        )

        logger.info(
            f"開始對 {len(all_stocks)} 檔股票執行參數優化（使用 {self.pool} workers）..."
        )

        # 生成參數組合
        param_grid = self.generate_param_grid()
        logger.info(f"每檔股票將測試 {len(param_grid)} 組參數")

        # 扁平化：建立所有 (stock, param) 組合
        all_tasks = [(stock, params) for stock in all_stocks for params in param_grid]
        total_tasks = len(all_tasks)
        logger.info(
            f"總共 {total_tasks} 個任務 ({len(all_stocks)} 股票 × {len(param_grid)} 參數)"
        )

        # 檢查是否有檢查點（使用固定名稱，跨日期持久化）
        checkpoint_name = "optimize_all_stocks_persistent"
        checkpoint_data = self._load_checkpoint(checkpoint_name)
        completed_tasks = set()
        all_param_results = {}

        if checkpoint_data:
            logger.info(
                f"📂 找到檢查點，已完成 {len(checkpoint_data.get('completed', []))} 個任務"
            )
            completed_tasks = set(
                tuple(t) for t in checkpoint_data.get("completed", [])
            )
            # 重建結果字典
            for stock_id, results in checkpoint_data.get("results", {}).items():
                all_param_results[stock_id] = results
        else:
            logger.info("🆕 未找到檢查點，開始新的優化")

        # 過濾掉已完成的任務
        remaining_tasks = [
            (s, p)
            for s, p in all_tasks
            if (s, self._param_to_key(p)) not in completed_tasks
        ]
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

            # 檢查位置快取（避免重複計算）
            import pickle

            cache_meta_file = self.position_cache_dir / "cache_metadata.json"
            param_positions = {}

            # 判斷是否可使用快取 (智能重用：如果快取數據涵蓋所需範圍，可以裁切使用)
            use_cache = False
            need_slice = False
            cached_end_date = None
            if cache_meta_file.exists():
                try:
                    with open(cache_meta_file, "r") as f:
                        cache_meta = json.load(f)

                    cached_end_date = cache_meta.get("end_date")

                    # 檢查關鍵參數是否匹配
                    if cache_meta.get(
                        "start_date"
                    ) == self.start_date and cache_meta.get("param_count") == len(
                        unique_params
                    ):
                        # 智能比對 end_date:
                        # 1. 快取 None, 請求 None → 完全匹配
                        # 2. 快取 None, 請求 某日期 → 可用，需裁切
                        # 3. 快取 某日期, 請求 None → 不可用（快取數據不足）
                        # 4. 快取 某日期, 請求 某日期 → 檢查快取 >= 請求

                        if cached_end_date is None and self.end_date is None:
                            # 完全匹配
                            use_cache = True
                            logger.info(
                                f"💾 找到位置快取 (建立於 {cache_meta.get('timestamp')})"
                            )
                            logger.info(
                                f"💾 快取參數完全匹配，將載入已計算的策略位置..."
                            )
                        elif cached_end_date is None and self.end_date is not None:
                            # 快取包含全部數據，可裁切使用
                            use_cache = True
                            need_slice = True
                            logger.info(
                                f"💾 找到位置快取 (建立於 {cache_meta.get('timestamp')})"
                            )
                            logger.info(f"💾 快取數據完整，將裁切至 {self.end_date}")
                        elif cached_end_date is not None and self.end_date is None:
                            # 快取數據不足，無法使用
                            logger.info(
                                f"⚠️  快取 end_date={cached_end_date}，但需要全部數據，快取不可用"
                            )
                        elif cached_end_date >= self.end_date:
                            # 快取數據涵蓋所需範圍
                            use_cache = True
                            need_slice = cached_end_date != self.end_date
                            logger.info(
                                f"💾 找到位置快取 (建立於 {cache_meta.get('timestamp')})"
                            )
                            if need_slice:
                                logger.info(
                                    f"💾 快取數據充足 ({cached_end_date})，將裁切至 {self.end_date}"
                                )
                            else:
                                logger.info(
                                    f"💾 快取參數完全匹配，將載入已計算的策略位置..."
                                )
                        else:
                            # 快取數據不足
                            logger.info(
                                f"⚠️  快取 end_date={cached_end_date} < 請求 {self.end_date}，快取數據不足"
                            )

                except Exception as e:
                    logger.warning(f"讀取快取元資料失敗: {e}")

            if use_cache:
                # 驗證快取完整性（不載入到記憶體，避免 11GB 佔用和 swap）
                logger.info(f"📂 驗證 {len(unique_params)} 組快取檔案...")
                loaded_count = 0
                need_resave = False  # 如果需要裁切，標記需要重新保存

                with tqdm(
                    total=len(unique_params),
                    desc="📂 驗證快取",
                    unit="param",
                    ncols=100,
                ) as pbar:
                    for param_key, params in unique_params.items():
                        cache_file = self.position_cache_dir / f"{param_key}.pkl"
                        if cache_file.exists():
                            # 如果需要裁切，重新保存裁切後的版本
                            if need_slice and self.end_date:
                                try:
                                    with open(cache_file, "rb") as f:
                                        base_position = pickle.load(f)
                                    base_position = base_position.loc[
                                        : self.end_date
                                    ].copy()
                                    # 重新保存裁切後的版本
                                    with open(cache_file, "wb") as f:
                                        pickle.dump(
                                            base_position,
                                            f,
                                            protocol=pickle.HIGHEST_PROTOCOL,
                                        )
                                    del base_position  # 立即釋放
                                    need_resave = True
                                except Exception as e:
                                    logger.warning(f"裁切快取失敗 {param_key}: {e}")
                                    pbar.update(1)
                                    continue

                            loaded_count += 1
                        pbar.update(1)

                # 建立參數映射（不含位置數據）
                param_positions = {
                    k: {"params": v, "position": None} for k, v in unique_params.items()
                }

                if loaded_count == len(unique_params):
                    if need_slice:
                        if need_resave:
                            logger.info(
                                f"✅ 快取驗證完成並重新保存: {loaded_count}/{len(unique_params)} 組，已裁切至 {self.end_date}（節省 5-15 分鐘）"
                            )
                        else:
                            logger.info(
                                f"✅ 快取驗證完成: {loaded_count}/{len(unique_params)} 組（節省 5-15 分鐘）"
                            )
                    else:
                        logger.info(
                            f"✅ 快取驗證完成: {loaded_count}/{len(unique_params)} 組（節省 5-15 分鐘）"
                        )
                else:
                    logger.warning(
                        f"⚠️  快取不完整: {loaded_count}/{len(unique_params)}，將重新計算缺失部分"
                    )
                    use_cache = False  # 快取不完整，重新計算

            if not use_cache:
                # 重新計算所有策略位置（並行處理）
                logger.info(f"⚡ 使用 {self.pool} workers 並行計算策略位置...")
                logger.info(f"預估需要 1-3 分鐘（並行加速，取決於機器性能）...")
                sys.stdout.flush()

                # 並行計算所有策略位置
                param_count = 0
                saved_params = {}

                from concurrent.futures import ProcessPoolExecutor, as_completed

                # 保存 market_data 到磁盤（避免序列化開銷和重複 API 調用）
                market_data_file = self.position_cache_dir / "_market_data.pkl"
                logger.info(f"保存市場數據到緩存: {market_data_file}")
                with open(market_data_file, "wb") as f:
                    pickle.dump(market_data, f)

                try:
                    # 初始化 workers（從磁盤載入市場數據）
                    logger.info(f"正在初始化 {self.pool} 個工作進程...")
                    with ProcessPoolExecutor(
                        max_workers=self.pool,
                        initializer=_init_worker,
                        initargs=(str(market_data_file), None),
                    ) as executor:
                        logger.info("工作進程初始化完成，開始並行計算...")

                        # 提交所有策略計算任務
                        future_to_param = {
                            executor.submit(
                                self._calculate_and_save_strategy,
                                param_key,
                                params,
                                self.start_date,
                                self.end_date,
                                str(self.position_cache_dir),
                            ): (param_key, params)
                            for param_key, params in unique_params.items()
                        }

                        # 使用 tqdm 追蹤進度
                        with tqdm(
                            total=len(unique_params),
                            desc="🔧 計算策略位置",
                            unit="param",
                            ncols=100,
                            miniters=1,
                        ) as pbar:
                            for future in as_completed(future_to_param):
                                param_key, params = future_to_param[future]
                                try:
                                    result_key, success, error = future.result()
                                    if success:
                                        saved_params[result_key] = params
                                        param_count += 1
                                        pbar.set_postfix(
                                            {
                                                "LagWin": f"{params['sar_signal_lag_min']}-{params['sar_signal_lag_max']}",
                                                "saved": param_count,
                                            }
                                        )
                                    else:
                                        logger.warning(
                                            f"策略計算失敗 {result_key}: {error}"
                                        )
                                except Exception as e:
                                    logger.warning(f"策略計算異常 {param_key}: {e}")
                                finally:
                                    pbar.update(1)

                    # 保存快取元資料
                    cache_metadata = {
                        "start_date": self.start_date,
                        "end_date": self.end_date,
                        "param_count": param_count,
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(cache_meta_file, "w") as f:
                        json.dump(cache_metadata, f, indent=2)

                    logger.info(
                        f"✅ 階段1完成: {param_count} 組策略位置已保存至磁盤（並行計算，記憶體已釋放）"
                    )
                    logger.info(f"💾 位置快取已保存至: {self.position_cache_dir}")

                finally:
                    # 清理臨時市場數據文件
                    if market_data_file.exists():
                        market_data_file.unlink()
                        logger.info(f"已清理臨時市場數據文件")

                # 用 saved_params 替換 param_positions（為 Stage 2 準備）
                param_positions = {
                    k: {"params": v, "position": None} for k, v in saved_params.items()
                }

            # 🚀 階段2: 並行執行回測（只做 sim，不做策略計算）
            logger.info("=" * 80)
            logger.info("⚠️  階段2: 並行執行回測（輕量級任務）")
            logger.info("=" * 80)

            # 建立所有 (stock, param_key) 回測任務
            # ⚡ 從磁盤按需載入位置（避免同時保留 11GB 在記憶體）
            logger.info("📂 準備回測任務（從磁盤載入位置）...")
            backtest_tasks = []
            for stock, params in remaining_tasks:
                param_key = self._param_to_key(params)
                if param_key in param_positions:
                    # 從磁盤快取載入該參數的位置
                    cache_file = self.position_cache_dir / f"{param_key}.pkl"
                    if cache_file.exists():
                        try:
                            import pickle

                            with open(cache_file, "rb") as f:
                                stock_position = pickle.load(f)

                            # 檢查該股票是否有訊號
                            if (
                                stock in stock_position.columns
                                and stock_position[stock].any()
                            ):
                                backtest_tasks.append(
                                    {
                                        "stock": stock,
                                        "param_key": param_key,
                                        "params": params,
                                        "position": stock_position[
                                            [stock]
                                        ].copy(),  # Only copy single column
                                    }
                                )
                            del stock_position  # 立即釋放記憶體
                        except Exception as e:
                            logger.warning(f"載入快取失敗 {param_key}: {e}")

            logger.info(
                f"階段2任務數: {len(backtest_tasks)} 個回測（{len(all_stocks)} 股票 × ~{len(unique_params)} 參數）"
            )
            logger.info(f"使用 {self.pool} 個並行workers，預估 10-30 分鐘...")
            sys.stdout.flush()

            # 並行執行所有回測
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=self.pool) as executor:
                # 提交所有回測任務
                future_to_task = {
                    executor.submit(
                        self._backtest_single_position,
                        task["stock"],
                        task["position"],
                        task["params"],
                        self.initial_capital,
                    ): task
                    for task in backtest_tasks
                }

                # 使用 tqdm 追蹤進度
                checkpoint_interval = max(1, len(backtest_tasks) // 20)  # 每5%保存一次

                with tqdm(
                    total=len(backtest_tasks),
                    desc="🚀 執行回測",
                    unit="test",
                    ncols=100,
                    miniters=1,
                ) as pbar:
                    for i, future in enumerate(as_completed(future_to_task)):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            if result:
                                # ✅ 使用task['stock']作為來源（更安全，避免partial/legacy結果遺失歸屬）
                                stock_id = task["stock"]
                                if stock_id not in all_param_results:
                                    all_param_results[stock_id] = []
                                all_param_results[stock_id].append(result)
                                completed_tasks.add((stock_id, task["param_key"]))

                                if result["annual_return"] > 0:
                                    pbar.set_postfix(
                                        {
                                            "stock": stock_id,
                                            "reward": f"{result['total_reward_amount']:.2f}",
                                        }
                                    )

                            pbar.update(1)

                            # 定期保存檢查點
                            if (i + 1) % checkpoint_interval == 0:
                                self._save_checkpoint(
                                    checkpoint_name,
                                    {
                                        "completed": [
                                            [s, p] for s, p in completed_tasks
                                        ],
                                        "results": all_param_results,
                                        "timestamp": datetime.now().isoformat(),
                                    },
                                )

                        except Exception as e:
                            logger.warning(f"回測失敗 {task['stock']}: {e}")
                            pbar.update(1)

                # 最終保存檢查點
                self._save_checkpoint(
                    checkpoint_name,
                    {
                        "completed": [[s, p] for s, p in completed_tasks],
                        "results": all_param_results,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            logger.info(f"✅ 階段2完成: {len(backtest_tasks)} 個回測已執行")

            # 清理 param_positions 釋放記憶體
            del param_positions
            logger.info("🗑️  已清理策略位置快取")

        # 為每檔股票找出最佳參數並生成報告
        results = []
        logger.info("\n開始為每檔股票生成完整報告...")
        with tqdm(
            total=len(all_stocks), desc="生成報告", unit="stock", ncols=100, miniters=1
        ) as report_pbar:
            for stock_id in all_stocks:
                stock_param_results = all_param_results.get(stock_id, [])

                if stock_param_results:
                    best_result = max(stock_param_results, key=_result_ranking_key)
                    best_result["stock_id"] = stock_id  # 添加股票代碼
                    results.append(best_result)

                    logger.info(
                        f"\n{stock_id} 最佳參數: SAR lag={best_result['sar_signal_lag_min']}~{best_result['sar_signal_lag_max']} "
                        f"(accel={best_result['sar_accel']:.2f}, max={best_result['sar_maximum']:.2f}), "
                        f"MACD=({best_result['macd_fast']},{best_result['macd_slow']},{best_result['macd_signal']}), "
                        f"總報酬金額: {best_result['total_reward_amount']:.2f}, "
                        f"年化報酬: {best_result['annual_return']:.2%}"
                    )

                    # 為該股票生成完整的報告和視覺化（使用最佳參數重新初始化策略）
                    best_params = best_result["params"]
                    best_strategy = _create_oscar_strategy_compat(
                        sar_signal_lag_min=best_params["sar_signal_lag_min"],
                        sar_signal_lag_max=best_params["sar_signal_lag_max"],
                        sar_params=best_params["sar_params"],
                        macd_params=best_params["macd_params"],
                        market_data=market_data,
                    )
                    # ⚠️ 重要：確保 position 從 start_date 開始
                    best_base_position = best_strategy.base_position
                    trading_days = market_data["close"].loc[self.start_date :].index
                    best_base_position = best_base_position.reindex(
                        trading_days, fill_value=False
                    )

                    # 生成報告和視覺化
                    self._run_single_stock_with_visualization(
                        stock_id=stock_id,
                        strategy=best_strategy,
                        base_position=best_base_position,
                    )

                    # 生成參數比較圖表（保存到 Stage 2 目錄）
                    param_comparison_path = (
                        self.stage2_dir / f"{stock_id}_param_comparison.html"
                    )
                    create_param_comparison_chart(
                        stock_id=stock_id,
                        param_results=stock_param_results,
                        output_path=str(param_comparison_path),
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
        column_order = [
            "stock_id",
            "total_reward_amount",
            "annual_return",
            "max_drawdown",
            "sharpe_ratio",
            "total_trades",
            "sar_signal_lag_min",
            "sar_signal_lag_max",
            "sar_accel",
            "sar_maximum",
            "macd_fast",
            "macd_slow",
            "macd_signal",
        ]
        df = df[column_order]

        # 依照總報酬金額與夏普比率排序
        df = df.sort_values(
            by=["total_reward_amount", "sharpe_ratio"],
            ascending=[False, False],
            na_position="last",
        )

        # Stage 1 結果：單一文件（無時間戳，每次更新覆蓋）
        csv_path = self.output_dir / "stage_1_optimized_params.csv"
        html_path = self.output_dir / "stage_1_optimized_params.html"

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"💾 Stage 1 優化參數已儲存至: {csv_path}")

        # 生成互動式 HTML 表格
        logger.info("生成 Stage 1 互動式 HTML 表格...")
        # 使用實際日期（在方法開始時已設定actual_end_date）
        parameter_meanings = {
            "stock_id": "Stock code (TSE/OTC).",
            "total_reward_amount": "Total reward amount using configurable initial capital.",
            "annual_return": "Annualized return for the backtest window.",
            "max_drawdown": "Maximum equity drawdown in the backtest window.",
            "sharpe_ratio": "Risk-adjusted return ratio (higher is better).",
            "total_trades": "Number of completed trades.",
            "sar_signal_lag_min": "Minimum days where SAR bullish flip can lead MACD golden cross.",
            "sar_signal_lag_max": "Maximum days where SAR bullish flip can lead MACD golden cross.",
            "sar_accel": "SAR acceleration factor.",
            "sar_maximum": "SAR maximum acceleration cap.",
            "macd_fast": "MACD fast EMA period.",
            "macd_slow": "MACD slow EMA period.",
            "macd_signal": "MACD signal EMA period.",
        }

        tested_params = {
            "search_mode": "single_stock_grid_search --optimize (all stocks mode)",
            "test_period": f"start_date={self.start_date}, end_date={actual_end_date}",
            "sar_lag_windows": "[(0,2), (0,3), (1,3)]",
            "sar_acceleration_values": "[0.01, 0.02, 0.03]",
            "sar_maximum_values": "[0.15, 0.20, 0.25]",
            "macd_fast_values": "[10, 12, 14]",
            "macd_slow_values": "[24, 26, 28]",
            "macd_signal_values": "[8, 9, 10]",
            "constraint": "macd_fast < macd_slow",
            "total_param_combinations": str(len(self.generate_param_grid())),
            "initial_capital": f"{self.initial_capital:.0f}",
            "fee_ratio": "0.001425",
            "tax_ratio": "0.003",
        }

        dataframe_to_sortable_html(
            df=df,
            output_path=str(html_path),
            title=f"Oscar Strategy - Grid Search Result (startdate: {self.start_date}, enddate: {actual_end_date})",
            is_grid_search_result=True,
            parameter_meanings=parameter_meanings,
            tested_params=tested_params,
        )
        logger.info(f"📊 Stage 1 HTML 表格已儲存至: {html_path}")

        # 打印統計摘要
        self._print_summary(df)

        # 找出表現最佳的股票
        self._print_top_performers(df)

        # 最終結果摘要（方便在 screen 重新連接後查看）
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Stage 1 優化完成！結果文件位置：")
        logger.info("=" * 80)
        logger.info(f"📊 Stage 1 HTML 表格: {html_path}")
        logger.info(f"💾 Stage 1 CSV 數據: {csv_path}")
        logger.info(f"📁 Stage 1 結果目錄: {self.output_dir}")
        logger.info(f"💡 檢查點位置: {self.checkpoint_dir / checkpoint_name}.json")
        logger.info("=" * 80)
        logger.info(f"✅ 成功優化 {len(df)} 檔股票")
        logger.info(
            f"🏆 最佳股票: {df.iloc[0]['stock_id']} (總報酬金額: {df.iloc[0]['total_reward_amount']:.2f})"
        )
        logger.info("=" * 80)

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
        # ⚠️ 保持 base_position 的完整日期範圍（從 start_date 開始）
        single_stock_position = pd.DataFrame(
            False, index=base_position.index, columns=base_position.columns
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
            position_limit=1.0,
        )

        # 儲存回測報告（到 Stage 2 目錄）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.stage2_dir / f"{stock_id}_report.html"
        report.display(save_report_path=str(report_path))
        logger.info(f"Stage 2 回測報告已儲存至: {report_path}")

        # 提取績效指標
        metrics = get_metrics_with_fixed_annual_return(
            report,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        trades = report.get_trades()

        # 打印績效摘要
        logger.info("\n" + "=" * 60)
        logger.info(f"股票 {stock_id} 績效摘要")
        logger.info("=" * 60)
        logger.info(f"交易次數: {len(trades)}")
        logger.info(f"年化報酬率: {metrics['profitability']['annualReturn']:.2%}")
        logger.info(f"最大回檔: {metrics['risk']['maxDrawdown']:.2%}")
        if "sharpeRatio" in metrics["ratio"]:
            logger.info(f"夏普比率: {metrics['ratio']['sharpeRatio']:.2f}")
        logger.info("=" * 60)

        # 準備視覺化數據
        logger.info("生成交易視覺化圖表...")

        # 準備價格數據
        price_df = prepare_price_data(
            stock_id=stock_id,
            market_data=strategy.market_data,
            start_date=self.start_date,
        )

        # 取得該股票的指標數據
        sar_stock = strategy.sar_values[stock_id].loc[self.start_date :]
        macd_dif_stock = strategy.macd_dif[stock_id].loc[self.start_date :]
        macd_dea_stock = strategy.macd_dea[stock_id].loc[self.start_date :]
        macd_hist_stock = strategy.macd_histogram[stock_id].loc[self.start_date :]

        # 取得該股票的交易價格（開盤價）
        trade_price_stock = strategy.trade_price[stock_id].loc[self.start_date :]

        # 取得該股票的法人買賣超數據
        foreign_buy_stock = strategy.institutional_condition["foreign_buy"][
            stock_id
        ].loc[self.start_date :]
        trust_buy_stock = strategy.institutional_condition["trust_buy"][stock_id].loc[
            self.start_date :
        ]
        dealer_buy_stock = strategy.institutional_condition["dealer_buy"][stock_id].loc[
            self.start_date :
        ]

        # 取得策略持倉訊號
        position_stock = base_position[stock_id].loc[self.start_date :]

        # 計算實際交易訊號（從持倉變化計算）
        position_changes = position_stock.astype(int).diff()
        actual_buy_signals = position_changes == 1  # 0->1 表示買入
        actual_sell_signals = position_changes == -1  # 1->0 表示賣出

        # 生成視覺化圖表（保存到 Stage 2 目錄）
        viz_path = self.stage2_dir / f"{stock_id}_visualization.html"
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
            output_path=str(viz_path),
        )
        logger.info(f"視覺化圖表已儲存至: {viz_path}")

        # 返回結果
        result = {
            "stock_id": stock_id,
            "total_trades": len(trades),
            "total_reward_amount": compute_total_reward_amount_from_creturn(
                creturn=report.creturn,
                initial_capital=self.initial_capital,
                start_date=self.start_date,
                end_date=self.end_date,
            ),
            "annual_return": metrics["profitability"]["annualReturn"],
            "max_drawdown": metrics["risk"]["maxDrawdown"],
            "sharpe_ratio": metrics["ratio"].get("sharpeRatio", None),
            "sortino_ratio": metrics["ratio"].get("sortinoRatio", None),
            "calmar_ratio": metrics["ratio"].get("calmarRatio", None),
            "report_path": str(report_path),
            "visualization_path": str(viz_path),
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
        stocks_with_signals = [
            stock for stock in all_stocks if base_position[stock].any()
        ]
        logger.info(
            f"股票過濾：{len(all_stocks)} 總數 → {len(stocks_with_signals)} 有訊號"
        )

        logger.info(
            f"開始測試 {len(stocks_with_signals)} 檔股票的個別績效 (使用 {self.pool} workers)"
        )

        results = []

        # 使用並行處理測試所有股票（base_position傳入全局初始化器，不再序列化）
        logger.info(f"正在初始化 {self.pool} 個工作進程...")
        with ProcessPoolExecutor(
            max_workers=self.pool,
            initializer=_init_worker,
            initargs=(None, base_position, self.initial_capital),
        ) as executor:
            logger.info("工作進程初始化完成，開始測試...")
            # 提交所有任務（不再傳遞base_position）
            future_to_stock = {
                executor.submit(self._test_single_stock_optimized, stock): stock
                for stock in stocks_with_signals
            }

            # 使用 tqdm 追蹤進度
            with tqdm(
                total=len(stocks_with_signals),
                desc="測試進度",
                unit="stock",
                ncols=100,
                miniters=1,
            ) as pbar:
                for future in as_completed(future_to_stock):
                    stock = future_to_stock[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            pbar.set_postfix(
                                {
                                    "stock": stock,
                                    "reward": f"{result['total_reward_amount']:.2f}",
                                }
                            )
                    except Exception as e:
                        logger.warning(f"股票 {stock} 回測失敗: {e}")
                    finally:
                        pbar.update(1)

        # 轉換為 DataFrame
        df = pd.DataFrame(results)

        # 依照總報酬金額排序後儲存 CSV
        df = df.sort_values(
            by=["total_reward_amount", "sharpe_ratio"],
            ascending=[False, False],
            na_position="last",
        )
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"結果已儲存至: {output_path}")
        logger.info(f"成功測試 {len(df)} / {len(all_stocks)} 檔股票")

        return df

    @staticmethod
    def _param_to_key(params):
        """將參數字典轉為可哈希的字符串鍵"""
        return (
            f"sar_lag_{params['sar_signal_lag_min']}_{params['sar_signal_lag_max']}_"
            f"{params['sar_params']['acceleration']:.3f}_{params['sar_params']['maximum']:.3f}_"
            f"macd_{params['macd_params']['fastperiod']}_{params['macd_params']['slowperiod']}_{params['macd_params']['signalperiod']}"
        )

    @staticmethod
    def _calculate_and_save_strategy(
        param_key, params, start_date, end_date, cache_dir
    ):
        """
        並行計算策略位置並保存到磁盤（Stage 1 worker）

        Args:
            param_key: 參數鍵
            params: 參數字典
            start_date: 起始日期
            end_date: 結束日期
            cache_dir: 快取目錄路徑

        Returns:
            tuple: (param_key, success, error_msg)
        """
        import pickle
        from pathlib import Path

        global _WORKER_MARKET_DATA

        try:
            # 使用全局市場數據（避免重複序列化）
            strategy = _create_oscar_strategy_compat(
                sar_signal_lag_min=params["sar_signal_lag_min"],
                sar_signal_lag_max=params["sar_signal_lag_max"],
                sar_params=params["sar_params"],
                macd_params=params["macd_params"],
                market_data=_WORKER_MARKET_DATA,
            )

            # ⚠️ 重要：確保 position 從 start_date 開始（而非第一個訊號日期）
            # 否則 finlab sim() 會從第一個訊號計算年化報酬，而非從 start_date
            base_position = strategy.base_position

            # 獲取完整交易日曆（從市場數據的 close 價格）
            trading_days = _WORKER_MARKET_DATA["close"].loc[start_date:].index
            if end_date:
                trading_days = trading_days[trading_days <= end_date]

            # 重新索引：確保包含從 start_date 開始的所有交易日（缺失值填 False）
            base_position = base_position.reindex(trading_days, fill_value=False)

            # 保存到磁盤快取
            cache_file = Path(cache_dir) / f"{param_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(base_position, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 清理記憶體
            OscarAndOrStrategy.clear_runtime_cache()
            del strategy, base_position

            return (param_key, True, None)

        except Exception as e:
            return (param_key, False, str(e))

    @staticmethod
    def _backtest_single_position(stock_id, position, params, initial_capital):
        """
        對預先計算好的位置執行回測（不重新計算策略）

        Args:
            stock_id: 股票代碼
            position: 預先計算好的持倉訊號 DataFrame
            params: 參數字典（用於記錄）

        Returns:
            dict: 回測結果（包含stock_id）
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
                position_limit=1.0,
            )

            # 提取績效指標
            metrics = get_metrics_with_fixed_annual_return(
                report,
                start_date=position.index[0],
                end_date=position.index[-1],
            )
            trades = report.get_trades()

            # 記錄結果（✅ 包含stock_id避免KeyError）
            result = {
                "stock_id": stock_id,  # ← 修正：添加stock_id
                "sar_signal_lag_min": params["sar_signal_lag_min"],
                "sar_signal_lag_max": params["sar_signal_lag_max"],
                "sar_accel": params["sar_params"]["acceleration"],
                "sar_maximum": params["sar_params"]["maximum"],
                "macd_fast": params["macd_params"]["fastperiod"],
                "macd_slow": params["macd_params"]["slowperiod"],
                "macd_signal": params["macd_params"]["signalperiod"],
                "total_trades": len(trades),
                "total_reward_amount": compute_total_reward_amount_from_creturn(
                    creturn=report.creturn,
                    initial_capital=initial_capital,
                    start_date=position.index[0],
                    end_date=position.index[-1],
                ),
                "annual_return": metrics["profitability"]["annualReturn"],
                "max_drawdown": metrics["risk"]["maxDrawdown"],
                "sharpe_ratio": metrics["ratio"].get("sharpeRatio", None),
                "params": params,
            }

            OscarAndOrStrategy.clear_runtime_cache()
            return result

        except Exception as e:
            logger.warning(f"回測失敗 {stock_id} ({params}): {e}")
            return None

    @staticmethod
    def _test_param_group(params, stock_list, start_date, initial_capital):
        """
        測試單組參數下的多個股票（智能緩存：策略位置只計算一次）

        Args:
            params: 參數字典
            stock_list: 該參數下需要測試的股票列表
            start_date: 回測開始日期

        Returns:
            list: 該參數組合下所有股票的回測結果
        """
        from finlab.backtest import sim

        global _WORKER_MARKET_DATA

        results = []

        try:
            # 初始化策略（只計算一次！）
            strategy = _create_oscar_strategy_compat(
                sar_signal_lag_min=params["sar_signal_lag_min"],
                sar_signal_lag_max=params["sar_signal_lag_max"],
                sar_params=params["sar_params"],
                macd_params=params["macd_params"],
                market_data=_WORKER_MARKET_DATA,
            )

            # ⚠️ 重要：確保 position 從 start_date 開始（而非第一個訊號日期）
            # 否則 finlab sim() 會從第一個訊號計算年化報酬，而非從 start_date
            base_position = strategy.base_position

            # 獲取完整交易日曆（從市場數據的 close 價格）
            trading_days = _WORKER_MARKET_DATA["close"].loc[start_date:].index

            # 重新索引：確保包含從 start_date 開始的所有交易日（缺失值填 False）
            base_position = base_position.reindex(trading_days, fill_value=False)

            # 對該參數下的每個股票執行回測（使用同一個位置DataFrame）
            for stock_id in stock_list:
                try:
                    # 快速過濾：如果該股票完全沒有訊號，跳過
                    if (
                        stock_id not in base_position.columns
                        or not base_position[stock_id].any()
                    ):
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
                        position_limit=1.0,
                    )

                    # 提取績效指標
                    metrics = get_metrics_with_fixed_annual_return(
                        report,
                        start_date=start_date,
                    )
                    trades = report.get_trades()

                    # 記錄結果
                    result = {
                        "stock_id": stock_id,
                        "sar_signal_lag_min": params["sar_signal_lag_min"],
                        "sar_signal_lag_max": params["sar_signal_lag_max"],
                        "sar_accel": params["sar_params"]["acceleration"],
                        "sar_maximum": params["sar_params"]["maximum"],
                        "macd_fast": params["macd_params"]["fastperiod"],
                        "macd_slow": params["macd_params"]["slowperiod"],
                        "macd_signal": params["macd_params"]["signalperiod"],
                        "total_trades": len(trades),
                        "total_reward_amount": compute_total_reward_amount_from_creturn(
                            creturn=report.creturn,
                            initial_capital=initial_capital,
                            start_date=start_date,
                            end_date=None,
                        ),
                        "annual_return": metrics["profitability"]["annualReturn"],
                        "max_drawdown": metrics["risk"]["maxDrawdown"],
                        "sharpe_ratio": metrics["ratio"].get("sharpeRatio", None),
                        "params": params,
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
    def _test_single_param(stock_id, params, start_date, initial_capital):
        """
        測試單一參數組合 (靜態方法供並行處理使用)
        使用全局 _WORKER_MARKET_DATA 避免昂貴的序列化

        Args:
            stock_id: 股票代碼
            params: 參數字典
            start_date: 回測開始日期

        Returns:
            dict: 單一參數組合的回測結果
        """
        from finlab.backtest import sim

        global _WORKER_MARKET_DATA

        try:
            # 初始化策略（使用全局預載數據）
            strategy = _create_oscar_strategy_compat(
                sar_signal_lag_min=params["sar_signal_lag_min"],
                sar_signal_lag_max=params["sar_signal_lag_max"],
                sar_params=params["sar_params"],
                macd_params=params["macd_params"],
                market_data=_WORKER_MARKET_DATA,
            )

            # ⚠️ 重要：確保 position 從 start_date 開始（而非第一個訊號日期）
            # 否則 finlab sim() 會從第一個訊號計算年化報酬，而非從 start_date
            base_position = strategy.base_position

            # 獲取完整交易日曆（從市場數據的 close 價格）
            trading_days = _WORKER_MARKET_DATA["close"].loc[start_date:].index

            # 重新索引：確保包含從 start_date 開始的所有交易日（缺失值填 False）
            base_position = base_position.reindex(trading_days, fill_value=False)

            # 檢查股票是否在資料中
            if stock_id not in base_position.columns:
                return None

            # 建立單一股票的持倉訊號
            single_stock_position = pd.DataFrame(
                False, index=base_position.index, columns=base_position.columns
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
                position_limit=1.0,
            )

            # 提取績效指標
            metrics = get_metrics_with_fixed_annual_return(
                report,
                start_date=start_date,
            )
            trades = report.get_trades()

            # 記錄結果（只儲存數值,不儲存大物件）
            result = {
                "sar_signal_lag_min": params["sar_signal_lag_min"],
                "sar_signal_lag_max": params["sar_signal_lag_max"],
                "sar_accel": params["sar_params"]["acceleration"],
                "sar_maximum": params["sar_params"]["maximum"],
                "macd_fast": params["macd_params"]["fastperiod"],
                "macd_slow": params["macd_params"]["slowperiod"],
                "macd_signal": params["macd_params"]["signalperiod"],
                "total_trades": len(trades),
                "total_reward_amount": compute_total_reward_amount_from_creturn(
                    creturn=report.creturn,
                    initial_capital=initial_capital,
                    start_date=start_date,
                    end_date=None,
                ),
                "annual_return": metrics["profitability"]["annualReturn"],
                "max_drawdown": metrics["risk"]["maxDrawdown"],
                "sharpe_ratio": metrics["ratio"].get("sharpeRatio", None),
                "params": params,  # 只保留參數字典，不保留 strategy 和 base_position
            }

            OscarAndOrStrategy.clear_runtime_cache()
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

        global _WORKER_BASE_POSITION, _WORKER_INITIAL_CAPITAL

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
                position_limit=1.0,
            )
            # 移除HTML生成（I/O瓶頸）- 只在需要時單獨生成

            # 提取績效指標
            metrics = get_metrics_with_fixed_annual_return(
                report,
                start_date=base_position.index[0] if len(base_position.index) else None,
                end_date=base_position.index[-1] if len(base_position.index) else None,
            )
            trades = report.get_trades()

            # 整理結果 (只保留關鍵指標，減少內存占用)
            result = {
                "stock_id": stock,
                "total_trades": len(trades),
                "total_reward_amount": compute_total_reward_amount_from_creturn(
                    creturn=report.creturn,
                    initial_capital=_WORKER_INITIAL_CAPITAL,
                    start_date=base_position.index[0],
                    end_date=base_position.index[-1] if len(base_position.index) else None,
                ),
                "annual_return": metrics["profitability"]["annualReturn"],
                "max_drawdown": metrics["risk"]["maxDrawdown"],
                "sharpe_ratio": metrics["ratio"].get("sharpeRatio", None),
                "sortino_ratio": metrics["ratio"].get("sortinoRatio", None),
                "calmar_ratio": metrics["ratio"].get("calmarRatio", None),
                "volatility": metrics["ratio"].get("volatility", None),
                "profit_factor": metrics["ratio"].get("profitFactor", None),
                "win_rate": metrics["winrate"].get("winRate", None),
                "expectancy": metrics["winrate"].get("expectancy", None),
                "mae": metrics["winrate"].get("mae", None),
                "mfe": metrics["winrate"].get("mfe", None),
                "avg_drawdown": metrics["risk"].get("avgDrawdown", None),
                "avg_drawdown_days": metrics["risk"].get("avgDrawdownDays", None),
                "alpha": metrics["profitability"].get("alpha", None),
                "beta": metrics["profitability"].get("beta", None),
                "total_days": len(base_position),
                "holding_days": base_position[stock].sum(),
            }

            OscarAndOrStrategy.clear_runtime_cache()
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
        return SingleStockGridSearchExecutor._test_single_stock_optimized(
            stock, fee_ratio, tax_ratio
        )

    def _print_summary(self, df):
        """打印統計摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("整體統計摘要")
        logger.info("=" * 60)

        logger.info(f"總測試股票數: {len(df)}")
        logger.info(f"平均總報酬金額: {df['total_reward_amount'].mean():.2f}")
        logger.info(f"平均年化報酬率: {df['annual_return'].mean():.2%}")
        logger.info(f"平均最大回檔: {df['max_drawdown'].mean():.2%}")

        if df["sharpe_ratio"].notna().any():
            logger.info(f"平均夏普比率: {df['sharpe_ratio'].mean():.2f}")

        logger.info(f"平均交易次數: {df['total_trades'].mean():.1f}")

        # 統計正報酬股票數
        positive_returns = (df["annual_return"] > 0).sum()
        logger.info(
            f"正報酬股票數: {positive_returns} ({positive_returns / len(df) * 100:.1f}%)"
        )

    def _print_top_performers(self, df, top_n=10):
        """打印表現最佳的股票"""
        logger.info("\n" + "=" * 60)
        logger.info(f"總報酬金額 Top {top_n}")
        logger.info("=" * 60)

        top_by_return = df.nlargest(top_n, "total_reward_amount")[
            [
                "stock_id",
            "total_reward_amount",
                "annual_return",
                "max_drawdown",
                "sharpe_ratio",
                "total_trades",
            ]
        ].copy()

        # 格式化顯示
        top_by_return["total_reward_amount"] = top_by_return[
            "total_reward_amount"
        ].apply(lambda x: f"{x:.2f}")
        top_by_return["annual_return"] = top_by_return["annual_return"].apply(
            lambda x: f"{x:.2%}"
        )
        top_by_return["max_drawdown"] = top_by_return["max_drawdown"].apply(
            lambda x: f"{x:.2%}"
        )
        top_by_return["sharpe_ratio"] = top_by_return["sharpe_ratio"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

        print(top_by_return.to_string(index=False))

        # 如果有夏普比率資料，也打印夏普比率 Top N
        if df["sharpe_ratio"].notna().any():
            logger.info("\n" + "=" * 60)
            logger.info(f"夏普比率 Top {top_n}")
            logger.info("=" * 60)

            top_by_sharpe = df.nlargest(top_n, "sharpe_ratio")[
                [
                    "stock_id",
                    "annual_return",
                    "max_drawdown",
                    "sharpe_ratio",
                    "total_trades",
                ]
            ].copy()

            # 格式化顯示
            top_by_sharpe["annual_return"] = top_by_sharpe["annual_return"].apply(
                lambda x: f"{x:.2%}"
            )
            top_by_sharpe["max_drawdown"] = top_by_sharpe["max_drawdown"].apply(
                lambda x: f"{x:.2%}"
            )
            top_by_sharpe["sharpe_ratio"] = top_by_sharpe["sharpe_ratio"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

            print(top_by_sharpe.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single Stock Grid Search Executor")
    parser.add_argument(
        "--start_date", type=str, default="2023-01-01", help="回測起始日期"
    )
    parser.add_argument(
        "--enddate",
        type=str,
        default=None,
        help="回測結束日期（可選，預設為最新價格日期）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/OscarTWStrategy/single_stock",
        help="結果輸出目錄",
    )
    parser.add_argument(
        "--sar_signal_lag_min", type=int, default=0, help="SAR與MACD訊號最小天數差"
    )
    parser.add_argument(
        "--sar_signal_lag_max", type=int, default=2, help="SAR與MACD訊號最大天數差"
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=None,
        help="並行處理的worker數量（默認自動：<100核心用cores-2，>=100核心用75%%。建議144核機器用100-110）",
    )
    parser.add_argument(
        "--initial_capital",
        type=float,
        default=100_000,
        help="計算 total reward amount 的初始資金",
    )
    parser.add_argument(
        "--stock_id", type=str, default=None, help="指定測試單一股票（可選）"
    )
    parser.add_argument("--optimize", action="store_true", help="啟用參數優化")

    args = parser.parse_args()

    # Auto-detect optimal pool size if not specified
    if args.pool is None:
        import os

        cpu_count = os.cpu_count() or 4
        # For high-core-count machines (>100 cores), use 70-75% to avoid context switching
        # For normal machines, use cores - 2
        if cpu_count > 100:
            args.pool = int(cpu_count * 0.75)  # e.g., 144 cores → 108 workers
            logger.info(
                f"自動偵測到高核心數機器 ({cpu_count} 核心)，設定並行處理數為 {args.pool} (~75%)"
            )
        else:
            args.pool = max(1, cpu_count - 2)  # Leave 2 cores for system
            logger.info(
                f"自動偵測到 {cpu_count} 個CPU核心，設定並行處理數為 {args.pool}"
            )

    executor = SingleStockGridSearchExecutor(
        start_date=args.start_date,
        end_date=args.enddate,
        output_dir=args.output_dir,
        sar_signal_lag_min=args.sar_signal_lag_min,
        sar_signal_lag_max=args.sar_signal_lag_max,
        pool=args.pool,
        stock_id=args.stock_id,
        optimize_params=args.optimize,
        initial_capital=args.initial_capital,
    )

    executor.run_test()
