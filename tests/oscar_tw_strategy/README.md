# Oscar TW Strategy — 測試腳本說明

本目錄包含 `OscarAndOrStrategy` 與 `OscarCompositeStrategy` 兩條策略線的貝葉斯參數優化與滾動前向驗證腳本。

---

## 腳本總覽

| 腳本 | 對應策略 | 功能 |
|---|---|---|
| `bayesian_optimize_andor_params.py` | AndOr | 單次貝葉斯參數優化 |
| `walk_forward_bayes_andor.py` | AndOr | 滾動前向驗證（包裝上方腳本） |
| `bayesian_optimize_composite_params.py` | Composite | 單次貝葉斯參數優化 |
| `walk_forward_bayes_composite.py` | Composite | 滾動前向驗證（包裝上方腳本） |

---

## 兩條策略線的差異

- **AndOr**（`OscarAndOrStrategy`）：SAR 與 MACD 訊號以「AND / OR」邏輯組合，參數較少、邏輯直觀。
- **Composite**（`OscarCompositeStrategy`）：SAR、MACD、成交量、法人買賣以加權分數合成，額外優化各訊號的權重、分位數分箱、sigmoid 衰減等大量超參數。

---

## 單次貝葉斯優化

### `bayesian_optimize_andor_params.py` / `bayesian_optimize_composite_params.py`

在**固定的時間區間**內，使用 [Optuna](https://optuna.org/) TPE Sampler 搜尋最佳策略參數。

**流程：**
1. 載入市場資料（優先從 pickle 讀取，不存在才從 finlab 抓取）
2. 每次 trial 建立策略 → 執行回測 → 以目標函數評分
3. 跑完所有 trial 後，從完成的 trial 中選出最佳結果
4. 輸出結果至 `study_dir/`

**可選目標函數（`--objective-name`）：**

| 值 | 說明 |
|---|---|
| `train_annual_return`（預設） | 年化報酬率 |
| `train_sharpe` | 夏普比率 |
| `train_calmar` | Calmar 比率（年化報酬 / 最大回撤） |

**輸出（`assets/.../andor_{timestamp}/` 或 `composite_{timestamp}/`）：**

| 檔案 | 內容 |
|---|---|
| `best_params.json` | 最佳參數與回測指標 |
| `trials.csv` | 所有 trial 的參數與分數 |
| `optuna_journal.log` | Optuna 原始 journal（可用於繼續優化） |

**使用範例：**
```bash
# AndOr
python -m tests.oscar_tw_strategy.bayesian_optimize_andor_params \
    --start-date 2020-01-01 --end-date 2024-12-31 \
    --n-trials 1000 --workers 4 --objective-name train_sharpe

# Composite
python -m tests.oscar_tw_strategy.bayesian_optimize_composite_params \
    --start-date 2020-01-01 --end-date 2024-12-31 \
    --n-trials 1000 --workers 4
```

---

## 滾動前向驗證（Walk-Forward）

### `walk_forward_bayes_andor.py` / `walk_forward_bayes_composite.py`

在多個滾動視窗上反覆進行貝葉斯優化，最終拼接所有 out-of-sample（OOS）倉位，輸出一份**純 OOS** 的回測報告。

**流程：**
```
整段期間
├── 視窗 0：train [t0, t0+12M)  →  以 Bayesian 找最佳參數
│            oos  [t0+12M, t0+15M)  →  用最佳參數產生倉位
├── 視窗 1：train [t0+3M, t0+15M)  →  重新優化
│            oos  [t0+15M, t0+18M)  →  用最佳參數產生倉位
│   ...
└── 拼接所有 OOS 倉位 → 執行一次 finlab sim → 輸出報告
```

> 預設值：`train_window_months=12`、`oos_window_months=3`，每次向前滾動 3 個月。

**輸出（`assets/.../wf_andor_{timestamp}/` 或 `wf_composite_{timestamp}/`）：**

| 檔案 / 目錄 | 內容 |
|---|---|
| `walk_forward_report.html` | **最終 OOS 回測報告**（equity curve、統計數據） |
| `walk_forward_positions.csv` | 所有 OOS 片段拼接的完整倉位表 |
| `walk_forward_summary.json` | 各視窗的訓練/驗證區間、OOS 交易天數、最佳參數 |
| `window_000/`, `window_001/`, ... | 各視窗的 Bayesian 優化產出（`best_params.json`、`trials.csv`） |

**使用範例：**
```bash
# AndOr
python -m tests.oscar_tw_strategy.walk_forward_bayes_andor \
    --start-date 2020-01-01 --end-date 2024-12-31 \
    --train-window-months 12 --oos-window-months 3 \
    --n-trials 300 --workers 4

# Composite
python -m tests.oscar_tw_strategy.walk_forward_bayes_composite \
    --start-date 2020-01-01 --end-date 2024-12-31 \
    --train-window-months 12 --oos-window-months 3 \
    --n-trials 300 --workers 4
```

---

## utils/

| 模組 | 說明 |
|---|---|
| `objective_functions.py` | 定義 `ObjectiveName` enum 與三種目標函數（Sharpe、AnnualReturn、Calmar） |
| `trial_result.py` | `TrialResult` dataclass，將每次 trial 的指標寫入 Optuna user attrs |
| `custom_report_metrics.py` | 修正 finlab 年化報酬計算（以實際回測區間為準，空倉期視為現金） |

---

## 市場資料快取

兩條策略線各自有一個 pickle 快取路徑：

- `finlab_db/workspace/oscar_andor_market_data.pkl`
- `finlab_db/workspace/oscar_composite_market_data.pkl`

首次執行若 pickle 不存在，會自動從 finlab 抓取並寫入。後續執行直接讀取快取，可大幅縮短啟動時間。多進程模式（`--workers > 1`）下，主進程負責寫入，worker 只讀不寫。
