# Stock Analysis

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/stock-analysis/actions/workflows/deploy.yml/badge.svg)](https://github.com/YOUR_USERNAME/stock-analysis/actions/workflows/deploy.yml)
[![Latest Release](https://img.shields.io/github/v/release/YOUR_USERNAME/stock-analysis?label=deployed)](https://github.com/YOUR_USERNAME/stock-analysis/releases/latest)
[![Production Status](https://img.shields.io/badge/status-production-success)](https://github.com/YOUR_USERNAME/stock-analysis/deployments)

> 💡 **Quick Status Check**: Latest deployed version is shown in the badge above. Click for deployment details.

此專案利用 FinLab 提供的即時財經資料，設計股票交易策略並進行歷史回測。透過自動化流程，每日定時執行回測與下單，並將結果以表格與圖表形式展示於 Dashboard 上。Dashboard 內容包含詳細下單資訊、帳戶資金水位變化圖及每月實際回報率圖表。

## 功能
+ 策略開發與回測
    + 利用 FinLab 的即時資料進行股票策略設計與歷史回測。
+ 多用戶與多券商支援
    + 支援多使用者，並可根據需求設定不同券商（如玉山、永豐），相關參數可於 config.yaml 中調整。
+ 自動化下單
    + crontab排程自動執行交易，透過玉山富果 API、Shioaji永豐API 進行下單。
+ 批次任務與 Dashboard
    + Cronjob 1 - fetech.sh 帳務資料抓取: 獲取成交明細、庫存明細、銀行餘額、交割款 → 紀錄 → 資料庫
    + Cronjob 2 - backtest.sh 回測獨立報表: FinLab 資料 → 策略回測 → 產生html報告 → File Browser(root/assets/)
    + Cronjob 3 - order.sh 下單流程: FinLab 資料 → 策略回測 → 下單（調整持倉） → 紀錄 → 資料庫
    + Web Dashboard: 展示下單資訊、帳戶資金水位變化圖、每月實際回報率圖以及 FinLab 回測報告。

<img width="1915" height="1077" alt="image" src="https://github.com/user-attachments/assets/9bf3fc18-3ab3-4662-b18f-9f3f04259341" />



## 依賴

+ 資料與交易相關
    + finlab：取得即時財經資料與回測分析（僅能用 pip 安裝）
    + shioaji[speed]：永豐證券下單（僅能用 pip 安裝）
    + fugle-trade：玉山證券下單（僅能用 pip 安裝）
    + keyring：管理與儲存敏感資訊

    + openpyxl：處理 Excel 文件
    + ta-lib：技術分析庫，提供多種股票技術指標
    + lxml：HTML 解析工具（用於 FinLab 中 order_executer.show_alerting_stocks() 依賴 pd.read_html(res.text)）
    + pandas：資料處理與分析

+ Web 與 Dashboard
    + dash：建立交互式 Web 應用
    + dash-bootstrap-components：搭配 Bootstrap 的 UI 組件
    + Flask-AutoIndex：自動生成目錄列表（僅能用 pip 安裝）
    + gunicorn：WSGI HTTP 伺服器（僅適用於 Unix 系統）

+ 其他
    + IPython：互動式 Python 介面
    + pyyaml：讀取配置文件

## 手冊

### 開發環境設定

#### 1. 安裝依賴
```bash
# 使用 conda 安裝環境
conda env create -f environment.yml
conda activate stock-analysis

# 或使用 Makefile (推薦)
make install
```

#### 2. 設定 Pre-commit Hooks (重要!)
```bash
# 安裝 git hooks - 提交前自動檢查程式碼品質
make setup-hooks

# 或手動安裝
pre-commit install
```

Pre-commit 會在每次 `git commit` 前自動執行:
- ✅ Ruff linting (程式碼品質檢查)
- ✅ Ruff formatting (程式碼格式化)
- ✅ Secret detection (防止提交敏感資訊)
- ✅ YAML/JSON/TOML 格式檢查
- ✅ Dockerfile linting

#### 3. 程式碼品質檢查

```bash
# 檢查程式碼 (不修改)
make lint

# 自動修復可修復的問題
make lint-fix

# 格式化程式碼
make format

# 完整檢查 (lint + format)
make check

# 在所有檔案上執行 pre-commit
make pre-commit-all
```

#### 4. 安全掃描
```bash
# 掃描硬編碼的 API keys 和密碼
make scan-secrets
```

⚠️ **重要提醒**:
- 提交前務必執行 `make check` 確保程式碼品質
- 絕不在程式碼中硬編碼 API keys (使用環境變數)
- 所有財務計算必須有類型標註
- 所有 datetime 物件必須帶時區資訊 (使用 UTC)

### Docker 部署 (推薦)

> 🚀 最簡單的部署方式,無需安裝 Python 環境,只需 Docker!

#### 前置需求

| 作業系統 | 下載連結 |
|---------|---------|
| **Windows/Mac** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| **Linux** | 執行指令: `curl -fsSL https://get.docker.com \| sh` |

#### 快速開始 (3 步)

**1. 下載專案**
```bash
git clone https://github.com/JunTingLin/stock-analysis.git
cd stock-analysis
```

**2. 準備配置檔**
```bash
# 複製 .env 範本並填入你的 API Keys
cp .env.example .env
nano .env

# 參考 .env.example 了解所有環境變數
# 參考 config/config.yaml 了解配置結構
```

**3. 啟動服務**
```bash
docker compose up -d --build

# 查看狀態
docker compose ps

# 訪問 Dashboard: http://localhost:5000
```

#### 配置架構 (三層系統)

```
.env (敏感值) → config.yaml (${VAR_NAME} 引用) → ConfigLoader (解析)
```

- **Layer 1**: `.env` - 實際敏感值 (在 .gitignore,永不提交)
- **Layer 2**: `config.yaml` - 模板 (${VAR_NAME} 引用,可安全提交)
- **Layer 3**: ConfigLoader - 自動解析變數

詳細文件請見 [DOCKER_SETUP.md](./docs/DOCKER_SETUP.md)
