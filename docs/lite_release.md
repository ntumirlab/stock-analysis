# GoldenAI Lite 發版流程

交付物 = prebuilt image tar + `lite/` 內的範本與 README。client 端不拿 repo。

## 建置與打包

```bash
# 於 repo 根目錄（版本號規則見下）
docker build -f lite/Dockerfile -t goldenai-lite:v1.0.0 .
docker save -o goldenai-lite_v1.0.0.tar goldenai-lite:v1.0.0
```

交付內容清單：

- `goldenai-lite_vX.Y.Z.tar`
- `lite/docker-compose.yml`、`lite/config.yaml.example`、`lite/.env.example`、`lite/README.md`

## 版本號規則

`vMAJOR.MINOR.PATCH`：

- MAJOR：schema 或 config 不相容變更（client 端需要重填設定/重建 DB）。
  SQLite 無 migration 機制（`CREATE TABLE IF NOT EXISTS` 不會改舊表欄位），
  改表結構必為 MAJOR
- MINOR：功能新增、排程變更（crontab 烘在 image 內，改排程必須發版）
- PATCH：修 bug、訊息文案

client 端狀態（持倉帳本 `finlab_db/workspace/*.position.json`、`data_prod.db`、
token）都在 host volume，換版不受影響；rollback＝tag 改回舊版。

## 發版前 checklist

1. `pytest tests/unit` 全綠（CI 亦會擋）。
2. **IP 驗證**：build 內建三道守門（strategy_class 白名單、開發方模組黑名單、
   排程/dashboard 模組 import 完整性），**build 成功即通過**。要人工抽查可跑：

   ```bash
   docker run --rm goldenai-lite:vX.Y.Z bash -c "ls strategy_class; ls jobs core dao"
   # strategy_class 只允許三個檔；jobs 不得有 backtest_executor、
   # recommendations_parser/publisher、drive_fetcher；/app 不得有 dashboard.py
   ```
3. 煙霧測試：照 `lite/README.md` 在乾淨目錄起 compose，手動跑一次
   `jobs.recommendations_fetcher`，Dashboard 開得起來、能看到清單。
4. 帶 finlab 升版的發版：先以**舊的 finlab_db 狀態**跑一次新 image
   （確認 `*.position.json` 格式相容），再交付。
