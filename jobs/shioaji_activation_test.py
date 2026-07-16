"""永豐 API 開通測試（部署手冊附錄 A 第 4 步，客戶自助執行）。

依主管機關規定，首次使用 API 須先在模擬環境完成 login 與 place_order
測試並通過審核（營業日隨到隨審），正式環境的下單權限才會開通。

    python -m jobs.shioaji_activation_test          # 模擬環境登入＋下單測試
    python -m jobs.shioaji_activation_test --check  # 確認正式環境已開通

輸出面向非技術使用者：一律 print 中文結果與下一步指引，不寫 log 檔。
shioaji 採函式內延遲載入，維持模組本身可被 build 期煙霧測試無副作用 import。
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# 測試單參數：模擬環境、限價 ROD，不涉及真實資金。標的用永豐金(2890)，
# 價格取當日參考價，確保必為合法限價。
TEST_STOCK_ID = "2890"

CHECK_COMMAND = (
    "docker exec goldenai-lite-scheduler "
    "/opt/conda/envs/stock-analysis/bin/python -m jobs.shioaji_activation_test --check"
)


def _load_credentials():
    load_dotenv(".env")
    api_key = os.environ.get("SHIOAJI_API_KEY", "").strip()
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "").strip()
    if not api_key or not secret_key:
        print("❌ .env 的 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY 尚未填寫。")
        print("   請先完成手冊步驟 3-1，再執行本測試。")
        sys.exit(1)
    return api_key, secret_key


def run_simulation_test(api_key: str, secret_key: str) -> None:
    import shioaji as sj

    api = sj.Shioaji(simulation=True)
    try:
        # contracts_timeout: 阻塞等商品檔下載完成，後續取測試標的需要
        api.login(api_key, secret_key, contracts_timeout=60000)
    except Exception as exc:
        print(f"❌ 模擬環境登入失敗：{exc}")
        print("   請依序檢查：")
        print("   1. .env 的 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY 是否貼錯或多了空格")
        print("   2. 附錄 A 第 1 步的條款簽署是否已完成")
        print("   3. 測試僅於營業日 08:00～20:00 開放，請確認目前時間")
        sys.exit(1)

    try:
        contract = api.Contracts.Stocks[TEST_STOCK_ID]
        order = api.Order(
            price=float(contract.reference),
            quantity=1,
            action=sj.constant.Action.Buy,
            price_type=sj.constant.StockPriceType.LMT,
            order_type=sj.constant.OrderType.ROD,
        )
        trade = api.place_order(contract, order)
    except Exception as exc:
        print(f"❌ 模擬環境下單測試失敗：{exc}")
        print("   測試僅於營業日 08:00～20:00 開放；若時間正確仍失敗，")
        print("   請將此畫面截圖回報開發方。")
        sys.exit(1)

    print(f"✅ 模擬環境登入與下單測試已完成（測試單狀態: {trade.status.status}）")
    print("   測試紀錄隨到隨審（約 5～10 分鐘）。請稍候執行以下指令確認開通：")
    print(f"   {CHECK_COMMAND}")
    api.logout()


def run_activation_check(api_key: str, secret_key: str) -> None:
    import shioaji as sj

    # 正式環境：只登入查簽署狀態，不下單、不需憑證檔
    api = sj.Shioaji()
    try:
        api.login(api_key, secret_key, fetch_contract=False)
    except Exception as exc:
        print(f"❌ 正式環境登入失敗：{exc}")
        print("   若模擬測試剛完成，可能仍在審核中，請 10 分鐘後再執行本指令；")
        print("   若超過一個營業日仍失敗，請洽永豐營業員或客服。")
        sys.exit(1)

    signed = bool(getattr(api.stock_account, "signed", False))
    api.logout()
    if signed:
        print("✅ API 正式環境已開通，永豐設定全部完成。")
    else:
        print("⏳ 尚未開通：測試紀錄可能仍在審核中，請 10 分鐘後再執行本指令。")
        print("   若超過一個營業日仍未開通，請洽永豐營業員或客服。")
        sys.exit(1)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="永豐 API 開通測試（附錄 A 第 4 步）")
    parser.add_argument("--check", action="store_true",
                        help="以正式環境登入確認 API 已開通")
    args = parser.parse_args()

    key, secret = _load_credentials()
    if args.check:
        run_activation_check(key, secret)
    else:
        run_simulation_test(key, secret)
