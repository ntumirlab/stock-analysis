import os
from os import path
import sys
import logging
import traceback
import keyring
from finlab.online.sinopac_account import SinopacAccount

# fugle 為 legacy 路徑（正式排程僅 shioaji）：fugle_trade 與 finlab 2.x 的
# Fugle 模組（依賴未公開發行的 esun_trade）都可能不在環境中，import 失敗
# 僅停用 fugle 分支，不得影響 shioaji 下單路徑
try:
    from fugle_trade.util import setup_keyring
except ImportError:
    setup_keyring = None
try:
    from finlab.online.fugle_account import FugleAccount
except ImportError:
    FugleAccount = None
from core.notification_formats import format_login_retry_notice
from core.retry import call_with_retry
from utils.config_loader import ConfigLoader
from utils.finlab_auth import login_finlab as finlab_login
from utils.notifier import create_notification_manager

logger = logging.getLogger(__name__)

# shioaji 登入失敗後的等待秒數（共 4 次嘗試）。等待合計 100 秒，加上每次嘗試
# 本身的逾時，08:10 下單最壞數分鐘內放棄，不影響 09:00 開盤。
# 不分錯誤類型一律重試：無法事先確定永豐端異常會丟哪種例外；帳密/憑證錯誤
# 的代價只是晚約 2 分鐘才報錯
SHIOAJI_LOGIN_RETRY_DELAYS = (10, 30, 60)


def _new_sinopac_account():
    """建立 SinopacAccount；初始化中途失敗時先登出已建立的 session 再拋出。

    SinopacAccount.__init__ 在 api.login() 之後還會 activate_ca 等，中途失敗
    時物件不會回傳給呼叫端，session 就沒人登出。先 __new__ 拿到物件，才能在
    失敗時取得 account.api 收尾，避免重試累積佔用 shioaji 連線數。
    """
    account = SinopacAccount.__new__(SinopacAccount)
    try:
        account.__init__()
    except Exception:
        api = getattr(account, "api", None)
        if api is not None:
            try:
                api.logout()
            except Exception as logout_error:
                logger.warning(f"登入失敗後登出 shioaji session 也失敗（忽略）: {logout_error!r}")
        raise
    return account


class Authenticator:
    def __init__(self, config_loader: ConfigLoader | None = None):
        self.config_loader = config_loader

    def login_finlab(self):
        if not self.config_loader:
            raise RuntimeError("ConfigLoader is required for Authenticator. Pass an instance when constructing.")
        finlab_login(self.config_loader.get_env_var("FINLAB_API_TOKEN"))

    def _login_fugle(self):
        if FugleAccount is None or setup_keyring is None:
            raise RuntimeError("環境缺少 Fugle 相關套件（fugle_trade / esun_trade），無法登入 fugle 帳戶")
        if not self.config_loader:
            raise RuntimeError("ConfigLoader is required for Authenticator. Pass an instance when constructing.")
        required_vars = [
            "FUGLE_CONFIG_PATH", 
            "FUGLE_MARKET_API_KEY", 
            "FUGLE_ACCOUNT", 
            "FUGLE_ACCOUNT_PASSWORD", 
            "FUGLE_CERT_PASSWORD"
        ]
        for var in required_vars:
            if not self.config_loader.get_env_var(var):
                raise EnvironmentError(f"Missing environment variable: {var}")

        # 根據作業系統判斷 cryptfile_pass.cfg 的位置
        if sys.platform.startswith("win"):
            cryptfile_path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Python Keyring", "cryptfile_pass.cfg")
        elif sys.platform.startswith("linux"):
            cryptfile_path = os.path.join(os.path.expanduser("~"), ".local", "share", "python_keyring", "cryptfile_pass.cfg")
        else:
            raise EnvironmentError("Unsupported operating system for cryptfile_pass.cfg handling")

        # 如果檔案存在，先刪除
        if os.path.exists(cryptfile_path):
            try:
                os.remove(cryptfile_path)
                logger.info(f"Removed existing cryptfile_pass.cfg at {cryptfile_path}")
            except Exception as e:
                logger.error(f"Failed to remove cryptfile_pass.cfg at {cryptfile_path}: {e}")
                raise

        fugle_account = self.config_loader.get_env_var("FUGLE_ACCOUNT")
        setup_keyring(fugle_account)
        keyring.set_password("fugle_trade_sdk:account", fugle_account, self.config_loader.get_env_var("FUGLE_ACCOUNT_PASSWORD"))
        keyring.set_password("fugle_trade_sdk:cert", fugle_account, self.config_loader.get_env_var("FUGLE_CERT_PASSWORD"))

        account = FugleAccount()
        logger.info("Successfully logged into Fugle")
        return account

    def _login_shioaji(self):
        if not self.config_loader:
            raise RuntimeError("ConfigLoader is required for Authenticator. Pass an instance when constructing.")
        required_vars = [
            "SHIOAJI_API_KEY", 
            "SHIOAJI_SECRET_KEY", 
            "SHIOAJI_CERT_PERSON_ID", 
            "SHIOAJI_CERT_PATH", 
            "SHIOAJI_CERT_PASSWORD"
        ]
        for var in required_vars:
            if not self.config_loader.get_env_var(var):
                raise EnvironmentError(f"Missing environment variable: {var}")

        cert_path = path.expandvars(self.config_loader.get_env_var("SHIOAJI_CERT_PATH"))
        if not path.isabs(cert_path):
            # Resolve relative to app root (jobs set cwd to repo root)
            cert_path = path.join(os.getcwd(), cert_path)
        if not path.exists(cert_path):
            raise FileNotFoundError(
                f"SHIOAJI_CERT_PATH points to a non-existent file: {cert_path}. "
                f"Ensure your .env sets a valid path (e.g., ./config/credentials/your_cert.pfx) "
                f"and that the file is present inside the container at /app/config/credentials/."
            )

        account, failures = call_with_retry(
            _new_sinopac_account,
            SHIOAJI_LOGIN_RETRY_DELAYS,
            label="Shioaji login",
            logger=logger,
        )
        logger.info("Successfully logged into Shioaji")
        if failures:
            self._notify_login_retried("shioaji", failures)
        return account

    def _notify_login_retried(self, broker_name: str, failures):
        """重試後才登入成功時發 TG 警告；通知失敗不得影響已成功的登入。"""
        try:
            notifier = create_notification_manager(
                self.config_loader.config.get("notification", {}), logger
            )
            notifier.send_warning(
                task_name="券商登入",
                body=format_login_retry_notice(failures),
                broker_name=broker_name,
            )
        except Exception as e:
            logger.warning(f"登入重試警告通知發送失敗（忽略）: {e!r}")

    def login_broker(self, broker_name: str):
        broker_name = broker_name.lower()
        if broker_name == "fugle":
            return self._login_fugle()
        elif broker_name == "shioaji":
            return self._login_shioaji()
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")

if __name__ == "__main__":

    root_dir = path.dirname(path.dirname(path.abspath(__file__)))
    os.chdir(root_dir)

    try:
        user_name = 'junting'
        broker_name = 'shioaji'
        config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
        config_loader.load_global_env_vars()
        config_loader.load_user_config(user_name, broker_name)
        auth = Authenticator(config_loader)
        auth.login_finlab()
        account = auth.login_broker(broker_name)
        print("Account:", account)

    except Exception as e:
        traceback.print_exc()

    # python -m utils.authentication