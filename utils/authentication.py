import os
from os import path
import sys
import logging
import traceback
import keyring
import finlab
from finlab.online.fugle_account import FugleAccount
from finlab.online.sinopac_account import SinopacAccount
from fugle_trade.util import setup_keyring
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class Authenticator:
    def __init__(self, config_loader: ConfigLoader | None = None):
        self.config_loader = config_loader

    def login_finlab(self):
        if not self.config_loader:
            raise RuntimeError("ConfigLoader is required for Authenticator. Pass an instance when constructing.")
        token = self.config_loader.get_env_var("FINLAB_API_TOKEN")
        if not token:
            raise EnvironmentError("Missing environment variable: FINLAB_API_TOKEN")

        finlab.login(token)
        logger.info("Successfully logged into FinLab")

    def _login_fugle(self):
        if not self.config_loader:
            raise RuntimeError("ConfigLoader is required for Authenticator. Pass an instance when constructing.")
        required_vars = [
            "FUGLE_CONFIG_PATH",
            "FUGLE_MARKET_API_KEY",
            "FUGLE_ACCOUNT",
            "FUGLE_ACCOUNT_PASSWORD",
            "FUGLE_CERT_PASSWORD",
        ]
        for var in required_vars:
            if not self.config_loader.get_env_var(var):
                raise EnvironmentError(f"Missing environment variable: {var}")

        # 根據作業系統判斷 cryptfile_pass.cfg 的位置
        if sys.platform.startswith("win"):
            cryptfile_path = os.path.join(
                os.path.expanduser("~"), "AppData", "Local", "Python Keyring", "cryptfile_pass.cfg"
            )
        elif sys.platform.startswith("linux"):
            cryptfile_path = os.path.join(
                os.path.expanduser("~"), ".local", "share", "python_keyring", "cryptfile_pass.cfg"
            )
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
        keyring.set_password(
            "fugle_trade_sdk:account", fugle_account, self.config_loader.get_env_var("FUGLE_ACCOUNT_PASSWORD")
        )
        keyring.set_password(
            "fugle_trade_sdk:cert", fugle_account, self.config_loader.get_env_var("FUGLE_CERT_PASSWORD")
        )

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
            "SHIOAJI_CERT_PASSWORD",
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

        account = SinopacAccount()
        logger.info("Successfully logged into Shioaji")
        return account

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
