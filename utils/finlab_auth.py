"""FinLab 登入：優先使用 session 環境變數，缺少時退回 legacy api_token。

容器無法使用 ~/.finlab/credentials.json（該檔綁定產生它的機器，複製過去解不開），
改在已登入的機器執行 `python -m finlab token >> .env` 產生下列三個變數。
"""

import logging
import os

import finlab

logger = logging.getLogger(__name__)

SESSION_ENV_VARS = ("FINLAB_REFRESH_TOKEN", "FINLAB_SESSION_ID", "FINLAB_API_KEY")


def _login_with_session() -> bool:
    """以 session 環境變數登入，成功回傳 True，否則讓呼叫端退回 legacy。"""
    if not all(os.environ.get(var) for var in SESSION_ENV_VARS):
        return False

    try:
        from finlab.auth import get_id_token
    except ImportError:
        logger.warning("finlab.auth 不存在（finlab < 2.x），改用 FINLAB_API_TOKEN")
        return False

    # 不用 finlab.login()：憑證失效時它會退回瀏覽器流程，容器內會卡到逾時（300 秒）
    try:
        if get_id_token():
            logger.info("Successfully logged into FinLab (session credentials)")
            return True
    except Exception as e:
        logger.warning(f"FinLab session 憑證換發 id_token 失敗：{e}")

    logger.warning("FinLab session 憑證無效或已過期，改用 FINLAB_API_TOKEN")
    return False


def login_finlab(api_token: str | None = None) -> None:
    """登入 FinLab。

    Args:
        api_token: legacy api_token；未提供時讀取 FINLAB_API_TOKEN 環境變數。
    """
    if _login_with_session():
        return

    token = api_token if api_token is not None else os.environ.get("FINLAB_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "找不到 FinLab 憑證：在已登入的機器執行 `python -m finlab token >> .env` "
            f"產生 {' / '.join(SESSION_ENV_VARS)}，或設定 FINLAB_API_TOKEN"
        )

    # 套件內標示 api_token 於 2026/08/01 後移除（官網未提期限），目前仍可用
    try:
        finlab.login(token)
    except (TypeError, AttributeError) as e:
        raise EnvironmentError(
            "finlab 已不接受 api_token 登入，請改用 session 環境變數："
            "在已登入的機器執行 `python -m finlab token >> .env` 產生 "
            f"{' / '.join(SESSION_ENV_VARS)}"
        ) from e
    logger.warning("FinLab 使用已棄用的 api_token 登入，請改設 session 環境變數")
