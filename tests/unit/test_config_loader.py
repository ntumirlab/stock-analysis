"""ConfigLoader 的 ${VAR} 解析與 env 注入測試。

「缺變數時保留字面佔位符」是 recommendations_publisher 判斷
publish_folder_id 未設定的依據，行為改變會讓發布悄悄開啟/關閉。
"""

import os

import pytest

from utils.config_loader import ConfigLoader

CONFIG_TEXT = """\
env:
  TESTCL_GLOBAL: "${TESTCL_SRC}"
  TESTCL_UNSET: "${TESTCL_NOPE}"

users:
  kiri:
    shioaji:
      env:
        TESTCL_BROKER_KEY: "${TESTCL_SECRET}"
      constant:
        hold_weeks: 4
        cycle_start_date: "2026-07-06"
"""

ENV_TEXT = "TESTCL_SRC=hello\nTESTCL_SECRET=s3cret\n"


@pytest.fixture(autouse=True)
def _clean_testcl_env():
    yield
    for key in list(os.environ):
        if key.startswith("TESTCL_"):
            del os.environ[key]


@pytest.fixture
def loader(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEXT, encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text(ENV_TEXT, encoding="utf-8")
    return ConfigLoader(config_path=str(config_path), env_path=str(env_path))


def test_placeholder_resolved_from_env_file(loader):
    assert loader.config["env"]["TESTCL_GLOBAL"] == "hello"


def test_missing_var_keeps_literal_placeholder(loader):
    # publisher 靠這個字面殘留判斷「未設定 → 跳過」，不可變
    assert loader.config["env"]["TESTCL_UNSET"] == "${TESTCL_NOPE}"


def test_load_global_env_vars_skips_unresolved(loader):
    loader.load_global_env_vars()
    assert os.environ["TESTCL_GLOBAL"] == "hello"
    assert "TESTCL_UNSET" not in os.environ  # 未解析的不注入


def test_load_global_env_vars_does_not_overwrite(tmp_path):
    os.environ["TESTCL_GLOBAL"] = "already-set"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEXT, encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text(ENV_TEXT, encoding="utf-8")

    loader = ConfigLoader(config_path=str(config_path), env_path=str(env_path))
    loader.load_global_env_vars()
    assert os.environ["TESTCL_GLOBAL"] == "already-set"


def test_load_user_config_injects_env_and_constants(loader):
    loader.load_user_config("kiri", "shioaji")
    assert os.environ["TESTCL_BROKER_KEY"] == "s3cret"
    assert loader.get_user_constant("hold_weeks") == 4
    assert loader.get_user_constant("cycle_start_date") == "2026-07-06"
    assert loader.get_user_constant("nonexistent") is None


def test_load_user_config_unknown_user_raises(loader):
    with pytest.raises(ValueError):
        loader.load_user_config("nobody", "shioaji")
    with pytest.raises(ValueError):
        loader.load_user_config("kiri", "unknown_broker")
