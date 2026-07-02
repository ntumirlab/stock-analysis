"""config.yaml 語法與關鍵欄位的守門測試（CI gate：config 改壞就擋下部署）。"""

import os
import yaml
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def config():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_yaml_parses(config):
    assert isinstance(config, dict)


def test_kiri_golden_ai_constants(config):
    constant = config["users"]["kiri"]["shioaji"]["constant"]
    assert constant["strategy_class_name"] == "GoldenAIStrategy"
    assert constant["golden_ai_frequency"] in ("weekly", "monthly", "weekly_4w")
    assert isinstance(constant["hold_weeks"], int) and constant["hold_weeks"] >= 1
    # cycle_start_date 必須存在且是 YYYY-MM-DD 字串（order_executor 啟動時依賴）
    from datetime import datetime
    datetime.strptime(str(constant["cycle_start_date"]), "%Y-%m-%d")


def test_golden_ai_strategy_sections(config):
    for task in ("weekly", "monthly", "weekly_4w"):
        section = config["golden_ai"][task]
        assert 1 <= section["buy_weekday"] <= 5
        assert 1 <= section["sell_weekday"] <= 5
        assert section["rank_start"] >= 1
        assert section["rank_end"] >= section["rank_start"]


def test_recommendation_tasks_defined(config):
    for task in ("weekly", "monthly"):
        assert "local_dir" in config["recommendation_tasks"][task]
