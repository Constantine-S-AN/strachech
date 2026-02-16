from __future__ import annotations

from pathlib import Path

import pytest
from stratcheck.cli import create_strategy, main


def test_create_strategy_generates_template_and_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    strategy_path, config_path = create_strategy("MyStrategy")

    assert strategy_path == Path("stratcheck/strategies/my_strategy.py")
    assert config_path == Path("configs/examples/my_strategy.toml")
    assert strategy_path.exists()
    assert config_path.exists()
    assert Path("stratcheck/strategies/__init__.py").exists()

    strategy_text = strategy_path.read_text(encoding="utf-8")
    config_text = config_path.read_text(encoding="utf-8")
    assert "class MyStrategy(StrategyTemplate)" in strategy_text
    assert "def build_signals" in strategy_text
    assert 'strategy = "stratcheck.strategies.my_strategy:MyStrategy"' in config_text
    assert "[strategy_params]" in config_text


def test_create_strategy_existing_file_requires_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    create_strategy("MyStrategy")

    with pytest.raises(FileExistsError):
        create_strategy("MyStrategy")

    strategy_path, config_path = create_strategy("MyStrategy", force=True)
    assert strategy_path.exists()
    assert config_path.exists()


def test_main_create_strategy_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["create-strategy", "AlphaStrategy"])

    assert exit_code == 0
    assert (tmp_path / "stratcheck/strategies/alpha_strategy.py").exists()
    assert (tmp_path / "configs/examples/alpha_strategy.toml").exists()
