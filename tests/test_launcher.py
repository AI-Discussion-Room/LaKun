"""The single entry point should follow visible GPU count by default."""

import pytest

import train_lakun


def test_automatic_multigpu_launch(monkeypatch):
    commands = []
    monkeypatch.setattr(train_lakun.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(train_lakun.subprocess, "call", lambda command: commands.append(command) or 0)
    with pytest.raises(SystemExit) as finished:
        train_lakun.main(["--epochs", "2"])
    assert finished.value.code == 0
    assert commands[0][commands[0].index("--nproc-per-node") + 1] == "3"
    assert commands[0][-2:] == ["--epochs", "2"]


def test_automatic_single_process(monkeypatch):
    calls = []
    monkeypatch.setattr(train_lakun.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(train_lakun, "training_main", lambda argv: calls.append(argv))
    train_lakun.main(["--epochs", "2"])
    assert calls == [["--epochs", "2"]]


def test_cpu_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(train_lakun.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(train_lakun, "training_main", lambda argv: calls.append(argv))
    train_lakun.main(["--smoke"])
    assert calls == [["--smoke"]]
