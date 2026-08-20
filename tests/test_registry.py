"""Smoke tests for the MIA-Bench attack registry and data model.

These tests are pure-Python (no PyTorch required) and exercise the core
extensibility mechanisms of the framework.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    """Load a module by path without triggering its package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register before exec so dataclasses resolve
    spec.loader.exec_module(module)
    return module


def test_register_and_get():
    registry = _load_module("registry", REPO_ROOT / "attacks" / "registry.py")

    @registry.register_attack("dummy")
    class DummyAttack:
        pass

    assert registry.get_attack("dummy") is DummyAttack
    assert "dummy" in registry.list_attacks()


def test_unknown_attack_raises():
    registry = _load_module("registry", REPO_ROOT / "attacks" / "registry.py")
    try:
        registry.get_attack("nonexistent_attack")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown attack")


def test_attack_context_defaults():
    base = _load_module("base", REPO_ROOT / "attacks" / "base.py")
    ctx = base.AttackContext(
        dataset="Cifar10",
        net="ResNet18",
        classes=10,
        machine_unlearning="finetune",
        para1="0.11",
        para2="10",
        seed=1,
        weight_path="dummy.pth",
    )
    assert ctx.experiment_name == "ResNet18-Cifar10-10"
    assert ctx.num_shadow == 8
    assert ctx.forget_perc == 0.1
    assert "finetune_0.11_10" in ctx.checkpoint_dir("unlearning")


def test_attack_result_serializable():
    base = _load_module("base", REPO_ROOT / "attacks" / "base.py")
    result = base.AttackResult(attack_name="dummy", metrics={"auc": 0.5})
    d = result.as_dict()
    assert d["attack_name"] == "dummy"
    assert d["metrics"]["auc"] == 0.5
