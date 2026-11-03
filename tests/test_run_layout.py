"""Tests for the benchmark artifact layout.

The published results (8 shadows, seed 42, 10% forget) must keep their exact
legacy paths, while the TIFS revision experiments use tagged paths that encode
``num_shadow`` / ``shadow_seed`` / ``forget_perc`` / ``seed``.

These tests are pure-Python (no PyTorch required).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark import run_layout as layout  # noqa: E402


def test_format_perc():
    assert layout.format_perc(0.1) == "0.1"
    assert layout.format_perc(0.05) == "0.05"
    assert layout.format_perc(0.15) == "0.15"
    assert layout.format_perc(0.2) == "0.2"
    assert layout.format_perc(10.0) == "10"


def test_validate_run_tag():
    assert layout.validate_run_tag("") == ""
    assert layout.validate_run_tag("e1_s16") == "e1_s16"
    for bad in ("a/b", "a b", "a\\b", "tag:1"):
        try:
            layout.validate_run_tag(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_legacy_shadow_path_matches_published_layout():
    path = layout.shadow_checkpoint_path(
        REPO_ROOT, "ResNet18-Cifar10-10", "ResNet18", 8, 42, 3, tagged=False
    )
    assert path == (
        REPO_ROOT / "saved_models" / "ResNet18-Cifar10-10"
        / "ResNet18_shadow_3_last.pth"
    )


def test_tagged_shadow_path_encodes_configuration():
    path = layout.shadow_checkpoint_path(
        REPO_ROOT, "ResNet18-Cifar10-10", "ResNet18", 16, 42, 3, tagged=True
    )
    assert path.name == "ResNet18_shadow16_seed42_id3_last.pth"
    other = layout.shadow_checkpoint_path(
        REPO_ROOT, "ResNet18-Cifar10-10", "ResNet18", 32, 42, 3, tagged=True
    )
    assert path != other


def test_legacy_shadow_fallback_only_for_published_protocol():
    def candidates(num_shadow, shadow_seed):
        return list(
            layout.shadow_checkpoint_candidates(
                REPO_ROOT, "ResNet18-Cifar10-10", "ResNet18",
                num_shadow, shadow_seed, 0, tagged=True,
            )
        )

    published = candidates(8, 42)
    assert len(published) == 2
    assert published[1].name == "ResNet18_shadow_0_last.pth"

    assert len(candidates(16, 42)) == 1
    assert len(candidates(8, 123)) == 1


def test_legacy_task_dir_matches_published_layout():
    path = layout.task_dir(
        "log_files/model",
        "ResNet18-Cifar10-10",
        "unlearning",
        "retrain_0.1_150",
    )
    assert path == Path(
        "log_files/model/forget_random_main/ResNet18-Cifar10-10"
        "/unlearning/retrain_0.1_150"
    )


def test_tagged_task_dir_encodes_forget_and_seed():
    path = layout.task_dir(
        "log_files/model",
        "ResNet18-Cifar10-10",
        "unlearning",
        "retrain_0.1_150",
        run_tag="e1_s16",
        forget_perc=0.05,
        seed=123,
    )
    assert path == Path(
        "log_files/model/forget_random_main/ResNet18-Cifar10-10"
        "/unlearning/retrain_0.1_150/e1_s16/forget0.05/seed123"
    )


def test_forget_index_is_shared_across_seeds_and_tags():
    path = layout.forget_index_path("log_files/model", "ResNet18-Cifar10-10", 0.15)
    assert path.name == "forgetting_dataset_index_0.15.npy"
    assert "seed" not in str(path)
    assert path == layout.forget_index_path(
        "log_files/model", "ResNet18-Cifar10-10", 0.15
    )


def test_attack_result_dir_legacy_and_tagged():
    legacy = layout.attack_result_dir(
        REPO_ROOT, "ResNet18-Cifar10-10", "retrain_0.1_150", "lira", seed=123,
    )
    assert legacy == (
        REPO_ROOT / "benchmark_results" / "samplewise" / "ResNet18-Cifar10-10"
        / "retrain_0.1_150" / "lira" / "seed_123"
    )

    tagged = layout.attack_result_dir(
        REPO_ROOT, "ResNet18-Cifar10-10", "retrain_0.1_150", "lira",
        run_tag="e2_seedstab", forget_perc=0.1, seed=123,
    )
    assert tagged == (
        REPO_ROOT / "benchmark_results" / "samplewise" / "ResNet18-Cifar10-10"
        / "retrain_0.1_150" / "lira" / "e2_seedstab" / "forget0.1" / "seed_123"
    )


def test_run_summary_path_is_unique_per_tag_and_seed():
    paths = {
        layout.run_summary_path(REPO_ROOT, "ResNet18-Cifar10-10", seed=42),
        layout.run_summary_path(
            REPO_ROOT, "ResNet18-Cifar10-10", run_tag="e1_s16", seed=42
        ),
        layout.run_summary_path(
            REPO_ROOT, "ResNet18-Cifar10-10", run_tag="e1_s32", seed=42
        ),
        layout.run_summary_path(
            REPO_ROOT, "ResNet18-Cifar10-10", run_tag="e1_s16", seed=123
        ),
    }
    assert len(paths) == 4


def test_attack_context_effective_shadow_seed():
    from attacks.base import AttackContext

    ctx = AttackContext(
        dataset="Cifar10", net="ResNet18", classes=10,
        machine_unlearning="retrain", para1="0.1", para2="150",
        seed=123, weight_path="dummy.pth",
    )
    assert ctx.effective_shadow_seed == 123
    ctx = AttackContext(
        dataset="Cifar10", net="ResNet18", classes=10,
        machine_unlearning="retrain", para1="0.1", para2="150",
        seed=123, weight_path="dummy.pth", shadow_seed=42,
    )
    assert ctx.effective_shadow_seed == 42
