"""Filesystem layout for benchmark runs.

The original codebase stored every shadow model, unlearned model, and attack
result under paths that did not encode the run configuration.  That made it
impossible to run more than one ``num_shadow`` / ``seed`` / ``forget_perc``
combination without silently reusing or overwriting artifacts.

This module centralises the path rules.  Two layouts are supported:

``legacy`` (``run_tag == ""``)
    Byte-for-byte compatible with the paths used by the published 8-shadow,
    10%-forget, seed-42 artifacts.

``tagged`` (``run_tag != ""``)
    Every artifact that depends on ``num_shadow``, ``seed``, or
    ``forget_perc`` is stored in its own directory / filename.  Two runs with
    different configurations can therefore execute concurrently and remain
    reproducible after the fact.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_run_tag(run_tag: str | None) -> str:
    tag = (run_tag or "").strip()
    if tag and not _TAG_RE.match(tag):
        raise ValueError(
            f"Invalid run tag {tag!r}. Only letters, digits, '_', '.', and '-' are allowed."
        )
    return tag


def is_tagged(run_tag: str | None) -> bool:
    return bool(validate_run_tag(run_tag))


def format_perc(forget_perc: float) -> str:
    """Render 0.1 -> '0.1', 0.05 -> '0.05' without trailing zeros."""
    return f"{float(forget_perc):g}"


# ── Shadow models ────────────────────────────────────────────────────────────

def legacy_shadow_stem(save_name: str, shadow_id: int) -> str:
    return f"{save_name}_shadow_{int(shadow_id)}"


def tagged_shadow_stem(
    save_name: str,
    num_shadow: int,
    shadow_seed: int,
    shadow_id: int,
) -> str:
    return (
        f"{save_name}_shadow{int(num_shadow)}"
        f"_seed{int(shadow_seed)}_id{int(shadow_id)}"
    )


def shadow_stem(
    save_name: str,
    num_shadow: int,
    shadow_seed: int,
    shadow_id: int,
    *,
    tagged: bool,
) -> str:
    if tagged:
        return tagged_shadow_stem(save_name, num_shadow, shadow_seed, shadow_id)
    return legacy_shadow_stem(save_name, shadow_id)


def shadow_checkpoint_path(
    repo_root: Path | str,
    experiment_name: str,
    save_name: str,
    num_shadow: int,
    shadow_seed: int,
    shadow_id: int,
    *,
    tagged: bool,
) -> Path:
    return (
        Path(repo_root)
        / "saved_models"
        / experiment_name
        / f"{shadow_stem(save_name, num_shadow, shadow_seed, shadow_id, tagged=tagged)}_last.pth"
    )


def shadow_checkpoint_candidates(
    repo_root: Path | str,
    experiment_name: str,
    save_name: str,
    num_shadow: int,
    shadow_seed: int,
    shadow_id: int,
    *,
    tagged: bool,
) -> Iterator[Path]:
    """Yield the primary checkpoint, then the legacy fallback when eligible.

    The published 8-shadow seed-42 checkpoints predate tagged filenames.  They
    were trained with exactly the protocol we still use (8 shadows, pkeep=0.83,
    seed=42), so tagged runs may reuse them instead of retraining.
    """
    primary = shadow_checkpoint_path(
        repo_root, experiment_name, save_name, num_shadow, shadow_seed, shadow_id,
        tagged=tagged,
    )
    yield primary
    if tagged and int(num_shadow) == 8 and int(shadow_seed) == 42:
        legacy = shadow_checkpoint_path(
            repo_root, experiment_name, save_name, 8, 42, shadow_id, tagged=False,
        )
        if legacy != primary:
            yield legacy


def resolve_shadow_checkpoint(
    repo_root: Path | str,
    experiment_name: str,
    save_name: str,
    num_shadow: int,
    shadow_seed: int,
    shadow_id: int,
    *,
    tagged: bool,
) -> Path:
    """Return the first existing candidate, or the primary path if none exist."""
    candidates = list(
        shadow_checkpoint_candidates(
            repo_root, experiment_name, save_name, num_shadow, shadow_seed, shadow_id,
            tagged=tagged,
        )
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# ── Unlearned / reminiscence checkpoints ─────────────────────────────────────

def method_key(method_name: str, para1: str, para2: str) -> str:
    return f"{method_name}_{para1}_{para2}"


def task_dir(
    checkpoint_root: Path | str,
    experiment_name: str,
    task: str,
    method_key_value: str | None,
    *,
    run_tag: str = "",
    forget_perc: float = 0.1,
    seed: int = 42,
) -> Path:
    base = (
        Path(checkpoint_root)
        / "forget_random_main"
        / experiment_name
        / task
    )
    if method_key_value is not None:
        base = base / method_key_value
    tag = validate_run_tag(run_tag)
    if tag:
        base = base / tag / f"forget{format_perc(forget_perc)}" / f"seed{int(seed)}"
    return base


def forget_index_path(
    checkpoint_root: Path | str,
    experiment_name: str,
    forget_perc: float,
) -> Path:
    """The forget set is part of the benchmark definition and is shared by seeds."""
    return (
        Path(checkpoint_root)
        / "forget_random_main"
        / experiment_name
        / "random_index_set"
        / f"forgetting_dataset_index_{format_perc(forget_perc)}.npy"
    )


# ── Attack results ───────────────────────────────────────────────────────────

def attack_result_dir(
    repo_root: Path | str,
    experiment_name: str,
    method_key_value: str,
    attack_name: str,
    *,
    run_tag: str = "",
    forget_perc: float = 0.1,
    seed: int = 42,
) -> Path:
    base = (
        Path(repo_root)
        / "benchmark_results"
        / "samplewise"
        / experiment_name
        / method_key_value
        / attack_name
    )
    tag = validate_run_tag(run_tag)
    if tag:
        base = base / tag / f"forget{format_perc(forget_perc)}"
    return base / f"seed_{int(seed)}"


def run_summary_path(
    repo_root: Path | str,
    experiment_name: str,
    *,
    run_tag: str = "",
    forget_perc: float = 0.1,
    seed: int = 42,
    suffix: str = "",
) -> Path:
    base = Path(repo_root) / "benchmark_results" / "samplewise" / experiment_name
    tag = validate_run_tag(run_tag)
    if tag:
        return base / (
            f"run_summary_{tag}_forget{format_perc(forget_perc)}"
            f"_seed{int(seed)}{suffix}.json"
        )
    return base / f"run_summary_seed_{int(seed)}{suffix}.json"
