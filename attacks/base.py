from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config


@dataclass(frozen=True)
class AttackContext:
    dataset: str
    net: str
    classes: int
    machine_unlearning: str
    para1: str
    para2: str
    seed: int
    weight_path: str
    num_shadow: int = 8
    num_aug: int = 10
    forget_perc: float = 0.1
    name: str | None = None
    save_name: str | None = None
    attack_sample_number: int | None = None
    attack_options: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_name(self) -> str:
        return self.name or f"{self.net}-{self.dataset}-{self.classes}"

    @property
    def model_save_name(self) -> str:
        return self.save_name or self.net

    def checkpoint_dir(self, task_path: str) -> str:
        return (
            f"{config.CHECKPOINT_PATH}/forget_random_main/"
            f"{self.net}-{self.dataset}-{self.classes}/"
            f"{task_path}/{self.machine_unlearning}_{self.para1}_{self.para2}"
        )

    def attack_result_dir(self, attack_name: str) -> str:
        path = (
            Path("benchmark_results")
            / "samplewise"
            / f"{self.net}-{self.dataset}-{self.classes}"
            / f"{self.machine_unlearning}_{self.para1}_{self.para2}"
            / attack_name
            / f"seed_{self.seed}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


@dataclass
class AttackResult:
    attack_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
