from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config
from benchmark.run_layout import attack_result_dir as layout_attack_result_dir
from benchmark.run_layout import method_key as layout_method_key
from benchmark.run_layout import task_dir as layout_task_dir


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
    # Size of the trained pool the checkpoints come from.  ``None`` means the
    # pool size equals ``num_shadow`` (the classic non-nested setting).  E1
    # trains one pool of 32 and evaluates the first 8/16/32 shadows of it.
    shadow_pool: int | None = None
    num_aug: int = 10
    forget_perc: float = 0.1
    run_tag: str = ""
    shadow_seed: int | None = None
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

    @property
    def effective_shadow_seed(self) -> int:
        return self.seed if self.shadow_seed is None else int(self.shadow_seed)

    @property
    def effective_shadow_pool(self) -> int:
        return self.num_shadow if self.shadow_pool is None else int(self.shadow_pool)

    def checkpoint_dir(self, task_path: str) -> str:
        return str(
            layout_task_dir(
                config.CHECKPOINT_PATH,
                self.experiment_name,
                task_path,
                layout_method_key(self.machine_unlearning, self.para1, self.para2),
                run_tag=self.run_tag,
                forget_perc=self.forget_perc,
                seed=self.seed,
            )
        )

    def attack_result_dir(self, attack_name: str) -> str:
        path = layout_attack_result_dir(
            Path("."),
            self.experiment_name,
            layout_method_key(self.machine_unlearning, self.para1, self.para2),
            attack_name,
            run_tag=self.run_tag,
            forget_perc=self.forget_perc,
            seed=self.seed,
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
