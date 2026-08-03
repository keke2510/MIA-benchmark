from __future__ import annotations

from typing import Callable


ATTACK_REGISTRY: dict[str, type] = {}


def register_attack(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        ATTACK_REGISTRY[name] = cls
        cls.attack_name = name
        return cls

    return decorator


def get_attack(name: str) -> type:
    if name not in ATTACK_REGISTRY:
        raise KeyError(f"Unknown attack '{name}'. Available: {sorted(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name]


def list_attacks() -> list[str]:
    return sorted(ATTACK_REGISTRY)

