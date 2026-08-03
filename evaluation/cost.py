from __future__ import annotations


def build_cost_report(
    *,
    num_shadow: int | None = None,
    num_aug: int | None = None,
    attack_sample_number: int | None = None,
    runtime_sec: float | None = None,
    reminiscence_epochs: int | None = None,
    storage_bytes: int | None = None,
) -> dict[str, int | float | None]:
    query_count = None
    if attack_sample_number is not None and num_aug is not None:
        query_count = int(attack_sample_number * num_aug)

    return {
        "num_shadow": num_shadow,
        "num_aug": num_aug,
        "attack_sample_number": attack_sample_number,
        "query_count": query_count,
        "runtime_sec": runtime_sec,
        "reminiscence_epochs": reminiscence_epochs,
        "storage_bytes": storage_bytes,
    }

