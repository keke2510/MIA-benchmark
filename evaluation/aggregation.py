from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def summarize_runs(records: list[dict], metric_keys: list[str]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        metrics = record.get("metrics", {})
        for key in metric_keys:
            value = metrics.get(key)
            if value is not None:
                grouped[key].append(float(value))

    summary: dict[str, dict[str, float | None]] = {}
    for key in metric_keys:
        values = grouped.get(key, [])
        if not values:
            summary[key] = {"mean": None, "std": None, "cv": None}
            continue
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        cv = None if math.isclose(mean, 0.0) else float(std / mean)
        summary[key] = {"mean": mean, "std": std, "cv": cv}

    return summary

