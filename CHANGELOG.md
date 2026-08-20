# Changelog

All notable changes to MIA-Bench are documented in this file.

## [1.0.0] - 2026-08

### Added

- **6 attack methods**: LiRA, REA, RULI, UnlearningLeaks (primary) + Threshold, Loss (black-box baselines).
- **4 unlearning algorithms**: Retrain, Finetuning, NegGrad, SCRUB.
- **4 benchmark datasets**: CIFAR-10, CIFAR-100, TinyImageNet, CINIC-10.
- **Five-dimensional evaluation framework**: Effectiveness, Stability, Applicability, Cost, Practicality.
- Unified registry-based attack API (`attacks/registry.py`, `attacks/base.py`).
- Full 288-setting experiment matrix runner (`run_full_experiment.py`).
- Per-attack runtime profiling (`measure_runtime.py`).
- Supplementary experiment runner and figure plotting scripts (`run_supplementary.py`, `scripts/plot_figures.py`).
- Comprehensive documentation (README, CITATION.cff, CONTRIBUTING.md).
- MIT License.

### Fixed

- Standardized TPR@FPR thresholds (0.1%, 1%, 10%) across all reporting.
- Unified 288 experimental settings across the paper and repository.
