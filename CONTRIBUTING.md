# Contributing to MIA-Bench

Thank you for your interest in contributing to MIA-Bench! We welcome contributions of all kinds, including bug fixes, new attacks, new unlearning algorithms, documentation improvements, and benchmark extensions.

## How to Contribute

### Reporting Bugs

Please open an issue using the **Bug Report** template and include:

- A clear, descriptive title.
- Steps to reproduce the issue.
- Expected vs. actual behavior.
- Your environment (Python version, PyTorch version, GPU, OS).
- Relevant logs or error messages.

### Suggesting Features

Please open an issue using the **Feature Request** template and describe:

- The motivation for the feature.
- A clear description of the proposed change.
- Any relevant references or prior work.

### Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, following the existing code style.
3. Add or update tests where applicable.
4. Run the test suite to ensure nothing is broken:
   ```bash
   python -m pytest tests/ -v
   ```
5. Commit with a clear message and open a pull request against `main`.

## Adding a New Attack

MIA-Bench uses a lightweight registry pattern to keep attacks pluggable:

1. Create a new file under `attacks/` (e.g., `attacks/myattack.py`).
2. Define an attack class implementing `run(context, dry_run=False) -> AttackResult`.
3. Register it with the `@register_attack("myattack")` decorator.
4. Import the module in `attacks/__init__.py` so it is auto-registered.

```python
from attacks.base import AttackContext, AttackResult
from attacks.registry import register_attack

@register_attack("myattack")
class MyAttack:
    @classmethod
    def run(cls, context: AttackContext, dry_run: bool = False) -> AttackResult | None:
        # ... implement your attack ...
        return AttackResult(attack_name="myattack", metrics={...})
```

## Adding a New Unlearning Algorithm

Unlearning algorithms are orchestrated through `benchmark/samplewise.py`. To add a new algorithm, register its hyperparameters in `ALL_METHODS` and provide the corresponding implementation in `forget_random_strategies.py`.

## Code Style

- Follow PEP 8.
- Use type hints where feasible.
- Write descriptive docstrings for public interfaces.
- Keep changes focused and atomic.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
