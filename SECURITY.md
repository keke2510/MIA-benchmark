# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in MIA-Bench, please report it responsibly by opening a [private security advisory](https://github.com/keke2510/MIA-benchmark/security/advisories/new).

Please do **not** disclose the vulnerability publicly until it has been addressed.

## Scope

MIA-Bench is a research benchmarking framework for membership inference attacks on machine unlearning. Security concerns are most relevant to:

- **Dependency vulnerabilities** — keep the installed packages (PyTorch, timm, etc.) up to date.
- **Untrusted model or data inputs** — exercise caution when running third-party unlearning algorithms or attack implementations.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.0   | ✅ |

## Disclosure Process

1. Report the vulnerability via a private security advisory.
2. The maintainers will acknowledge within 48 hours and triage the report.
3. A fix will be prepared and released as soon as possible.
4. Credit will be given to the reporter upon request.
