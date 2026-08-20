"""
MIA-Bench Supplementary Experiment Runner
==========================================
Runs ONLY the missing experiments:
  1. CINIC-10 full benchmark: 4 methods x 4 attacks x 3 seeds = 48 groups
  2. Runtime: CIFAR-100 + TinyImageNet + CINIC-10

Existing CIFAR-10/100/TinyImageNet benchmark data is NOT re-run.
Runtime for CIFAR-10 already exists in the paper.

Usage:
    python run_supplementary.py           # run all
    python run_supplementary.py --dry-run # verify commands
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_BIN = sys.executable

# ── What we need to run ──────────────────────────────
SEEDS = [42, 123, 999]
METHODS = {
    "retrain":       ["0.1", "150"],
    "finetune":      ["0.11", "10"],
    "negative_grad": ["0.04", "8"],
    "scrub":         ["0.0004", "7"],
}
ATTACKS = ["lira", "rea", "ruli", "unlearningleaks"]
RUNTIME_DATASETS = ["CIFAR-100", "TinyImageNet", "CINIC-10"]  # CIFAR-10 already done

def run(cmd, dry_run=False, timeout=None):
    print(f"\n[CMD] {' '.join(cmd)}")
    if dry_run:
        return True
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip CINIC-10 benchmark")
    parser.add_argument("--skip-runtime", action="store_true", help="Skip runtime measurements")
    args = parser.parse_args()

    benchmark_script = str(REPO_ROOT / "run_benchmark.py")
    runtime_script = str(REPO_ROOT / "measure_runtime.py")
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "results": []}

    total = len(METHODS) * len(ATTACKS) * len(SEEDS) + len(RUNTIME_DATASETS)
    current = 0

    print("MIA-Bench Supplementary Experiments")
    print(f"  CINIC-10 benchmark: {len(METHODS)} methods x {len(ATTACKS)} attacks x {len(SEEDS)} seeds")
    print(f"  Runtime datasets: {RUNTIME_DATASETS}")
    print(f"  Dry run: {args.dry_run}")
    print()

    if not args.skip_benchmark:
        # ── CINIC-10: Pretrain + Shadow (once, reused) ──
        print(f"\n{'#'*50}\n# CINIC-10: Pretrain\n{'#'*50}")
        run([PYTHON_BIN, benchmark_script,
             "--dataset", "Cinic10", "--seed", str(SEEDS[0]),
             "--methods", "finetune:0.11:10", "--attacks", "lira",
             "--stages", "pretrain"], dry_run=args.dry_run, timeout=7200)

        print(f"\n{'#'*50}\n# CINIC-10: Shadow Models\n{'#'*50}")
        run([PYTHON_BIN, benchmark_script,
             "--dataset", "Cinic10", "--seed", str(SEEDS[0]),
             "--methods", "finetune:0.11:10",
             "--attacks"] + ATTACKS + ["--stages", "shadow"],
            dry_run=args.dry_run, timeout=14400)

        # ── CINIC-10: All method x attack x seed ──
        for method_name, (para1, para2) in METHODS.items():
            method_str = f"{method_name}:{para1}:{para2}"
            for seed in SEEDS:
                current += 1
                print(f"\n{'#'*50}\n# [{current}/{total}] Cinic10 | {method_str} | seed={seed}\n{'#'*50}")

                ok = run([PYTHON_BIN, benchmark_script,
                     "--dataset", "Cinic10", "--seed", str(seed),
                     "--methods", method_str,
                     "--attacks"] + ATTACKS + ["--stages", "unlearn", "attack"],
                    dry_run=args.dry_run, timeout=28800)

                log["results"].append({
                    "dataset": "Cinic10", "method": method_str,
                    "seed": seed, "success": ok,
                })

    if not args.skip_runtime:
        # ── Runtime for CIFAR-100, TinyImageNet, CINIC-10 ──
        for ds in RUNTIME_DATASETS:
            current += 1
            print(f"\n{'#'*50}\n# [{current}/{total}] Runtime: {ds}\n{'#'*50}")
            ok = run([PYTHON_BIN, runtime_script, "--dataset", ds],
                    dry_run=args.dry_run, timeout=7200)
            log["results"].append({
                "type": "runtime", "dataset": ds, "success": ok,
            })

    # ── Save ────────────────────────────────────────────
    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path = REPO_ROOT / "supplementary_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    succeeded = sum(1 for r in log["results"] if r.get("success"))
    print(f"\n{'='*50}")
    print(f"COMPLETE: {succeeded}/{len(log['results'])} succeeded")
    print(f"Log: {log_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
