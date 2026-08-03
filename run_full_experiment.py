"""
MIA-Bench Full Experiment Runner
================================
Runs the complete benchmark: 4 datasets x 4 unlearning methods x 4 attacks x 3 seeds.
Also measures runtime for all 6 attacks on each dataset.
Total: ~192 main experiment groups + runtime benchmarks.

Usage:
    python run_full_experiment.py                    # full matrix
    python run_full_experiment.py --dry-run          # print commands only
    python run_full_experiment.py --dataset Cifar10  # single dataset
    python run_full_experiment.py --skip-pretrain     # skip base model training

Output:
    benchmark_results/samplewise/<experiment_name>/   # per-run results
    runtime_results.json                              # timing data
    full_experiment_log.json                          # progress log
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ── Experiment Matrix ───────────────────────────────
DATASETS = ["Cifar10", "Cifar100", "TinyImageNet", "Cinic10"]
METHODS = {
    "retrain":       ["0.1", "150"],
    "finetune":      ["0.11", "10"],
    "negative_grad": ["0.04", "8"],
    "scrub":         ["0.0004", "7"],
}
ATTACKS = ["lira", "rea", "ruli", "unlearningleaks"]
SEEDS = [42, 123, 999]

# ── Helpers ──────────────────────────────────────────
def run_cmd(cmd, dry_run=False, timeout=None):
    """Run a command, print output, return success."""
    print(f"\n{'='*60}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'='*60}")
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

def method_key(method_name, para1, para2):
    return f"{method_name}_{para1}_{para2}"

# ── Main ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MIA-Bench Full Experiment Runner")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--dataset", type=str, help="Run single dataset only")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip base model pretraining")
    parser.add_argument("--skip-runtime", action="store_true", help="Skip runtime measurement")
    parser.add_argument("--start-seed", type=int, default=0, help="Seed index to start from (0-2)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    seeds = SEEDS[args.start_seed:]

    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "results": []}
    total_groups = len(datasets) * len(METHODS) * len(ATTACKS) * len(seeds)
    current = 0

    print(f"MIA-Bench Full Experiment")
    print(f"  Datasets: {datasets}")
    print(f"  Methods: {list(METHODS.keys())}")
    print(f"  Attacks: {ATTACKS}")
    print(f"  Seeds: {seeds}")
    print(f"  Total groups: ~{total_groups}")
    print(f"  Dry run: {args.dry_run}")
    print()

    python_bin = sys.executable
    benchmark_script = str(REPO_ROOT / "run_benchmark.py")

    for dataset in datasets:
        # ── Stage 1: Pretrain base model ─────────────
        if not args.skip_pretrain:
            print(f"\n{'#'*60}")
            print(f"# DATASET: {dataset} — Pretrain")
            print(f"{'#'*60}")
            ok = run_cmd([
                python_bin, benchmark_script,
                "--dataset", dataset,
                "--seed", str(seeds[0]),
                "--methods", "finetune:0.11:10",
                "--attacks", "lira",
                "--stages", "pretrain",
            ], dry_run=args.dry_run, timeout=7200)
            if not ok and not args.dry_run:
                print(f"  WARNING: Pretrain may have failed for {dataset}")

        # ── Stage 2: Shadow models (reusable) ────────
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset} — Shadow Models")
        print(f"{'#'*60}")
        run_cmd([
            python_bin, benchmark_script,
            "--dataset", dataset,
            "--seed", str(seeds[0]),
            "--methods", "finetune:0.11:10",
            "--attacks", " ".join(ATTACKS),
            "--stages", "shadow",
        ], dry_run=args.dry_run, timeout=14400)

        # ── Stage 3: Full matrix ─────────────────────
        for method_name, (para1, para2) in METHODS.items():
            method_str = f"{method_name}:{para1}:{para2}"
            for seed in seeds:
                # Determine stages
                if method_name == "retrain":
                    stages = "unlearn attack"
                elif method_name == "rea":
                    # REA needs reminiscence stage
                    continue  # REA is an attack, not a method — handled below

                # Build the unlearn + attack stages
                stages_list = ["unlearn"]
                stages_list.append("attack")
                stages = " ".join(stages_list)

                current += 1
                print(f"\n{'#'*60}")
                print(f"# [{current}/{total_groups}] {dataset} | {method_str} | seed={seed} | all attacks")
                print(f"{'#'*60}")

                ok = run_cmd([
                    python_bin, benchmark_script,
                    "--dataset", dataset,
                    "--seed", str(seed),
                    "--methods", method_str,
                    "--attacks", " ".join(ATTACKS),
                    "--stages", stages,
                ], dry_run=args.dry_run, timeout=28800)

                log["results"].append({
                    "dataset": dataset,
                    "method": method_str,
                    "seed": seed,
                    "attacks": ATTACKS,
                    "success": ok,
                })

            # ── REA attack (needs reminiscence stage) ──
            # REA is handled within the attack stage via the benchmark runner
            # The benchmark runner automatically adds reminiscence when attack=rea

        # ── Runtime measurement ───────────────────────
        if not args.skip_runtime:
            print(f"\n{'#'*60}")
            print(f"# DATASET: {dataset} — Runtime Measurement")
            print(f"{'#'*60}")
            runtime_script = str(REPO_ROOT / "measure_runtime.py")
            if os.path.exists(runtime_script):
                run_cmd([python_bin, runtime_script, "--dataset", dataset],
                        dry_run=args.dry_run, timeout=7200)
            else:
                print("  measure_runtime.py not found — skipping")

    # ── Save log ─────────────────────────────────────
    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path = REPO_ROOT / "full_experiment_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    succeeded = sum(1 for r in log["results"] if r["success"])
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"  Total: {len(log['results'])}, Succeeded: {succeeded}, Failed: {len(log['results'])-succeeded}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()