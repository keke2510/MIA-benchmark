"""Runner for the TIFS revision experiments.

Implements the three experiment blocks agreed in ``补实验量清单.md``:

* ``e1`` -- shadow scaling, 16 and 32 shadows (strict retraining, seed 42)
* ``e2`` -- seed stability, 5 seeds with per-seed shadow models (protocol B)
* ``e3`` -- forget-rate sensitivity, 5% / 15% / 20% on CIFAR-10 and CIFAR-100

All runs use the tagged layout (``--run-tag``), so they never overwrite the
published 8-shadow / 10% / 3-seed results.

Usage
-----
    # Inspect the full matrix without touching the GPU
    python run_missing_experiments.py --dry-run

    # Run everything on 4 GPUs
    python run_missing_experiments.py --gpus 0,1,2,3

    # Only E1 on one GPU, resuming an interrupted run
    python run_missing_experiments.py --experiments e1 --gpus 0 --resume

    # Calibration: train exactly one shadow (id 0) on one GPU, then time it
    python run_missing_experiments.py --experiments e1 --datasets Cifar10 \
        --phase shadow --gpus 0 --limit 1
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import config

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_BIN = sys.executable
BENCHMARK_SCRIPT = str(REPO_ROOT / "run_benchmark.py")

DATASETS = ["Cifar10", "Cifar100", "Cinic10", "TinyImageNet"]

# Mirrors ``benchmark.samplewise.DATASET_DEFAULTS`` (net, classes) without
# importing torch just to run a preflight check.
DATASET_SPECS = {
    "Cifar10": ("ResNet18", 10),
    "Cifar100": ("ResNet50", 100),
    "Cinic10": ("ResNet18", 10),
    "TinyImageNet": ("ViT", 200),
}

# Same hyper-parameters as the published matrix.
METHODS: dict[str, dict[str, tuple[str, str]]] = {
    "Cifar10": {
        "retrain": ("0.1", "150"),
        "finetune": ("0.11", "10"),
        "negative_grad": ("0.04", "8"),
        "scrub": ("0.0004", "7"),
    },
    "Cifar100": {
        "retrain": ("0.1", "150"),
        "finetune": ("0.11", "10"),
        "negative_grad": ("0.04", "8"),
        "scrub": ("0.0004", "7"),
    },
    "Cinic10": {
        "retrain": ("0.1", "150"),
        "finetune": ("0.11", "10"),
        "negative_grad": ("0.04", "8"),
        "scrub": ("0.0004", "7"),
    },
    "TinyImageNet": {
        "retrain": ("3e-4", "30"),
        "finetune": ("8e-4", "10"),
        "negative_grad": ("1e-3", "12"),
        "scrub": ("4e-4", "10"),
    },
}

PRIMARY_ATTACKS = ["lira", "rea", "ruli", "unlearningleaks"]
BASELINE_ATTACKS = ["threshold", "loss"]
ALL_ATTACKS = PRIMARY_ATTACKS + BASELINE_ATTACKS

E1_SHADOWS = [16, 32]
E1_SEEDS = [42, 123, 999]
E1_SHADOW_SEED = 42

E2_SEEDS = [123, 999, 7, 2026]  # seed 42 is the published baseline
E2_NUM_SHADOW = 8

E3_DATASETS = ["Cifar10", "Cifar100"]
E3_PERCS = [0.05, 0.15, 0.2]
E3_SEEDS = [42, 123, 999]
E3_SHADOW_SEED = 42
E3_ATTACKS = ["lira", "rea", "unlearningleaks"] + BASELINE_ATTACKS


def e1_tag(num_shadow: int) -> str:
    return f"e1_s{num_shadow}"


E2_TAG = "e2_seedstab"
E3_TAG = "e3_forget"


@dataclass(frozen=True)
class RunSpec:
    job_id: str
    phase: str
    experiment: str
    dataset: str
    method: str | None
    seed: int
    shadow_seed: int
    num_shadow: int
    forget_perc: float
    run_tag: str
    attacks: tuple[str, ...]
    stages: tuple[str, ...]
    shadow_ids: tuple[int, ...] | None = None

    def command(self) -> list[str]:
        cmd = [
            PYTHON_BIN,
            BENCHMARK_SCRIPT,
            "--dataset", self.dataset,
            "--seed", str(self.seed),
            "--shadow-seed", str(self.shadow_seed),
            "--run-tag", self.run_tag,
            "--num-shadow", str(self.num_shadow),
            "--ruli-shadow-num", str(self.num_shadow),
            "--unlearningleaks-num-shadow", str(self.num_shadow),
            "--forget-perc", f"{self.forget_perc:g}",
        ]
        if self.method is not None:
            para1, para2 = METHODS[self.dataset][self.method]
            cmd += ["--methods", f"{self.method}:{para1}:{para2}"]
        else:
            # The shadow stage does not use the method, but the CLI requires a
            # syntactically valid spec; reuse the dataset's finetune preset.
            para1, para2 = METHODS[self.dataset]["finetune"]
            cmd += ["--methods", f"finetune:{para1}:{para2}"]
        cmd += ["--attacks", *self.attacks]
        cmd += ["--stages", *self.stages]
        if self.shadow_ids is not None:
            cmd += ["--shadow-ids", *(str(i) for i in self.shadow_ids)]
        return cmd


def _job_id(
    phase: str,
    experiment: str,
    dataset: str,
    method: str | None,
    seed: int,
    shadow_seed: int,
    num_shadow: int,
    forget_perc: float,
    shadow_id: int | None = None,
) -> str:
    parts = [
        phase,
        experiment,
        dataset,
        method or "shadows",
        f"seed{seed}",
        f"shadowseed{shadow_seed}",
        f"s{num_shadow}",
        f"f{forget_perc:g}",
    ]
    if shadow_id is not None:
        parts.append(f"id{shadow_id}")
    return "__".join(parts)


def _shadow_jobs_for_config(
    experiment: str,
    dataset: str,
    *,
    seed: int,
    shadow_seed: int,
    num_shadow: int,
    run_tag: str,
    forget_perc: float = 0.1,
) -> list[RunSpec]:
    """One job per shadow id.

    ``train.py`` derives the shadow split from ``(num_shadow, seed)`` only, so
    splitting the ids across jobs does not change the protocol.  Per-id jobs
    let the scheduler fill every GPU and keep each job short enough that a
    timeout does not discard hours of work.
    """
    return [
        RunSpec(
            job_id=_job_id(
                "shadow", experiment, dataset, None, seed, shadow_seed,
                num_shadow, forget_perc, shadow_id,
            ),
            phase="shadow",
            experiment=experiment,
            dataset=dataset,
            method=None,
            seed=seed,
            shadow_seed=shadow_seed,
            num_shadow=num_shadow,
            forget_perc=forget_perc,
            run_tag=run_tag,
            attacks=("lira",),
            stages=("shadow",),
            shadow_ids=(shadow_id,),
        )
        for shadow_id in range(num_shadow)
    ]


def build_shadow_jobs(experiments: Sequence[str], datasets: Sequence[str]) -> list[RunSpec]:
    jobs: list[RunSpec] = []
    if "e1" in experiments:
        for num_shadow in E1_SHADOWS:
            for dataset in datasets:
                jobs.extend(
                    _shadow_jobs_for_config(
                        "e1", dataset,
                        seed=E1_SHADOW_SEED,
                        shadow_seed=E1_SHADOW_SEED,
                        num_shadow=num_shadow,
                        run_tag=e1_tag(num_shadow),
                    )
                )
    if "e2" in experiments:
        for dataset in datasets:
            for seed in E2_SEEDS:
                jobs.extend(
                    _shadow_jobs_for_config(
                        "e2", dataset,
                        seed=seed,
                        shadow_seed=seed,
                        num_shadow=E2_NUM_SHADOW,
                        run_tag=E2_TAG,
                    )
                )
    if "e3" in experiments:
        # E3 reuses the published 8-shadow seed-42 checkpoints.  If those are
        # missing, these jobs train tagged copies instead of letting LiRA fail
        # halfway through the attack phase.  Existing checkpoints are skipped.
        for dataset in datasets:
            if dataset not in E3_DATASETS:
                continue
            jobs.extend(
                _shadow_jobs_for_config(
                    "e3", dataset,
                    seed=E3_SHADOW_SEED,
                    shadow_seed=E3_SHADOW_SEED,
                    num_shadow=E2_NUM_SHADOW,
                    run_tag=E3_TAG,
                )
            )
    return jobs


def build_attack_jobs(experiments: Sequence[str], datasets: Sequence[str]) -> list[RunSpec]:
    jobs: list[RunSpec] = []
    if "e1" in experiments:
        for num_shadow in E1_SHADOWS:
            for dataset in datasets:
                for method in METHODS[dataset]:
                    for seed in E1_SEEDS:
                        jobs.append(
                            RunSpec(
                                job_id=_job_id(
                                    "attack", "e1", dataset, method, seed,
                                    E1_SHADOW_SEED, num_shadow, 0.1,
                                ),
                                phase="attack",
                                experiment="e1",
                                dataset=dataset,
                                method=method,
                                seed=seed,
                                shadow_seed=E1_SHADOW_SEED,
                                num_shadow=num_shadow,
                                forget_perc=0.1,
                                run_tag=e1_tag(num_shadow),
                                attacks=tuple(ALL_ATTACKS),
                                stages=("unlearn", "reminiscence", "attack"),
                            )
                        )
    if "e2" in experiments:
        for dataset in datasets:
            for method in METHODS[dataset]:
                for seed in E2_SEEDS:
                    jobs.append(
                        RunSpec(
                            job_id=_job_id(
                                "attack", "e2", dataset, method, seed, seed,
                                E2_NUM_SHADOW, 0.1,
                            ),
                            phase="attack",
                            experiment="e2",
                            dataset=dataset,
                            method=method,
                            seed=seed,
                            shadow_seed=seed,
                            num_shadow=E2_NUM_SHADOW,
                            forget_perc=0.1,
                            run_tag=E2_TAG,
                            attacks=tuple(ALL_ATTACKS),
                            stages=("unlearn", "reminiscence", "attack"),
                        )
                    )
    if "e3" in experiments:
        selected = [d for d in datasets if d in E3_DATASETS]
        for dataset in selected:
            for forget_perc in E3_PERCS:
                for method in METHODS[dataset]:
                    for seed in E3_SEEDS:
                        jobs.append(
                            RunSpec(
                                job_id=_job_id(
                                    "attack", "e3", dataset, method, seed,
                                    E3_SHADOW_SEED, E2_NUM_SHADOW, forget_perc,
                                ),
                                phase="attack",
                                experiment="e3",
                                dataset=dataset,
                                method=method,
                                seed=seed,
                                shadow_seed=E3_SHADOW_SEED,
                                num_shadow=E2_NUM_SHADOW,
                                forget_perc=forget_perc,
                                run_tag=E3_TAG,
                                attacks=tuple(E3_ATTACKS),
                                stages=("unlearn", "reminiscence", "attack"),
                            )
                        )
    return jobs


def build_jobs(
    experiments: Sequence[str],
    datasets: Sequence[str],
    phase: str,
) -> list[RunSpec]:
    jobs: list[RunSpec] = []
    if phase in ("all", "shadow"):
        jobs.extend(build_shadow_jobs(experiments, datasets))
    if phase in ("all", "attack"):
        jobs.extend(build_attack_jobs(experiments, datasets))
    return jobs


# ── Preflight ────────────────────────────────────────────────────────────────

def preflight(jobs: Sequence[RunSpec]) -> list[str]:
    """Cheap checks that fail before hours of GPU time are wasted."""
    problems: list[str] = []
    checkpoint_root = Path(config.CHECKPOINT_PATH)
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO_ROOT / checkpoint_root
    if not checkpoint_root.exists():
        problems.append(
            f"CHECKPOINT_PATH does not exist: {checkpoint_root} "
            f"(set MIA_CHECKPOINT_PATH or restore the backup)"
        )

    attack_datasets = sorted({job.dataset for job in jobs if job.phase == "attack"})
    for dataset in attack_datasets:
        net, classes = DATASET_SPECS[dataset]
        weight_path = (
            checkpoint_root / "pretrain" / f"{net}-{dataset}-{classes}" / "best.pth"
        )
        if not weight_path.exists():
            problems.append(f"missing pretrain model: {weight_path}")

    data_root = REPO_ROOT / "data"
    if not data_root.exists():
        problems.append(
            f"dataset root not found: {data_root} "
            "(the first job would download it; parallel downloads can race)"
        )
    return problems


# ── Execution ────────────────────────────────────────────────────────────────

def _parse_gpus(value: str) -> list[str]:
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id, e.g. 0 or 0,1,2,3")
    return gpus


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a job and every child it spawned (train.py, unlearn_mia.py...).

    Killing only the direct child leaves grandchildren holding GPU memory and
    writing checkpoints, which then corrupts the retry.
    """
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    else:  # pragma: no cover - Windows convenience
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            proc.kill()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        proc.kill()


class Scheduler:
    def __init__(
        self,
        *,
        log_root: Path,
        gpus: Sequence[str],
        workers_per_gpu: int,
        timeout_seconds: float | None,
        resume: bool,
        dry_run: bool,
    ) -> None:
        self.log_root = log_root
        self.done_dir = log_root / "done"
        self.failed_dir = log_root / "failed"
        self.log_dir = log_root / "logs"
        for path in (self.done_dir, self.failed_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.gpus = list(gpus)
        self.workers_per_gpu = workers_per_gpu
        self.timeout_seconds = timeout_seconds
        self.resume = resume
        self.dry_run = dry_run
        self._gpu_queue: queue.Queue[str] = queue.Queue()
        for gpu in self.gpus:
            for _ in range(self.workers_per_gpu):
                self._gpu_queue.put(gpu)
        self._print_lock = threading.Lock()
        self._results: list[dict[str, object]] = []
        self._results_lock = threading.Lock()

    def _say(self, message: str) -> None:
        with self._print_lock:
            print(message, flush=True)

    def _marker(self, job: RunSpec, kind: str) -> Path:
        base = self.done_dir if kind == "done" else self.failed_dir
        return base / f"{job.job_id}.{kind}"

    def _already_done(self, job: RunSpec) -> bool:
        return self._marker(job, "done").exists()

    def _run_one(self, job: RunSpec, gpu: str) -> None:
        cmd = job.command()
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = self.log_dir / f"{job.job_id}.log"
        started = time.time()
        self._say(f"[start] gpu={gpu} {job.job_id}")
        popen_kwargs: dict[str, object] = {}
        if os.name == "posix":
            # Put each job in its own process group so a timeout can kill the
            # whole tree (run_benchmark.py -> train.py / unlearn_mia.py ...).
            popen_kwargs["start_new_session"] = True
        else:  # pragma: no cover - Windows convenience
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"# cwd={REPO_ROOT}\n")
                log_file.write(f"# CUDA_VISIBLE_DEVICES={gpu}\n")
                log_file.write("# " + " ".join(cmd) + "\n\n")
                log_file.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    **popen_kwargs,
                )
                try:
                    returncode = proc.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(proc)
                    returncode = proc.returncode if proc.returncode is not None else -9
                    log_file.write(f"\n[runner] TIMEOUT after {self.timeout_seconds}s\n")
                    log_file.flush()
            elapsed = time.time() - started
            ok = returncode == 0
        except Exception as exc:  # pragma: no cover - defensive
            elapsed = time.time() - started
            ok = False
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n[runner] ERROR: {exc!r}\n")

        marker = self._marker(job, "done" if ok else "failed")
        stale = self._marker(job, "failed" if ok else "done")
        marker.write_text(
            json.dumps({"job_id": job.job_id, "cmd": cmd, "elapsed_sec": elapsed}),
            encoding="utf-8",
        )
        # A resumed job may flip from failed to done (or vice versa); keep only
        # the latest outcome so `failures()` reflects the current state.
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
        with self._results_lock:
            self._results.append(
                {"job_id": job.job_id, "phase": job.phase, "ok": ok, "elapsed_sec": elapsed}
            )
        status = "done" if ok else "FAILED"
        self._say(f"[{status}] gpu={gpu} {job.job_id} ({elapsed/60:.1f} min)")

    def _worker(self, job_queue: queue.Queue[RunSpec | None]) -> None:
        while True:
            job = job_queue.get()
            if job is None:
                job_queue.task_done()
                return
            gpu = self._gpu_queue.get()
            try:
                self._run_one(job, gpu)
            finally:
                self._gpu_queue.put(gpu)
                job_queue.task_done()

    def run(self, jobs: Sequence[RunSpec], phase_label: str) -> tuple[int, int, int]:
        if not jobs:
            return 0, 0, 0
        pending: list[RunSpec] = []
        skipped = 0
        for job in jobs:
            if self.resume and self._already_done(job):
                skipped += 1
                continue
            pending.append(job)

        self._say(
            f"[phase:{phase_label}] {len(jobs)} jobs, "
            f"{skipped} already done, {len(pending)} to run "
            f"on {len(self.gpus)} GPU(s) x {self.workers_per_gpu} worker(s)"
        )
        if self.dry_run or not pending:
            return 0, 0, skipped

        job_queue: queue.Queue[RunSpec | None] = queue.Queue()
        for job in pending:
            job_queue.put(job)
        worker_count = len(self.gpus) * self.workers_per_gpu
        threads = [
            threading.Thread(target=self._worker, args=(job_queue,), daemon=True)
            for _ in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        job_queue.join()
        for _ in threads:
            job_queue.put(None)
        for thread in threads:
            thread.join()

        return len(pending), 0, skipped

    def failures(self, jobs: Sequence[RunSpec]) -> list[RunSpec]:
        return [
            job
            for job in jobs
            if self._marker(job, "failed").exists()
            and not self._marker(job, "done").exists()
        ]

    def summary(self) -> dict[str, int]:
        done = sum(1 for r in self._results if r["ok"])
        failed = len(self._results) - done
        return {"finished": done, "failed": failed}


def print_matrix(jobs: Sequence[RunSpec]) -> None:
    by_phase: dict[str, int] = {}
    by_experiment: dict[str, int] = {}
    for job in jobs:
        by_phase[job.phase] = by_phase.get(job.phase, 0) + 1
        by_experiment[job.experiment] = by_experiment.get(job.experiment, 0) + 1
    print("Experiment matrix")
    print(f"  total jobs: {len(jobs)}")
    for phase, count in sorted(by_phase.items()):
        print(f"  {phase:8s}: {count}")
    for experiment, count in sorted(by_experiment.items()):
        print(f"  {experiment:8s}: {count}")
    print()
    print("Expected attack results (before resume):")
    attack_jobs = [j for j in jobs if j.phase == "attack"]
    by_exp: dict[str, int] = {}
    for job in attack_jobs:
        by_exp[job.experiment] = by_exp.get(job.experiment, 0) + len(job.attacks)
    for experiment, count in sorted(by_exp.items()):
        print(f"  {experiment}: {count} attack results")
    if attack_jobs:
        print(f"  total: {sum(by_exp.values())} attack results")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the TIFS revision experiments (shadow scaling / seeds / forget rate)."
    )
    parser.add_argument("--experiments", nargs="+", choices=["e1", "e2", "e3"], default=["e1", "e2", "e3"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--phase", choices=["all", "shadow", "attack"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix and commands; do not execute.")
    parser.add_argument("--print-commands", action="store_true", help="Print every command (implies --dry-run only if combined).")
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU ids, e.g. 0,1,2,3.")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip jobs whose done marker exists.")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N pending jobs (calibration).")
    parser.add_argument("--job-timeout-hours", type=float, default=48.0)
    parser.add_argument("--log-root", default="runtime_logs/missing_experiments")
    parser.add_argument("--plan-name", default=None)
    parser.add_argument(
        "--continue-after-shadow-failure",
        action="store_true",
        help="Run the attack phase even if some shadow jobs failed.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip checks for pretrain checkpoints / dataset root.",
    )
    args = parser.parse_args(argv)

    experiments = list(dict.fromkeys(args.experiments))
    datasets = list(dict.fromkeys(args.datasets))
    gpus = _parse_gpus(args.gpus)

    jobs = build_jobs(experiments, datasets, args.phase)

    plan_name = args.plan_name or time.strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_root) / plan_name
    log_root.mkdir(parents=True, exist_ok=True)
    plan_path = log_root / "plan.json"
    plan_payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": experiments,
        "datasets": datasets,
        "phase": args.phase,
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "jobs": [
            {**asdict(job), "command": job.command()}
            for job in jobs
        ],
    }
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_payload, f, indent=2, ensure_ascii=False)

    print_matrix(jobs)
    print(f"Plan written to: {plan_path}")
    if args.dry_run or args.print_commands:
        for job in jobs:
            print(f"[{job.phase}:{job.experiment}] {job.job_id}")
            print("  " + " ".join(job.command()))
    if args.dry_run:
        return

    if not args.skip_preflight:
        problems = preflight(jobs)
        if problems:
            print("Preflight failed:")
            for problem in problems:
                print(f"  - {problem}")
            print("Fix these first, or pass --skip-preflight to override.")
            raise SystemExit(1)

    scheduler = Scheduler(
        log_root=log_root,
        gpus=gpus,
        workers_per_gpu=args.workers_per_gpu,
        timeout_seconds=args.job_timeout_hours * 3600 if args.job_timeout_hours > 0 else None,
        resume=args.resume,
        dry_run=False,
    )

    shadow_jobs = [j for j in jobs if j.phase == "shadow"]
    attack_jobs = [j for j in jobs if j.phase == "attack"]
    if args.limit > 0:
        # Limit across phases while keeping shadow-before-attack ordering.
        shadow_jobs = shadow_jobs[: args.limit]
        remaining = max(0, args.limit - len(shadow_jobs))
        attack_jobs = attack_jobs[:remaining]

    scheduler.run(shadow_jobs, "shadow")
    failed_shadows = scheduler.failures(shadow_jobs)
    if failed_shadows and not args.continue_after_shadow_failure:
        print(f"Aborting before attack phase: {len(failed_shadows)} shadow job(s) failed.")
        for job in failed_shadows:
            print(f"  - {job.job_id}")
    else:
        scheduler.run(attack_jobs, "attack")

    summary = scheduler.summary()
    summary.update(
        {
            "plan": str(plan_path),
            "requested_shadow_jobs": len(shadow_jobs),
            "requested_attack_jobs": len(attack_jobs),
            "failed_shadow_jobs": [j.job_id for j in failed_shadows],
        }
    )
    summary_path = log_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
