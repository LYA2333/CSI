"""Compare RecursiveMAS and CSI-GCN on code-generation tasks.

Each runner is loaded exactly once. For every task we time only ``generate``;
candidate execution happens afterwards and is reported separately. The primary
quality metric is pass@1: a generated program passes every test for the task.

Runner contract
---------------
The factory selected by ``file.py:factory`` must return an object exposing:

    generate(prompt: str, task_id: str, task: Mapping[str, Any]) -> str

The returned string may be raw Python or a Markdown Python code block.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Mapping, Protocol, Sequence


class CodegenRunner(Protocol):
    def generate(
        self,
        prompt: str,
        task_id: str,
        task: Mapping[str, Any],
    ) -> str: ...


@dataclass
class GenerationResult:
    task_id: str
    method: str
    completed: bool
    inference_seconds: float
    raw_output: str
    code: str
    generation_error: str = ""


@dataclass
class ExecutionResult:
    passed: bool
    status: str
    execution_seconds: float
    stderr: str = ""


@dataclass
class EvaluatedResult:
    generation: GenerationResult
    execution: ExecutionResult


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"No tasks found in {path}")

    seen: set[str] = set()
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        tests = row.get("tests")
        if not task_id or not prompt:
            raise ValueError(f"Task {index} needs non-empty task_id and prompt")
        if task_id in seen:
            raise ValueError(f"Duplicate task_id: {task_id}")
        if not isinstance(tests, list) or not tests or not all(
            isinstance(test, str) and test.strip() for test in tests
        ):
            raise ValueError(f"Task {task_id} needs a non-empty string list 'tests'")
        seen.add(task_id)
    return rows


def load_module(spec: str) -> tuple[ModuleType, str]:
    if ":" not in spec:
        raise ValueError("Runner must look like module:factory or /path/file.py:factory")
    module_name, factory_name = spec.rsplit(":", 1)
    if module_name.endswith(".py") or "/" in module_name:
        path = Path(module_name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Runner file not found: {path}")
        unique_name = f"codegen_runner_{abs(hash(path))}"
        module_spec = importlib.util.spec_from_file_location(unique_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot import runner file: {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module, factory_name
    return importlib.import_module(module_name), factory_name


def build_runner(spec: str, kwargs_json: str) -> CodegenRunner:
    module, factory_name = load_module(spec)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise AttributeError(f"Runner factory is not callable: {spec}")
    kwargs = json.loads(kwargs_json)
    if not isinstance(kwargs, dict):
        raise ValueError("Runner kwargs must decode to a JSON object")
    runner = factory(**kwargs)
    if not callable(getattr(runner, "generate", None)):
        raise TypeError(f"Factory {spec} did not return an object with generate()")
    return runner


def synchronize_accelerator() -> None:
    """Prevent asynchronous GPU work from escaping the timed region."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except (ImportError, RuntimeError):
        pass


def extract_python_code(text: str) -> str:
    python_blocks = re.findall(r"```(?:python|py)\s*(.*?)```", text, re.DOTALL | re.I)
    if python_blocks:
        return "\n\n".join(block.strip() for block in python_blocks)
    generic_blocks = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if generic_blocks:
        return "\n\n".join(block.strip() for block in generic_blocks)
    return text.strip()


def generate_one(
    runner: CodegenRunner,
    method: str,
    task: Mapping[str, Any],
) -> GenerationResult:
    task_id = str(task["task_id"])
    prompt = str(task["prompt"])
    synchronize_accelerator()
    started = time.perf_counter()
    raw_output = ""
    error = ""
    try:
        raw_output = runner.generate(prompt, task_id, task)
        if not isinstance(raw_output, str):
            raise TypeError(f"generate() returned {type(raw_output).__name__}, not str")
    except Exception as exc:  # A failed request counts against completion and pass@1.
        error = f"{type(exc).__name__}: {exc}"
    synchronize_accelerator()
    elapsed = time.perf_counter() - started
    code = extract_python_code(raw_output)
    return GenerationResult(
        task_id=task_id,
        method=method,
        completed=not error and bool(code),
        inference_seconds=elapsed,
        raw_output=raw_output,
        code=code,
        generation_error=error,
    )


def build_test_script(task: Mapping[str, Any], code: str) -> str:
    setup = str(task.get("setup", "")).strip()
    tests = "\n".join(str(test) for test in task["tests"])
    return (
        "# Generated benchmark program\n"
        + (setup + "\n\n" if setup else "")
        + code
        + "\n\n# Tests\n"
        + tests
        + "\n"
    )


def local_resource_limiter(timeout_seconds: float, memory_mb: int):
    """Best-effort resource limits; this is not a security sandbox."""
    def limit() -> None:
        try:
            import resource

            cpu_seconds = max(1, int(math.ceil(timeout_seconds)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
            if sys.platform.startswith("linux") and hasattr(resource, "RLIMIT_AS"):
                memory_bytes = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ImportError, OSError, ValueError):
            pass

    return limit


def execute_local(
    script_path: Path,
    timeout_seconds: float,
    memory_mb: int,
) -> ExecutionResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(script_path)],
            cwd=script_path.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            preexec_fn=local_resource_limiter(timeout_seconds, memory_mb),
            check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode == 0:
            return ExecutionResult(True, "passed", elapsed)
        stderr = completed.stderr[-4000:]
        status = "syntax_or_runtime_error"
        return ExecutionResult(False, status, elapsed, stderr)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return ExecutionResult(False, "timeout", elapsed, stderr[-4000:])


def execute_docker(
    script_path: Path,
    timeout_seconds: float,
    memory_mb: int,
    docker_image: str,
) -> ExecutionResult:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker executable not found; install Docker or choose local executor")
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--cpus",
        "1",
        "--memory",
        f"{memory_mb}m",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        f"type=bind,src={script_path},dst=/workspace/candidate.py,readonly",
        docker_image,
        "python",
        "-I",
        "-B",
        "/workspace/candidate.py",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds + 2.0,
            check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode == 0:
            return ExecutionResult(True, "passed", elapsed)
        return ExecutionResult(
            False,
            "syntax_or_runtime_error",
            elapsed,
            completed.stderr[-4000:],
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return ExecutionResult(False, "timeout", elapsed, stderr[-4000:])


def evaluate_generation(
    generation: GenerationResult,
    task: Mapping[str, Any],
    *,
    executor: str,
    timeout_seconds: float,
    memory_mb: int,
    docker_image: str,
) -> ExecutionResult:
    if not generation.completed:
        return ExecutionResult(False, "generation_failed", 0.0, generation.generation_error)
    script = build_test_script(task, generation.code)
    with tempfile.TemporaryDirectory(prefix="codegen_eval_") as temp_dir:
        script_path = Path(temp_dir) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        if executor == "docker":
            return execute_docker(script_path, timeout_seconds, memory_mb, docker_image)
        return execute_local(script_path, timeout_seconds, memory_mb)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(results: Sequence[EvaluatedResult]) -> dict[str, Any]:
    total = len(results)
    completed = [result for result in results if result.generation.completed]
    passed = [result for result in results if result.execution.passed]
    latencies_ms = [result.generation.inference_seconds * 1000.0 for result in completed]
    return {
        "tasks": total,
        "generation_completion_rate_percent": 100.0 * len(completed) / total,
        "pass_at_1_percent": 100.0 * len(passed) / total,
        "passed": len(passed),
        "inference_ms_mean": statistics.fmean(latencies_ms) if latencies_ms else None,
        "inference_ms_p50": percentile(latencies_ms, 0.50) if latencies_ms else None,
        "inference_ms_p95": percentile(latencies_ms, 0.95) if latencies_ms else None,
        "inference_ms_total": sum(latencies_ms),
        "test_time_seconds_total": sum(r.execution.execution_seconds for r in results),
        "timeouts": sum(r.execution.status == "timeout" for r in results),
    }


def display_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def print_report(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    print("\nCode-generation comparison (batch size = 1)")
    print(
        f"{'method':<18} {'complete%':>10} {'pass@1%':>10} "
        f"{'mean ms':>10} {'p50 ms':>10} {'p95 ms':>10}"
    )
    print("-" * 72)
    for method, summary in summaries.items():
        print(
            f"{method:<18} "
            f"{summary['generation_completion_rate_percent']:>10.2f} "
            f"{summary['pass_at_1_percent']:>10.2f} "
            f"{display_number(summary['inference_ms_mean']):>10} "
            f"{display_number(summary['inference_ms_p50']):>10} "
            f"{display_number(summary['inference_ms_p95']):>10}"
        )


def close_runner(runner: CodegenRunner) -> None:
    close = getattr(runner, "close", None)
    if callable(close):
        close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RecursiveMAS and CSI-GCN code generation"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--recursive-runner", required=True)
    parser.add_argument("--ours-runner", required=True)
    parser.add_argument("--recursive-kwargs", default="{}")
    parser.add_argument("--ours-kwargs", default="{}")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--executor", choices=["docker", "local"], default="docker")
    parser.add_argument("--docker-image", default="python:3.10-slim")
    parser.add_argument("--allow-unsafe-local-execution", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("codegen_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.executor == "local" and not args.allow_unsafe_local_execution:
        raise SystemExit(
            "Local execution is not a security sandbox. Re-run with Docker, or explicitly "
            "pass --allow-unsafe-local-execution for trusted generated code."
        )
    if args.timeout <= 0 or args.memory_mb <= 0 or args.warmup < 0:
        raise ValueError("timeout/memory must be positive and warmup must be non-negative")

    tasks = load_jsonl(args.tasks.resolve())
    if args.limit > 0:
        tasks = tasks[: args.limit]
    random.seed(args.seed)

    print("Loading RecursiveMAS runner (not timed)...")
    recursive_runner = build_runner(args.recursive_runner, args.recursive_kwargs)
    print("Loading CSI-GCN runner (not timed)...")
    ours_runner = build_runner(args.ours_runner, args.ours_kwargs)
    runners = {"RecursiveMAS": recursive_runner, "CSI-GCN": ours_runner}

    try:
        for method, runner in runners.items():
            for index in range(min(args.warmup, len(tasks))):
                print(f"Warm-up {method} {index + 1}/{args.warmup}...", flush=True)
                generate_one(runner, method, tasks[index])

        evaluated: dict[str, list[EvaluatedResult]] = {
            method: [] for method in runners
        }
        rng = random.Random(args.seed)
        for index, task in enumerate(tasks, start=1):
            method_order = list(runners)
            rng.shuffle(method_order)
            print(f"Task {index}/{len(tasks)}: {task['task_id']}", flush=True)
            generated: dict[str, GenerationResult] = {}
            for method in method_order:
                generated[method] = generate_one(runners[method], method, task)
            for method in runners:
                execution = evaluate_generation(
                    generated[method],
                    task,
                    executor=args.executor,
                    timeout_seconds=args.timeout,
                    memory_mb=args.memory_mb,
                    docker_image=args.docker_image,
                )
                evaluated[method].append(EvaluatedResult(generated[method], execution))

        summaries = {method: summarize(rows) for method, rows in evaluated.items()}
        baseline = summaries["RecursiveMAS"]
        ours = summaries["CSI-GCN"]
        speedup = (
            baseline["inference_ms_mean"] / ours["inference_ms_mean"]
            if baseline["inference_ms_mean"] is not None
            and ours["inference_ms_mean"] not in {None, 0}
            else None
        )
        pass_delta = ours["pass_at_1_percent"] - baseline["pass_at_1_percent"]

        paired = {"both_pass": 0, "recursive_only": 0, "ours_only": 0, "both_fail": 0}
        for recursive_row, ours_row in zip(
            evaluated["RecursiveMAS"], evaluated["CSI-GCN"]
        ):
            recursive_pass = recursive_row.execution.passed
            ours_pass = ours_row.execution.passed
            if recursive_pass and ours_pass:
                paired["both_pass"] += 1
            elif recursive_pass:
                paired["recursive_only"] += 1
            elif ours_pass:
                paired["ours_only"] += 1
            else:
                paired["both_fail"] += 1

        report = {
            "settings": {
                "tasks": str(args.tasks.resolve()),
                "seed": args.seed,
                "warmup": args.warmup,
                "executor": args.executor,
                "timeout_seconds": args.timeout,
                "memory_mb": args.memory_mb,
            },
            "summaries": summaries,
            "comparison": {
                "inference_speedup_vs_recursive_mas": speedup,
                "pass_at_1_delta_percentage_points": pass_delta,
                "paired_outcomes": paired,
            },
            "results": {
                method: [
                    {
                        "generation": asdict(row.generation),
                        "execution": asdict(row.execution),
                    }
                    for row in rows
                ]
                for method, rows in evaluated.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print_report(summaries)
        print(f"\nInference speedup vs RecursiveMAS: {display_number(speedup)}x")
        print(f"Pass@1 delta: {pass_delta:+.2f} percentage points")
        print(f"Report: {args.output.resolve()}")
        return 0
    finally:
        close_runner(recursive_runner)
        close_runner(ours_runner)


if __name__ == "__main__":
    raise SystemExit(main())
