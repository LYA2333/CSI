"""Compare untrained and trained CSI accuracy on a local MATH-500 dataset.

The three frozen Qwen models are loaded once. Evaluation first uses freshly
initialized CSI modules, then loads the trained CSI checkpoint into the same
system and evaluates the identical examples again. Model loading and checkpoint
loading are excluded from per-example latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import torch

from csi_qwen import CSIQwenSystem, GenerationResult, build_qwen_system


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("test", "data", "records"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain a list of JSON records")
    return value


def _read_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Parquet input requires pandas and pyarrow") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON record")
                rows.append(value)
        return rows
    if suffix == ".json":
        return _read_json(path)
    raise ValueError(f"Unsupported dataset file: {path}")


def load_records(path: Path, split: str) -> list[dict[str, Any]]:
    """Load a Parquet/JSON file, HF save_to_disk directory, or data directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        return _read_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    if (path / "dataset_dict.json").is_file() or (path / "state.json").is_file():
        try:
            from datasets import DatasetDict, load_from_disk
        except ImportError as exc:
            raise ImportError("A save_to_disk dataset requires the datasets package") from exc
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(
                    f"Split {split!r} is unavailable; found {list(dataset.keys())}"
                )
            dataset = dataset[split]
        return [dict(row) for row in dataset]

    parquet_files = sorted(path.rglob(f"*{split}*.parquet"))
    if not parquet_files:
        parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet files found under {path}")
    rows: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        rows.extend(_read_file(parquet_file))
    return rows


def _balanced_command_argument(text: str, command_start: int) -> tuple[str, int] | None:
    brace_start = text.find("{", command_start)
    if brace_start < 0:
        return None
    depth = 0
    for index in range(brace_start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index], index + 1
    return None


def extract_last_boxed(text: str) -> str:
    candidates: list[tuple[int, str]] = []
    for command in (r"\boxed", r"\fbox"):
        start = 0
        while True:
            index = text.find(command, start)
            if index < 0:
                break
            argument = _balanced_command_argument(text, index + len(command))
            if argument is None:
                break
            content, end = argument
            candidates.append((index, content.strip()))
            start = end
    return max(candidates, key=lambda item: item[0])[1] if candidates else ""


def extract_final_answer(text: str) -> str:
    text = _text(text)
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed

    matches = list(
        re.finditer(
            r"(?is)(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)",
            text,
        )
    )
    if matches:
        candidate = matches[-1].group(1).strip()
        return candidate.splitlines()[0].strip().strip("$ ")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].strip("$ ") if lines else ""


def normalize_answer(text: str) -> str:
    value = extract_final_answer(text)
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\!", "").replace(r"\,", "")
    value = value.replace(r"\;", "").replace(r"\:", "")
    value = value.replace(r"\displaystyle", "")
    value = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", value)
    value = value.replace("$", "").replace(" ", "")
    value = value.rstrip(".,;:")
    while len(value) >= 2 and value[0] == "{" and value[-1] == "}":
        depth = 0
        closes_at_end = False
        for index, character in enumerate(value):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    closes_at_end = index == len(value) - 1
                    break
        if not closes_at_end:
            break
        value = value[1:-1]
    return value.lower()


def _replace_latex_binary_command(text: str, command: str, operator: str) -> str:
    start = 0
    while True:
        index = text.find(command, start)
        if index < 0:
            return text
        first = _balanced_command_argument(text, index + len(command))
        if first is None:
            return text
        numerator, first_end = first
        second = _balanced_command_argument(text, first_end)
        if second is None:
            return text
        denominator, second_end = second
        replacement = f"(({numerator}){operator}({denominator}))"
        text = text[:index] + replacement + text[second_end:]
        start = 0


def _replace_latex_sqrt(text: str) -> str:
    start = 0
    command = r"\sqrt"
    while True:
        index = text.find(command, start)
        if index < 0:
            return text
        argument = _balanced_command_argument(text, index + len(command))
        if argument is None:
            return text
        content, end = argument
        replacement = f"sqrt({content})"
        text = text[:index] + replacement + text[end:]
        start = 0


def _simple_sympy_expression(text: str) -> Any | None:
    value = normalize_answer(text)
    if not value or any(token in value for token in ("\\begin", "\\end", "\\cup")):
        return None
    value = _replace_latex_binary_command(value, r"\frac", "/")
    value = _replace_latex_sqrt(value)
    value = value.replace(r"\cdot", "*").replace(r"\times", "*")
    value = value.replace(r"\pi", "pi").replace(r"\infty", "oo")
    value = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", value)
    value = value.replace("^", "**")
    value = value.replace("{", "(").replace("}", ")")
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    if re.search(r"\\[a-zA-Z]+", value):
        return None
    try:
        import sympy

        return sympy.sympify(value, locals={"pi": sympy.pi, "oo": sympy.oo})
    except Exception:
        return None


def normalized_equivalent(reference: str, response: str) -> bool:
    reference_normalized = normalize_answer(reference)
    response_normalized = normalize_answer(response)
    if not reference_normalized or not response_normalized:
        return False
    if reference_normalized == response_normalized:
        return True

    reference_expression = _simple_sympy_expression(reference)
    response_expression = _simple_sympy_expression(response)
    if reference_expression is None or response_expression is None:
        return False
    try:
        import sympy

        difference = sympy.simplify(reference_expression - response_expression)
        if difference == 0:
            return True
        numeric = complex(sympy.N(difference))
        return math.isfinite(numeric.real) and math.isfinite(numeric.imag) and abs(numeric) < 1e-8
    except Exception:
        return False


class MathGrader:
    def __init__(self, mode: str) -> None:
        self.requested_mode = mode
        self._parse = None
        self._verify = None
        if mode in {"auto", "math_verify"}:
            try:
                from math_verify import parse, verify

                self._parse = parse
                self._verify = verify
            except ImportError:
                if mode == "math_verify":
                    raise ImportError(
                        "--grader math_verify requires: pip install math-verify"
                    )
        self.name = (
            "math_verify_with_normalized_fallback"
            if self._parse is not None
            else "normalized_fallback"
        )
        if mode == "auto" and self._parse is None:
            print(
                "WARNING: math-verify is not installed; accuracy uses the less "
                "complete normalized fallback. Install it with: pip install math-verify",
                flush=True,
            )

    def _math_verify_equivalent(self, reference: str, response: str) -> bool | None:
        if self._parse is None or self._verify is None:
            return None
        reference_answer = extract_final_answer(reference)
        response_answer = extract_final_answer(response)
        try:
            gold = self._parse(f"$\\boxed{{{reference_answer}}}$")
            prediction = self._parse(response)
            if not prediction:
                prediction = self._parse(f"$\\boxed{{{response_answer}}}$")
            if not gold or not prediction:
                return None
            return bool(self._verify(gold, prediction))
        except Exception:
            return None

    def equivalent(self, reference: str, response: str) -> bool:
        verified = self._math_verify_equivalent(reference, response)
        if verified is not None:
            return verified
        return normalized_equivalent(reference, response)


def normalize_math500_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected = list(rows[offset : None if limit <= 0 else offset + limit])
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=offset):
        problem = _text(row.get("problem") or row.get("question") or row.get("prompt"))
        solution = _text(row.get("solution"))
        answer = _text(row.get("answer")) or extract_final_answer(solution)
        if not problem or not answer:
            raise ValueError(
                f"Dataset row {index} needs problem/question and answer/solution"
            )
        task_id = _text(
            row.get("unique_id") or row.get("id") or row.get("task_id")
        ) or f"math500_{index}"
        normalized.append(
            {
                "task_id": task_id,
                "prompt": problem,
                "reference_answer": answer,
                "reference_solution": solution,
                "subject": _text(row.get("subject")),
                "level": _text(row.get("level")),
                "task_type": "math",
            }
        )
    if not normalized:
        raise ValueError("No examples selected for evaluation")
    return normalized


def _seed_for_task(seed: int, task_id: str) -> None:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    task_seed = seed + int.from_bytes(digest[:4], "big")
    torch.manual_seed(task_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(task_seed)


def _synchronize(system: CSIQwenSystem) -> None:
    if not torch.cuda.is_available():
        return
    devices = {agent.device for agent in system.agents}
    devices.add(system.csi_device)
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def evaluate_stage(
    system: CSIQwenSystem,
    examples: Sequence[Mapping[str, Any]],
    grader: MathGrader,
    *,
    stage: str,
    output_path: Path,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_on_boxed: bool,
    stop_on_repetition: bool,
    repetition_min_tokens: int,
    repetition_max_pattern_tokens: int,
    profile_latency: bool,
    seed: int,
    log_every: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    correct_count = 0
    latency_sum = 0.0
    generated_token_sum = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            task_id = str(example["task_id"])
            _seed_for_task(seed, task_id)
            task = {
                "task_id": task_id,
                "prompt": example["prompt"],
                "task_type": "math",
                "subject": example.get("subject", ""),
                "level": example.get("level", ""),
            }
            response = ""
            error = ""
            correct = False
            generation_result: GenerationResult | None = None
            _synchronize(system)
            started = time.perf_counter()
            try:
                generated = system.generate_code(
                    str(example["prompt"]),
                    task,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop_on_boxed=stop_on_boxed,
                    stop_on_repetition=stop_on_repetition,
                    repetition_min_tokens=repetition_min_tokens,
                    repetition_max_pattern_tokens=repetition_max_pattern_tokens,
                    return_result=True,
                    profile_latency=profile_latency,
                )
                if not isinstance(generated, GenerationResult):
                    raise TypeError("Expected GenerationResult from CSI generation")
                generation_result = generated
                response = generated.text
                _synchronize(system)
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:
                _synchronize(system)
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - started
            if not error:
                try:
                    correct = grader.equivalent(
                        str(example["reference_answer"]), response
                    )
                except Exception as exc:
                    error = f"grader {type(exc).__name__}: {exc}"
            correct_count += int(correct)
            latency_sum += latency
            generated_tokens = (
                generation_result.generated_tokens
                if generation_result is not None
                else 0
            )
            generated_token_sum += generated_tokens
            record = {
                "stage": stage,
                "index": index - 1,
                "task_id": task_id,
                "subject": example.get("subject", ""),
                "level": example.get("level", ""),
                "prompt": example["prompt"],
                "reference_answer": example["reference_answer"],
                "response": response,
                "extracted_answer": extract_final_answer(response),
                "correct": correct,
                "latency_seconds": latency,
                "generated_tokens": generated_tokens,
                "solver_prompt_tokens": (
                    generation_result.solver_prompt_tokens
                    if generation_result is not None
                    else 0
                ),
                "csi_prefix_tokens": (
                    generation_result.csi_prefix_tokens
                    if generation_result is not None
                    else 0
                ),
                "finish_reason": (
                    generation_result.finish_reason
                    if generation_result is not None
                    else "error"
                ),
                "timing_seconds": (
                    generation_result.timing_seconds
                    if generation_result is not None
                    else {}
                ),
                "error": error,
            }
            results.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            if index % log_every == 0 or index == len(examples):
                print(
                    f"[{stage}] {index}/{len(examples)} "
                    f"accuracy={correct_count / index:.4f} "
                    f"avg_latency={latency_sum / index:.3f}s "
                    f"avg_tokens={generated_token_sum / index:.1f} "
                    f"tokens/s={generated_token_sum / latency_sum:.2f}",
                    flush=True,
                )
    return results


def summarize_stage_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ValueError("Cannot summarize an empty evaluation stage")

    latencies = [float(row["latency_seconds"]) for row in rows]
    sorted_latencies = sorted(latencies)
    midpoint = count // 2
    median_latency = (
        sorted_latencies[midpoint]
        if count % 2 == 1
        else (sorted_latencies[midpoint - 1] + sorted_latencies[midpoint]) / 2.0
    )
    total_latency = sum(latencies)
    correct = sum(int(bool(row["correct"])) for row in rows)
    total_generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in rows)
    finish_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("finish_reason", "unknown"))
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1

    summary: dict[str, Any] = {
        "correct": correct,
        "accuracy": correct / count,
        "mean_latency_seconds": total_latency / count,
        "median_latency_seconds": median_latency,
        "p95_latency_seconds": sorted_latencies[max(0, math.ceil(0.95 * count) - 1)],
        "total_latency_seconds": total_latency,
        "total_generated_tokens": total_generated_tokens,
        "mean_generated_tokens": total_generated_tokens / count,
        "generated_tokens_per_second": (
            total_generated_tokens / total_latency if total_latency > 0 else 0.0
        ),
        "mean_solver_prompt_tokens": sum(
            int(row.get("solver_prompt_tokens", 0)) for row in rows
        ) / count,
        "mean_csi_prefix_tokens": sum(
            int(row.get("csi_prefix_tokens", 0)) for row in rows
        ) / count,
        "finish_reasons": finish_reasons,
        "errors": sum(bool(row["error"]) for row in rows),
    }

    profiled = [row for row in rows if row.get("timing_seconds")]
    if profiled:
        timing_names = sorted(
            {
                str(name)
                for row in profiled
                for name in row["timing_seconds"]
            }
        )
        summary["profiled_examples"] = len(profiled)
        summary["mean_timing_seconds"] = {
            name: sum(float(row["timing_seconds"].get(name, 0.0)) for row in profiled)
            / len(profiled)
            for name in timing_names
        }
    return summary


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint must be a train_csi.py checkpoint payload")
    return payload


def _architecture_value(
    name: str,
    cli_value: Any,
    model_config: Mapping[str, Any],
    fallback: Any,
) -> Any:
    return cli_value if cli_value is not None else model_config.get(name, fallback)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fresh and trained CSI accuracy on local MATH-500"
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("math500_csi_eval"))
    parser.add_argument(
        "--eval-stage",
        choices=("both", "before", "after"),
        default="both",
        help="Evaluate both states or only one state (default: both)",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all rows")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--warmup-samples", type=int, default=1)

    parser.add_argument("--planner-model-path", required=True)
    parser.add_argument("--critic-model-path", required=True)
    parser.add_argument("--executor-model-path", required=True)
    parser.add_argument("--planner-device", default="cuda:4")
    parser.add_argument("--critic-device", default="cuda:5")
    parser.add_argument("--executor-device", default="cuda:6")
    parser.add_argument("--csi-device", default="cuda:7")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--csi-dtype", default="float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="")

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--latent-tokens", type=int, default=None)
    parser.add_argument("--prefix-tokens", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--dilations", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--max-feature-tokens", type=int, default=512)

    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--profile-latency",
        action="store_true",
        help=(
            "Record collaboration/prefix/prefill/decode timings. This inserts "
            "CUDA synchronization barriers, so use a short diagnostic run rather "
            "than this mode for the headline latency number."
        ),
    )
    parser.add_argument(
        "--stop-on-boxed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after the first complete, non-empty \\boxed{...} (default: on)",
    )
    parser.add_argument(
        "--stop-on-repetition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when a short token pattern repeats for a long suffix (default: on)",
    )
    parser.add_argument(
        "--repetition-min-tokens",
        type=int,
        default=64,
        help="Minimum periodic suffix length before repetition stopping",
    )
    parser.add_argument(
        "--repetition-max-pattern-tokens",
        type=int,
        default=16,
        help="Largest repeated token pattern checked by repetition stopping",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grader",
        choices=("auto", "math_verify", "normalized"),
        default="auto",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.offset < 0 or args.limit < 0:
        raise ValueError("offset and limit must be non-negative")
    if args.log_every < 1 or args.max_new_tokens < 1:
        raise ValueError("log_every and max_new_tokens must be positive")
    if args.warmup_samples < 0:
        raise ValueError("warmup_samples must be non-negative")
    if args.repetition_min_tokens < 1:
        raise ValueError("repetition_min_tokens must be positive")
    if args.repetition_max_pattern_tokens < 1:
        raise ValueError("repetition_max_pattern_tokens must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")


def _warm_up(
    system: CSIQwenSystem,
    examples: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    stage: str,
) -> None:
    count = min(args.warmup_samples, len(examples))
    if count == 0:
        return
    print(f"[{stage}] warming up with {count} unscored sample(s)...", flush=True)
    for example in examples[:count]:
        task_id = str(example["task_id"])
        _seed_for_task(args.seed, task_id)
        system.generate_code(
            str(example["prompt"]),
            {
                "task_id": task_id,
                "prompt": example["prompt"],
                "task_type": "math",
            },
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop_on_boxed=args.stop_on_boxed,
            stop_on_repetition=args.stop_on_repetition,
            repetition_min_tokens=args.repetition_min_tokens,
            repetition_max_pattern_tokens=args.repetition_max_pattern_tokens,
        )
    _synchronize(system)


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    rows = load_records(args.dataset_path, args.dataset_split)
    examples = normalize_math500_rows(rows, offset=args.offset, limit=args.limit)
    grader = MathGrader(args.grader)
    checkpoint_payload = _checkpoint_payload(checkpoint_path)
    model_config = checkpoint_payload.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Checkpoint model_config is missing or invalid")

    d_model = int(_architecture_value("d_model", args.d_model, model_config, 512))
    latent_tokens = int(
        _architecture_value("latent_tokens", args.latent_tokens, model_config, 32)
    )
    prefix_tokens = int(
        _architecture_value("prefix_tokens", args.prefix_tokens, model_config, 8)
    )
    kernel_size = int(
        _architecture_value("kernel_size", args.kernel_size, model_config, 5)
    )
    dilations = list(
        _architecture_value("dilations", args.dilations, model_config, [1])
    )
    max_steps = int(
        _architecture_value("max_steps", args.max_steps, model_config, 3)
    )
    tolerance = float(
        _architecture_value("tolerance", args.tolerance, model_config, 1e-3)
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Loaded {len(examples)} MATH-500 example(s); grader={grader.name}")
    print(
        "CSI architecture: "
        f"d_model={d_model}, latent_tokens={latent_tokens}, "
        f"prefix_tokens={prefix_tokens}, kernel_size={kernel_size}, "
        f"dilations={dilations}, max_steps={max_steps}",
        flush=True,
    )
    print("Loading frozen Qwen models once...", flush=True)
    system = build_qwen_system(
        planner_model_path=args.planner_model_path,
        critic_model_path=args.critic_model_path,
        executor_model_path=args.executor_model_path,
        planner_device=args.planner_device,
        critic_device=args.critic_device,
        executor_device=args.executor_device,
        csi_device=args.csi_device,
        model_dtype=args.model_dtype,
        csi_dtype=args.csi_dtype,
        d_model=d_model,
        latent_tokens=latent_tokens,
        prefix_tokens=prefix_tokens,
        kernel_size=kernel_size,
        dilations=dilations,
        max_steps=max_steps,
        tolerance=tolerance,
        max_feature_tokens=args.max_feature_tokens,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    # Validate checkpoint compatibility before spending hours on the untrained
    # pass. Only small CSI tensors are copied; frozen LLM weights are not part
    # of CSIQwenSystem.state_dict(). Keeping checkpoint tensors on CPU also
    # avoids moving saved optimizer states onto the CSI GPU during evaluation.
    try:
        current_config = system.checkpoint_config()
        for key in ("intra_conv_axis", "prefix_position", "inner_adapter_type"):
            if model_config.get(key) != current_config.get(key):
                raise RuntimeError(
                    f"Checkpoint {key}={model_config.get(key)!r} is incompatible "
                    f"with evaluator {key}={current_config.get(key)!r}"
                )
        checkpoint_state = checkpoint_payload["model_state_dict"]
        if not isinstance(checkpoint_state, Mapping):
            raise ValueError("Checkpoint model_state_dict is invalid")
        trained_state = {
            key: value.detach().cpu().clone()
            for key, value in checkpoint_state.items()
        }
        fresh_state = None
        if args.eval_stage in {"both", "before"}:
            fresh_state = {
                key: value.detach().cpu().clone()
                for key, value in system.state_dict().items()
            }
        system.load_state_dict(trained_state, strict=True)
        if fresh_state is not None:
            system.load_state_dict(fresh_state, strict=True)
        del checkpoint_payload, checkpoint_state
    except Exception:
        system.close()
        raise

    args.output_dir.mkdir(parents=True, exist_ok=True)
    before: list[dict[str, Any]] | None = None
    after: list[dict[str, Any]] | None = None
    try:
        if args.eval_stage in {"both", "before"}:
            print(
                "[before] evaluating freshly initialized CSI modules with the same "
                "frozen Qwen models...",
                flush=True,
            )
            _warm_up(system, examples, args, "before")
            before = evaluate_stage(
                system,
                examples,
                grader,
                stage="before",
                output_path=args.output_dir / "before.jsonl",
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stop_on_boxed=args.stop_on_boxed,
                stop_on_repetition=args.stop_on_repetition,
                repetition_min_tokens=args.repetition_min_tokens,
                repetition_max_pattern_tokens=args.repetition_max_pattern_tokens,
                profile_latency=args.profile_latency,
                seed=args.seed,
                log_every=args.log_every,
            )

        if args.eval_stage in {"both", "after"}:
            print(
                f"[after] activating trained CSI checkpoint: {checkpoint_path}",
                flush=True,
            )
            system.load_state_dict(trained_state, strict=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _warm_up(system, examples, args, "after")
            after = evaluate_stage(
                system,
                examples,
                grader,
                stage="after",
                output_path=args.output_dir / "after.jsonl",
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stop_on_boxed=args.stop_on_boxed,
                stop_on_repetition=args.stop_on_repetition,
                repetition_min_tokens=args.repetition_min_tokens,
                repetition_max_pattern_tokens=args.repetition_max_pattern_tokens,
                profile_latency=args.profile_latency,
                seed=args.seed,
                log_every=args.log_every,
            )
    finally:
        system.close()

    count = len(examples)
    summary = {
        "dataset_path": str(args.dataset_path.expanduser().resolve()),
        "checkpoint_path": str(checkpoint_path),
        "grader": grader.name,
        "eval_stage": args.eval_stage,
        "num_examples": count,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stop_on_boxed": args.stop_on_boxed,
            "stop_on_repetition": args.stop_on_repetition,
            "repetition_min_tokens": args.repetition_min_tokens,
            "repetition_max_pattern_tokens": args.repetition_max_pattern_tokens,
            "profile_latency": args.profile_latency,
            "token_metric": "executor_generated_tokens_excluding_eos",
            "seed": args.seed,
        },
        "architecture": {
            "d_model": d_model,
            "latent_tokens": latent_tokens,
            "prefix_tokens": prefix_tokens,
            "kernel_size": kernel_size,
            "dilations": dilations,
            "max_steps": max_steps,
            "tolerance": tolerance,
        },
    }
    if before is not None:
        summary["before"] = summarize_stage_results(before)
        before_correct = int(summary["before"]["correct"])
    if after is not None:
        summary["after"] = summarize_stage_results(after)
        after_correct = int(summary["after"]["correct"])
    if before is not None and after is not None:
        improved = sum(
            int(not before_row["correct"] and after_row["correct"])
            for before_row, after_row in zip(before, after)
        )
        regressed = sum(
            int(before_row["correct"] and not after_row["correct"])
            for before_row, after_row in zip(before, after)
        )
        summary["accuracy_delta"] = (after_correct - before_correct) / count
        summary["paired_changes"] = {
            "improved": improved,
            "regressed": regressed,
            "unchanged": count - improved - regressed,
        }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
