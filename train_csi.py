"""Train CSI adapters over three frozen Qwen-family agents.

Model and dataset paths intentionally default to empty strings. Fill them in
below or pass them on the command line on the machine that stores the models.

Training data can be JSONL or Parquet. The RecursiveMAS Sequential-Math
Parquet schema is normalized as ``question -> prompt`` and
``answer -> target_text``; ``plan`` and ``refined_plan`` supervise the
role-specific Inner Adapter warm-up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from csi_qwen import AGENT_ROLES, CSIQwenSystem, build_qwen_system


INNER_ROLE_LABELS = {
    "planner": "Planner",
    "critic": "Refiner",
    "executor": "Solver",
}
TRAINING_SCHEDULE = "separate_role_inner_then_outer_v2"


class CodeTrainingDataset(Dataset[dict[str, Any]]):
    """Normalize code JSONL or Sequential-Math Parquet into CSI records."""
    def __init__(self, path: Path) -> None:
        self.rows: list[dict[str, Any]] = []
        for row_number, raw_row in enumerate(self._read_records(path), start=1):
            if not isinstance(raw_row, dict):
                raise ValueError(f"{path}: row {row_number} must be a record")
            self.rows.append(self._normalize_record(path, row_number, raw_row))
        if not self.rows:
            raise ValueError(f"No training samples found in {path}")

    @staticmethod
    def _read_records(path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError(
                    "Parquet training requires pandas and a Parquet engine; "
                    "install them with: pip install pandas pyarrow"
                ) from exc
            frame = pd.read_parquet(path)
            return frame.to_dict(orient="records")

        if suffix in {".jsonl", ".json"}:
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"{path}:{line_number} must be a JSON object"
                        )
                    rows.append(value)
            return rows

        raise ValueError(
            f"Unsupported training-data extension {suffix!r}; use .parquet or .jsonl"
        )

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        # pandas/NumPy missing scalar values compare unequal to themselves.
        try:
            if bool(value != value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    @classmethod
    def _normalize_record(
        cls,
        path: Path,
        row_number: int,
        raw_row: dict[str, Any],
    ) -> dict[str, Any]:
        row = dict(raw_row)
        is_sequential_math = "question" in row or "answer" in row

        if is_sequential_math:
            prompt = cls._text(row.get("question"))
            target = cls._text(row.get("answer"))
            row.setdefault("task_type", "math")
        else:
            prompt = cls._text(row.get("prompt"))
            target = cls._text(
                row.get("canonical_solution", row.get("solution", row.get("code")))
            )
            row.setdefault("task_type", "code")

        if not prompt or not target:
            expected = (
                "question and answer"
                if is_sequential_math
                else "prompt and target code"
            )
            raise ValueError(f"{path}: row {row_number} needs non-empty {expected}")

        row["prompt"] = prompt
        row["target_text"] = target
        # These traces supervise role-specific Inner Adapters during warm-up;
        # they are never inserted into inference-time CSI feature inputs.
        row["plan"] = cls._text(row.get("plan"))
        row["refined_plan"] = cls._text(row.get("refined_plan"))
        # Retain the old normalized key for compatibility with existing code
        # and checkpoints whose recorded arguments mention this field.
        row["canonical_solution"] = target
        row.setdefault("task_id", f"{path.stem}_{row_number}")
        return row

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    # Fill these three defaults on your other computer, or pass CLI values.
    parser.add_argument("--planner-model-path", default="pre-trained/Qwen3-0.6B")
    parser.add_argument("--critic-model-path", default="pre-trained/Qwen3.5-0.8B")
    parser.add_argument("--executor-model-path", default="pre-trained/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--planner-device", default="cuda:4")
    parser.add_argument("--critic-device", default="cuda:5")
    parser.add_argument("--executor-device", default="cuda:6")
    parser.add_argument("--csi-device", default="cuda:7")
    parser.add_argument(
        "--model-dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--csi-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CSI Conv1D+GCN with three frozen Qwen agents"
    )
    add_model_arguments(parser)
    parser.add_argument(
        "--train-parquet",
        default="",
        help="Training Parquet file, e.g. Sequential-Math/data/train-00000-of-00001.parquet",
    )
    # Kept for the original code-generation examples.
    parser.add_argument("--train-jsonl", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("csi_checkpoints"))
    parser.add_argument("--resume", default="")

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--latent-tokens", type=int, default=32)
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Odd Conv1D kernel size over the five feature types",
    )
    parser.add_argument(
        "--dilations",
        type=int,
        nargs="+",
        default=[1],
        help="Dilations over the five-element feature axis (default: 1 2)",
    )
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--max-feature-tokens", type=int, default=512)
    parser.add_argument(
        "--intermediate-loss-weight",
        type=float,
        default=0.0,
        help="Non-final CSI-round alignment weight (default: disabled)",
    )

    parser.add_argument(
        "--inner-warmup-epochs",
        type=int,
        default=1,
        help=(
            "Dataset passes completed separately by each Inner role before "
            "Outer CSI training (default: 1; use 0 to disable)"
        ),
    )
    parser.add_argument("--inner-learning-rate", type=float, default=5e-4)
    parser.add_argument("--inner-cos-weight", type=float, default=1.0)
    parser.add_argument("--inner-mse-weight", type=float, default=0.0)
    parser.add_argument(
        "--inner-max-updates",
        type=int,
        default=-1,
        help="Maximum Inner updates per role; -1 is unlimited",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-updates", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_paths(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "planner_model_path",
            "critic_model_path",
            "executor_model_path",
        )
        if not getattr(args, name).strip()
    ]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(f"Fill or pass the required paths: {flags}")
    if bool(args.train_parquet) == bool(args.train_jsonl):
        raise SystemExit("Pass exactly one of --train-parquet or --train-jsonl")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.log_every < 1 or args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ValueError("log_every, learning_rate and max_grad_norm must be positive")
    if args.inner_warmup_epochs < 0 or args.inner_learning_rate <= 0:
        raise ValueError("inner_warmup_epochs must be non-negative and inner_learning_rate positive")
    if args.inner_cos_weight < 0 or args.inner_mse_weight < 0:
        raise ValueError("inner warm-up loss weights must be non-negative")
    if args.inner_warmup_epochs > 0 and args.inner_cos_weight + args.inner_mse_weight <= 0:
        raise ValueError("At least one inner warm-up loss weight must be positive")
    if args.intermediate_loss_weight < 0:
        raise ValueError("intermediate_loss_weight must be non-negative")


def construct_system(args: argparse.Namespace) -> CSIQwenSystem:
    return build_qwen_system(
        planner_model_path=args.planner_model_path,
        critic_model_path=args.critic_model_path,
        executor_model_path=args.executor_model_path,
        planner_device=args.planner_device,
        critic_device=args.critic_device,
        executor_device=args.executor_device,
        csi_device=args.csi_device,
        model_dtype=args.model_dtype,
        csi_dtype=args.csi_dtype,
        d_model=args.d_model,
        latent_tokens=args.latent_tokens,
        prefix_tokens=args.prefix_tokens,
        kernel_size=args.kernel_size,
        dilations=args.dilations,
        max_steps=args.max_steps,
        tolerance=args.tolerance,
        max_feature_tokens=args.max_feature_tokens,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )


def batch_loss(
    system: CSIQwenSystem,
    rows: list[dict[str, Any]],
    *,
    intermediate_loss_weight: float = 0.0,
) -> dict[str, Any]:
    prompts = [row["prompt"] for row in rows]
    targets = [row["target_text"] for row in rows]
    return system.training_loss(
        prompts,
        rows,
        targets,
        intermediate_loss_weight=intermediate_loss_weight,
    )


def validate_inner_supervision(dataset: CodeTrainingDataset) -> None:
    missing: list[str] = []
    for row in dataset.rows:
        absent = [key for key in ("plan", "refined_plan") if not row.get(key)]
        if absent:
            missing.append(f"{row.get('task_id', '<unknown>')}: {', '.join(absent)}")
            if len(missing) == 5:
                break
    if missing:
        details = "; ".join(missing)
        raise ValueError(
            "Inner warm-up requires real plan and refined_plan supervision. "
            f"Missing fields in: {details}. Supply those fields or pass "
            "--inner-warmup-epochs 0."
        )


def run_inner_warmup(
    system: CSIQwenSystem,
    train_loader: DataLoader,
    args: argparse.Namespace,
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    start_updates: dict[str, int] | None = None,
) -> dict[str, int]:
    """Train Planner, Refiner, and Solver Inner Adapters sequentially."""
    missing_optimizers = [role for role in AGENT_ROLES if role not in optimizers]
    if missing_optimizers:
        raise ValueError(f"Missing Inner optimizers for roles: {missing_optimizers}")

    role_updates = {
        role: int((start_updates or {}).get(role, 0))
        for role in AGENT_ROLES
    }
    summaries: dict[str, tuple[int, float]] = {}
    print(
        "stage=inner starting separate role training: "
        "Planner -> Refiner -> Solver",
        flush=True,
    )

    for role, adapter in zip(AGENT_ROLES, system.inner_adapters):
        label = INNER_ROLE_LABELS[role]
        optimizer = optimizers[role]
        parameters = list(adapter.parameters())
        system.set_inner_adapters_trainable(True, role=role)
        optimizer.zero_grad(set_to_none=True)
        log_sums = {"loss": 0.0, "cosine_loss": 0.0, "mse_loss": 0.0}
        log_updates = 0
        last_loss = float("nan")
        stop_role = (
            args.inner_max_updates > 0
            and role_updates[role] >= args.inner_max_updates
        )

        print(
            f"{label}: starting {args.inner_warmup_epochs} pass(es)",
            flush=True,
        )
        for inner_epoch in range(1, args.inner_warmup_epochs + 1):
            if stop_role:
                break
            accumulated = {"loss": 0.0, "cosine_loss": 0.0, "mse_loss": 0.0}
            accumulated_batches = 0
            for batch_index, rows in enumerate(train_loader, start=1):
                result = system.inner_role_loss(
                    rows,
                    role,
                    cosine_weight=args.inner_cos_weight,
                    mse_weight=args.inner_mse_weight,
                )
                group_start = (
                    (batch_index - 1) // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                group_size = min(
                    args.gradient_accumulation_steps,
                    len(train_loader) - group_start,
                )
                (result["loss"] / group_size).backward()
                for name in accumulated:
                    accumulated[name] += float(result[name].detach().item())
                accumulated_batches += 1

                should_update = (
                    batch_index % args.gradient_accumulation_steps == 0
                    or batch_index == len(train_loader)
                )
                if not should_update:
                    continue

                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                role_updates[role] += 1
                update_metrics = {
                    name: value / max(accumulated_batches, 1)
                    for name, value in accumulated.items()
                }
                last_loss = update_metrics["loss"]
                for name in log_sums:
                    log_sums[name] += update_metrics[name]
                    accumulated[name] = 0.0
                accumulated_batches = 0
                log_updates += 1

                if role_updates[role] % args.log_every == 0:
                    averages = {
                        name: value / max(log_updates, 1)
                        for name, value in log_sums.items()
                    }
                    print(
                        f"{label}: inner_epoch={inner_epoch} "
                        f"update={role_updates[role]} "
                        f"loss={averages['loss']:.4f} "
                        f"cosine={averages['cosine_loss']:.4f} "
                        f"mse={averages['mse_loss']:.4f}",
                        flush=True,
                    )
                    log_sums = {name: 0.0 for name in log_sums}
                    log_updates = 0

                if (
                    args.inner_max_updates > 0
                    and role_updates[role] >= args.inner_max_updates
                ):
                    stop_role = True
                    break

        if log_updates:
            averages = {
                name: value / log_updates
                for name, value in log_sums.items()
            }
            last_loss = averages["loss"]
            print(
                f"{label}: update={role_updates[role]} "
                f"loss={averages['loss']:.4f} "
                f"cosine={averages['cosine_loss']:.4f} "
                f"mse={averages['mse_loss']:.4f}",
                flush=True,
            )
        summaries[role] = (role_updates[role], last_loss)
        print(
            f"{label}: complete updates={role_updates[role]} "
            f"last_loss={last_loss:.4f}",
            flush=True,
        )

    system.set_inner_adapters_trainable(False)
    print("stage=inner summary", flush=True)
    for role in AGENT_ROLES:
        updates, last_loss = summaries[role]
        print(
            f"{INNER_ROLE_LABELS[role]}: updates={updates} "
            f"last_loss={last_loss:.4f}",
            flush=True,
        )
    return role_updates


def save_checkpoint(
    path: Path,
    system: CSIQwenSystem,
    outer_optimizer: torch.optim.Optimizer,
    *,
    inner_optimizers: dict[str, torch.optim.Optimizer] | None,
    epoch: int,
    update: int,
    inner_updates: dict[str, int],
    phase: str,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": system.state_dict(),
        "optimizer_state_dict": outer_optimizer.state_dict(),
        "model_config": system.checkpoint_config(),
        "training_schedule": TRAINING_SCHEDULE,
        "training_phase": phase,
        "epoch": epoch,
        "update": update,
        "inner_update": sum(inner_updates.values()),
        "inner_updates": dict(inner_updates),
        "training_args": vars(args),
    }
    if inner_optimizers:
        payload["inner_optimizer_state_dicts"] = {
            role: optimizer.state_dict()
            for role, optimizer in inner_optimizers.items()
        }
    torch.save(payload, path)


def main() -> int:
    args = build_parser().parse_args()
    validate_paths(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_path = args.train_parquet or args.train_jsonl
    train_dataset = CodeTrainingDataset(Path(train_path).expanduser().resolve())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_records,
        generator=generator,
    )
    print(
        f"micro_batch_size={args.batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"effective_batch_size="
        f"{args.batch_size * args.gradient_accumulation_steps}",
        flush=True,
    )
    print("Loading three frozen Qwen agents and CSI modules...", flush=True)
    system = construct_system(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resume_payload: dict[str, Any] | None = None
    if args.resume:
        resume_payload = system.load_trainable_checkpoint(args.resume)
        schedule = resume_payload.get("training_schedule")
        if schedule != TRAINING_SCHEDULE:
            raise RuntimeError(
                "This checkpoint was not produced by the separate-role Inner "
                "schedule. Start a fresh run so that joint or interleaved "
                "Inner updates are not mixed into this experiment."
            )
        print(
            "Resume checkpoint loaded; continuing separate-role Inner/Outer training.",
            flush=True,
        )

    inner_optimizers: dict[str, torch.optim.Optimizer] = {}
    if args.inner_warmup_epochs > 0:
        validate_inner_supervision(train_dataset)
        for role, adapter in zip(AGENT_ROLES, system.inner_adapters):
            parameters = list(adapter.parameters())
            parameter_count = sum(parameter.numel() for parameter in parameters)
            print(
                f"{INNER_ROLE_LABELS[role]} Inner Adapter parameters: "
                f"{parameter_count:,}",
                flush=True,
            )
            inner_optimizers[role] = torch.optim.AdamW(
                parameters,
                lr=args.inner_learning_rate,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.95),
            )

    # Outer training always freezes the fully warmed Inner Adapters. Constructing
    # the outer optimizer after this switch keeps the parameter sets disjoint.
    system.set_inner_adapters_trainable(False)
    trainable_parameters = [parameter for parameter in system.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(f"Trainable outer CSI parameters: {trainable_count:,}", flush=True)
    outer_optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    global_update = 0
    inner_updates = {role: 0 for role in AGENT_ROLES}
    inner_complete = args.inner_warmup_epochs == 0
    if resume_payload is not None:
        if "optimizer_state_dict" in resume_payload:
            outer_optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if inner_optimizers:
            saved_inner_optimizers = resume_payload.get(
                "inner_optimizer_state_dicts"
            )
            if not isinstance(saved_inner_optimizers, dict):
                print(
                    "WARNING: checkpoint has no separate Inner optimizer states; "
                    "restarting the role optimizers.",
                    flush=True,
                )
            else:
                for role, optimizer in inner_optimizers.items():
                    if role in saved_inner_optimizers:
                        optimizer.load_state_dict(saved_inner_optimizers[role])
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        global_update = int(resume_payload.get("update", 0))
        saved_inner_updates = resume_payload.get("inner_updates", {})
        if isinstance(saved_inner_updates, dict):
            inner_updates = {
                role: int(saved_inner_updates.get(role, 0))
                for role in AGENT_ROLES
            }
        resume_phase = str(resume_payload.get("training_phase", ""))
        if resume_phase not in {"inner_complete", "outer"}:
            raise RuntimeError(
                f"Unsupported checkpoint training_phase={resume_phase!r}; "
                "expected 'inner_complete' or 'outer'."
            )
        inner_complete = True
        if resume_phase == "inner_complete":
            start_epoch = 1
            global_update = 0

    if inner_optimizers and not inner_complete:
        inner_updates = run_inner_warmup(
            system,
            train_loader,
            args,
            inner_optimizers,
            start_updates=inner_updates,
        )
        print("Inner stage complete; starting Outer training.", flush=True)

    stop_training = False
    last_epoch = start_epoch - 1
    log_loss_sum = 0.0
    log_final_ce_sum = 0.0
    log_intermediate_sum = 0.0
    log_update_count = 0
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        print(f"stage=outer epoch={epoch} starting CSI training...", flush=True)
        system.train()
        system.set_inner_adapters_trainable(False)
        outer_optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_final_ce = 0.0
        accumulated_intermediate = 0.0
        accumulated_batches = 0

        for batch_index, rows in enumerate(train_loader, start=1):
            result = batch_loss(
                system,
                rows,
                intermediate_loss_weight=args.intermediate_loss_weight,
            )
            loss = result["loss"]
            # The last accumulation group can contain fewer micro-batches than
            # requested. Scale it by its actual size so that its gradients are
            # not accidentally made smaller than a normal optimizer update.
            group_start = (
                (batch_index - 1) // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps,
                len(train_loader) - group_start,
            )
            (loss / group_size).backward()
            accumulated_loss += float(loss.detach().item())
            accumulated_final_ce += float(result["final_ce_loss"].item())
            accumulated_intermediate += float(result["intermediate_loss"].item())
            accumulated_batches += 1

            should_update = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                outer_optimizer.step()
                outer_optimizer.zero_grad(set_to_none=True)
                global_update += 1

                update_loss = accumulated_loss / max(accumulated_batches, 1)
                log_loss_sum += update_loss
                log_final_ce_sum += accumulated_final_ce / max(accumulated_batches, 1)
                log_intermediate_sum += (
                    accumulated_intermediate / max(accumulated_batches, 1)
                )
                log_update_count += 1
                accumulated_loss = 0.0
                accumulated_final_ce = 0.0
                accumulated_intermediate = 0.0
                accumulated_batches = 0

                if global_update % args.log_every == 0:
                    average = log_loss_sum / max(log_update_count, 1)
                    final_ce_average = log_final_ce_sum / max(log_update_count, 1)
                    intermediate_average = (
                        log_intermediate_sum / max(log_update_count, 1)
                    )
                    print(
                        f"epoch={epoch} update={global_update} loss={average:.4f} "
                        f"final_ce={final_ce_average:.4f} "
                        f"intermediate={intermediate_average:.4f} "
                        f"csi_steps={result['steps']}",
                        flush=True,
                    )
                    log_loss_sum = 0.0
                    log_final_ce_sum = 0.0
                    log_intermediate_sum = 0.0
                    log_update_count = 0
                if args.max_updates > 0 and global_update >= args.max_updates:
                    stop_training = True
                    break

        if stop_training:
            break

    final_path = args.output_dir / "csi_final.pt"
    save_checkpoint(
        final_path,
        system,
        outer_optimizer,
        inner_optimizers=inner_optimizers,
        epoch=last_epoch,
        update=global_update,
        inner_updates=inner_updates,
        phase="outer",
        args=args,
    )
    print(f"Final checkpoint: {final_path.resolve()}")
    system.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
