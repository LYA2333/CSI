"""Tests for JSONL and Sequential-Math Parquet dataset normalization."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from csi_qwen import MATH_SYSTEM_PROMPT, build_agent_feature_texts, generation_prompt
from train_csi import (
    CodeTrainingDataset,
    INNER_ROLE_LABELS,
    TRAINING_SCHEDULE,
    build_parser,
    run_inner_warmup,
    save_checkpoint,
)


class _FakeTwoStageSystem(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner_adapters = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False) for _ in range(3)]
        )

    def set_inner_adapters_trainable(
        self,
        enabled: bool,
        *,
        role: str | None = None,
    ) -> None:
        role_index = {"planner": 0, "critic": 1, "executor": 2}
        for index, adapter in enumerate(self.inner_adapters):
            selected = enabled and (role is None or index == role_index[role])
            for parameter in adapter.parameters():
                parameter.requires_grad_(selected)

    def inner_role_loss(
        self,
        rows,
        role: str,
        *,
        cosine_weight: float,
        mse_weight: float,
    ):
        del rows, cosine_weight, mse_weight
        role_index = {"planner": 0, "critic": 1, "executor": 2}
        index = role_index[role]
        weight = self.inner_adapters[index].weight
        loss = (weight - float(index + 1)).square().mean()
        return {
            "loss": loss,
            "cosine_loss": loss.detach(),
            "mse_loss": torch.zeros_like(loss.detach()),
        }

    @staticmethod
    def checkpoint_config():
        return {"test_model": True}


class _FakeFrame:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def to_dict(self, *, orient: str) -> list[dict[str, str]]:
        if orient != "records":
            raise AssertionError(f"Unexpected orientation: {orient}")
        return self.rows


class TrainingDatasetTests(unittest.TestCase):
    def test_separate_role_inner_is_enabled_and_intermediate_is_disabled(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.inner_warmup_epochs, 1)
        self.assertEqual(args.intermediate_loss_weight, 0.0)

    def test_inner_stage_trains_each_role_separately_and_labels_output(self) -> None:
        system = _FakeTwoStageSystem()
        roles = ("planner", "critic", "executor")
        optimizers = {
            role: torch.optim.AdamW(system.inner_adapters[index].parameters(), lr=0.05)
            for index, role in enumerate(roles)
        }
        args = SimpleNamespace(
            inner_max_updates=-1,
            inner_warmup_epochs=2,
            gradient_accumulation_steps=1,
            inner_cos_weight=1.0,
            inner_mse_weight=0.0,
            max_grad_norm=1.0,
            log_every=100,
        )
        loader = [
            [{"task_id": "sample-1"}],
            [{"task_id": "sample-2"}],
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            update_counts = run_inner_warmup(
                system,
                loader,
                args,
                optimizers,
            )

        self.assertEqual(update_counts, {role: 4 for role in roles})
        self.assertTrue(all(optimizer.state for optimizer in optimizers.values()))
        self.assertFalse(
            any(parameter.requires_grad for parameter in system.inner_adapters.parameters())
        )
        log = output.getvalue()
        for label in INNER_ROLE_LABELS.values():
            self.assertIn(f"{label}:", log)

    def test_checkpoint_contains_outer_and_role_optimizer_states(self) -> None:
        system = _FakeTwoStageSystem()
        roles = ("planner", "critic", "executor")
        inner_optimizers = {
            role: torch.optim.AdamW(system.inner_adapters[index].parameters(), lr=0.01)
            for index, role in enumerate(roles)
        }
        outer_optimizer = torch.optim.AdamW(system.parameters(), lr=0.02)
        args = SimpleNamespace(epochs=3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                system,
                outer_optimizer,
                inner_optimizers=inner_optimizers,
                epoch=2,
                update=7,
                inner_updates={"planner": 3, "critic": 4, "executor": 5},
                phase="outer",
                args=args,
            )
            try:
                payload = torch.load(path, weights_only=False)
            except TypeError:
                payload = torch.load(path)

        self.assertEqual(payload["training_schedule"], TRAINING_SCHEDULE)
        self.assertEqual(payload["training_phase"], "outer")
        self.assertEqual(payload["inner_update"], 12)
        self.assertEqual(
            payload["inner_updates"],
            {"planner": 3, "critic": 4, "executor": 5},
        )
        self.assertIn("optimizer_state_dict", payload)
        self.assertEqual(
            set(payload["inner_optimizer_state_dicts"]),
            set(roles),
        )

    def test_sequential_math_parquet_mapping(self) -> None:
        source_rows = [
            {
                "question": "What is 2 + 3?",
                "plan": "Add the two integers.",
                "refined_plan": "Compute the sum and verify it.",
                "answer": "2 + 3 = 5.",
                "answer_field": "gemini_attempt",
            }
        ]
        calls: list[Path] = []

        def read_parquet(path: Path) -> _FakeFrame:
            calls.append(path)
            return _FakeFrame(source_rows)

        fake_pandas = SimpleNamespace(read_parquet=read_parquet)
        path = Path("Sequential-Math/data/train-00000-of-00001.parquet")
        with patch.dict(sys.modules, {"pandas": fake_pandas}):
            dataset = CodeTrainingDataset(path)

        self.assertEqual(calls, [path])
        self.assertEqual(len(dataset), 1)
        row = dataset[0]
        self.assertEqual(row["prompt"], source_rows[0]["question"])
        self.assertEqual(row["target_text"], source_rows[0]["answer"])
        self.assertEqual(row["task_type"], "math")
        self.assertEqual(row["plan"], source_rows[0]["plan"])
        self.assertNotIn("experience", row)

    def test_existing_jsonl_mapping_is_preserved(self) -> None:
        record = {
            "task_id": "add",
            "prompt": "Write add(a, b).",
            "canonical_solution": "def add(a, b): return a + b",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = CodeTrainingDataset(path)

        self.assertEqual(dataset[0]["task_type"], "code")
        self.assertEqual(dataset[0]["target_text"], record["canonical_solution"])

    def test_math_records_select_math_prompts(self) -> None:
        task = {"task_type": "math"}
        system_prompt, user_prompt = generation_prompt("Compute 2 + 3.", task)
        self.assertEqual(system_prompt, MATH_SYSTEM_PROMPT)
        self.assertIn("final answer", user_prompt)

        features = build_agent_feature_texts("Compute 2 + 3.", task, "planner")
        self.assertTrue(features[0].startswith("Mathematical reasoning task:"))
        self.assertIn("mathematical problem", features[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
