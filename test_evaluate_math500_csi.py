"""Dependency-light tests for the MATH-500 CSI evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluate_math500_csi import (
    MathGrader,
    build_parser,
    extract_final_answer,
    extract_last_boxed,
    load_records,
    normalize_answer,
    normalize_math500_rows,
    normalized_equivalent,
    summarize_stage_results,
)


class Math500EvaluatorTests(unittest.TestCase):
    def test_eval_stage_can_select_after_only(self) -> None:
        args = build_parser().parse_args(
            [
                "--dataset-path",
                "math500.parquet",
                "--checkpoint-path",
                "csi.pt",
                "--planner-model-path",
                "planner",
                "--critic-model-path",
                "critic",
                "--executor-model-path",
                "solver",
                "--eval-stage",
                "after",
            ]
        )
        self.assertEqual(args.eval_stage, "after")
        self.assertTrue(args.stop_on_boxed)
        self.assertTrue(args.stop_on_repetition)
        self.assertEqual(args.repetition_min_tokens, 64)
        self.assertEqual(args.repetition_max_pattern_tokens, 16)
        self.assertFalse(args.profile_latency)

    def test_generation_stops_can_be_disabled(self) -> None:
        args = build_parser().parse_args(
            [
                "--dataset-path",
                "math500.parquet",
                "--checkpoint-path",
                "csi.pt",
                "--planner-model-path",
                "planner",
                "--critic-model-path",
                "critic",
                "--executor-model-path",
                "solver",
                "--no-stop-on-boxed",
                "--no-stop-on-repetition",
            ]
        )
        self.assertFalse(args.stop_on_boxed)
        self.assertFalse(args.stop_on_repetition)

    def test_stage_summary_reports_tokens_latency_and_finish_reasons(self) -> None:
        summary = summarize_stage_results(
            [
                {
                    "correct": True,
                    "latency_seconds": 2.0,
                    "generated_tokens": 20,
                    "solver_prompt_tokens": 10,
                    "csi_prefix_tokens": 4,
                    "finish_reason": "boxed",
                    "timing_seconds": {"collaboration": 0.5, "solver_decode": 1.0},
                    "error": "",
                },
                {
                    "correct": False,
                    "latency_seconds": 4.0,
                    "generated_tokens": 40,
                    "solver_prompt_tokens": 14,
                    "csi_prefix_tokens": 4,
                    "finish_reason": "length",
                    "timing_seconds": {"collaboration": 1.5, "solver_decode": 2.0},
                    "error": "",
                },
            ]
        )
        self.assertEqual(summary["total_generated_tokens"], 60)
        self.assertEqual(summary["mean_generated_tokens"], 30)
        self.assertEqual(summary["generated_tokens_per_second"], 10)
        self.assertEqual(summary["finish_reasons"], {"boxed": 1, "length": 1})
        self.assertEqual(summary["mean_timing_seconds"]["collaboration"], 1.0)

    def test_extracts_last_balanced_boxed_answer(self) -> None:
        text = r"First \fbox{wrong}, then final $\boxed{\frac{1}{2}}$."
        self.assertEqual(extract_last_boxed(text), r"\frac{1}{2}")
        self.assertEqual(extract_final_answer(text), r"\frac{1}{2}")

    def test_normalization_and_simple_symbolic_equivalence(self) -> None:
        self.assertEqual(normalize_answer(r"$\boxed{\dfrac{1}{2}}$"), r"\frac{1}{2}")
        self.assertTrue(normalized_equivalent(r"\frac{1}{2}", "Final Answer: 0.5"))
        self.assertTrue(normalized_equivalent(r"\sqrt{4}", r"\boxed{2}"))
        self.assertFalse(normalized_equivalent("3", r"\boxed{4}"))

    def test_math_grader_normalized_mode(self) -> None:
        grader = MathGrader("normalized")
        self.assertEqual(grader.name, "normalized_fallback")
        self.assertTrue(grader.equivalent("5", r"The answer is \boxed{5}."))

    def test_jsonl_loading_and_math500_schema(self) -> None:
        records = [
            {
                "unique_id": "sample-1",
                "problem": "Compute one half.",
                "solution": r"Therefore the answer is $\boxed{\frac{1}{2}}$.",
                "answer": r"\frac{1}{2}",
                "subject": "Algebra",
                "level": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            loaded = load_records(path, "test")

        examples = normalize_math500_rows(loaded, offset=0, limit=0)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["task_id"], "sample-1")
        self.assertEqual(examples[0]["prompt"], records[0]["problem"])
        self.assertEqual(examples[0]["reference_answer"], records[0]["answer"])
        self.assertEqual(examples[0]["task_type"], "math")

    def test_solution_box_is_used_when_answer_field_is_missing(self) -> None:
        examples = normalize_math500_rows(
            [
                {
                    "problem": "Compute 2 + 3.",
                    "solution": r"We obtain $\boxed{5}$.",
                }
            ],
            offset=0,
            limit=0,
        )
        self.assertEqual(examples[0]["reference_answer"], "5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
