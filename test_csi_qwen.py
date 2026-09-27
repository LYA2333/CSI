"""Dependency-light tests for the CSI-Qwen integration.

These fakes validate tensor flow and gradients without downloading Qwen weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from csi_qwen import (
    CSIQwenSystem,
    FEATURE_NAMES,
    GenerationResult,
    INNER_ADAPTER_TYPE,
    InnerRecursiveAdapter,
    _complete_boxed_end,
    _has_repeated_token_suffix,
    generation_prompt,
    role_supervision_example,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [3 + (ord(character) % 25) for character in text][:64] or [3]

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        return_tensors: str,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Tensor]:
        del return_tensors, truncation, add_special_tokens
        rows = [text] if isinstance(text, str) else list(text)
        encoded = [self._encode(row)[:max_length] for row in rows]
        width = max(len(row) for row in encoded) if padding else len(encoded[0])
        ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(encoded):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
        enable_thinking: bool = False,
    ) -> Tensor:
        del tokenize, add_generation_prompt, return_tensors, enable_thinking
        text = "\n".join(message["content"] for message in messages)
        return torch.tensor([self._encode(text)], dtype=torch.long)

    @staticmethod
    def decode(token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"token_{token_id}" for token_id in token_ids)


class FakeCausalLM(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.explicit_embedding_inputs: list[Tensor] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, past_key_values, return_dict
        if inputs_embeds is not None:
            self.explicit_embedding_inputs.append(inputs_embeds.detach().clone())
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Need input_ids or inputs_embeds")
            inputs_embeds = self.embedding(input_ids)

        # Cumulative context makes target logits depend on the soft prefix.
        positions = torch.arange(
            1,
            inputs_embeds.shape[1] + 1,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).view(1, -1, 1)
        contextual = inputs_embeds.cumsum(dim=1) / positions
        logits = self.lm_head(contextual)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        cache = (torch.ones(1),) if use_cache else None
        return SimpleNamespace(logits=logits, loss=loss, past_key_values=cache)


class GenerationStopHelperTests(unittest.TestCase):
    def test_math_generation_prompt_requires_boxed_final_answer(self) -> None:
        _, user_prompt = generation_prompt(
            "Compute 2 + 3.",
            {"task_type": "math"},
        )
        self.assertIn(r"inside \boxed{}", user_prompt)
        self.assertIn(r"\boxed{1}", user_prompt)

    def test_complete_boxed_requires_balanced_non_empty_content(self) -> None:
        text = r"Reasoning... final answer: \boxed{\frac{1}{2}} trailing text"
        end = _complete_boxed_end(text)
        self.assertIsNotNone(end)
        assert end is not None
        self.assertEqual(text[:end], r"Reasoning... final answer: \boxed{\frac{1}{2}}")
        self.assertIsNone(_complete_boxed_end(r"unfinished \boxed{\frac{1}{2}"))
        self.assertIsNone(_complete_boxed_end(r"empty \boxed{}"))

    def test_repetition_stop_ignores_short_legitimate_runs(self) -> None:
        self.assertFalse(
            _has_repeated_token_suffix(
                [1] * 8,
                min_repeated_tokens=64,
                max_pattern_tokens=16,
            )
        )
        self.assertTrue(
            _has_repeated_token_suffix(
                [1] * 64,
                min_repeated_tokens=64,
                max_pattern_tokens=16,
            )
        )
        self.assertTrue(
            _has_repeated_token_suffix(
                [4, 5, 6, 7] * 16,
                min_repeated_tokens=64,
                max_pattern_tokens=16,
            )
        )
        self.assertFalse(
            _has_repeated_token_suffix(
                list(range(64)),
                min_repeated_tokens=64,
                max_pattern_tokens=16,
            )
        )


class FakeQwenAgent:
    def __init__(self, role: str, hidden_size: int) -> None:
        self.role = role
        self.hidden_size = hidden_size
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer()
        self.model = FakeCausalLM(hidden_size)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_feature_texts(
        self,
        batch_feature_texts: Sequence[Sequence[str]],
        *,
        latent_tokens: int,
    ) -> Tensor:
        rows = []
        base = torch.arange(self.hidden_size, dtype=torch.float32)
        for sample in batch_feature_texts:
            self_features = []
            for text in sample:
                positions = torch.arange(latent_tokens, dtype=torch.float32)[:, None]
                self_features.append(torch.sin(base[None, :] + positions + len(text)))
            rows.append(torch.stack(self_features))
        return torch.stack(rows)

    def build_generation_prompt_ids(
        self,
        prompt: str,
        task: Mapping[str, Any],
    ) -> Tensor:
        del task
        return torch.tensor(self.tokenizer._encode(prompt), dtype=torch.long)

    def tokenize_target(self, code: str) -> Tensor:
        return torch.tensor(self.tokenizer._encode(code) + [2], dtype=torch.long)

    def encode_supervision_pair(
        self,
        system_prompt: str,
        user_prompt: str,
        target_text: str,
    ) -> tuple[Tensor, Tensor]:
        del system_prompt, user_prompt
        target_ids = self.tokenize_target(target_text)
        with torch.no_grad():
            target_embeddings = self.model.get_input_embeddings()(target_ids)
            offsets = torch.linspace(
                0.05,
                0.25,
                target_ids.numel(),
                dtype=target_embeddings.dtype,
            ).unsqueeze(-1)
            hidden = torch.tanh(target_embeddings + offsets)
        return hidden, target_embeddings


class CSIQwenSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.system = CSIQwenSystem(
            FakeQwenAgent("planner", 12),
            FakeQwenAgent("critic", 16),
            FakeQwenAgent("executor", 20),
            d_model=8,
            latent_tokens=8,
            prefix_tokens=4,
            dilations=(1, 2),
            max_steps=2,
            csi_device="cpu",
            csi_dtype="float32",
        )
        self.tasks = [
            {
                "task_id": "add",
                "prompt": "Write add(a, b).",
                "entry_point": "add",
                "plan": "Return the sum of both arguments.",
                "refined_plan": "Use direct addition and return the result.",
                "target_text": "def add(a, b): return a + b",
            },
            {
                "task_id": "sub",
                "prompt": "Write sub(a, b).",
                "entry_point": "sub",
                "plan": "Subtract the second argument from the first.",
                "refined_plan": "Use direct subtraction and return the result.",
                "target_text": "def sub(a, b): return a - b",
            },
        ]

    def test_inner_supervision_uses_role_specific_context_and_targets(self) -> None:
        task = self.tasks[0]

        _, planner_prompt, planner_target = role_supervision_example(
            task["prompt"], task, "planner"
        )
        self.assertIn(task["prompt"], planner_prompt)
        self.assertNotIn(task["plan"], planner_prompt)
        self.assertNotIn(task["refined_plan"], planner_prompt)
        self.assertEqual(planner_target, task["plan"])

        _, critic_prompt, critic_target = role_supervision_example(
            task["prompt"], task, "critic"
        )
        self.assertIn(task["prompt"], critic_prompt)
        self.assertIn(task["plan"], critic_prompt)
        self.assertNotIn(task["refined_plan"], critic_prompt)
        self.assertEqual(critic_target, task["refined_plan"])

        _, executor_prompt, executor_target = role_supervision_example(
            task["prompt"], task, "executor"
        )
        self.assertIn(task["prompt"], executor_prompt)
        self.assertIn(task["plan"], executor_prompt)
        self.assertIn(task["refined_plan"], executor_prompt)
        self.assertEqual(executor_target, task["target_text"])

    def test_inner_warmup_uses_role_targets_and_reaches_adapters(self) -> None:
        result = self.system.inner_warmup_loss(self.tasks)
        self.assertTrue(torch.isfinite(result["loss"]).item())
        result["loss"].backward()

        for adapter in self.system.inner_adapters:
            self.assertIsNotNone(adapter.proj2.weight.grad)
            self.assertTrue(torch.isfinite(adapter.proj2.weight.grad).all().item())

    def test_inner_adapter_post_norm_controls_output_scale(self) -> None:
        adapter = InnerRecursiveAdapter(8)
        hidden = 100.0 * torch.randn(2, 4, 8) + 50.0
        output = adapter(hidden)

        torch.testing.assert_close(
            output.mean(dim=-1),
            torch.zeros(2, 4),
            atol=1e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output.var(dim=-1, unbiased=False),
            torch.ones(2, 4),
            atol=1e-4,
            rtol=0.0,
        )

    def test_inner_role_loss_updates_only_selected_adapter(self) -> None:
        self.system.zero_grad(set_to_none=True)
        self.system.set_inner_adapters_trainable(True, role="critic")
        result = self.system.inner_role_loss(self.tasks, "critic")
        result["loss"].backward()

        self.assertIsNone(self.system.inner_adapters[0].proj2.weight.grad)
        self.assertIsNotNone(self.system.inner_adapters[1].proj2.weight.grad)
        self.assertIsNotNone(self.system.inner_adapters[1].post_norm.weight.grad)
        self.assertIsNone(self.system.inner_adapters[2].proj2.weight.grad)

    def test_feature_projection_shape_and_directed_cycle(self) -> None:
        features = self.system.encode_static_features(
            [task["prompt"] for task in self.tasks],
            self.tasks,
        )
        self.assertEqual(
            tuple(features.shape),
            (2, 3, len(FEATURE_NAMES), 8, 8),
        )
        expected_graph = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        torch.testing.assert_close(self.system.collaboration_graph, expected_graph)
        config = self.system.checkpoint_config()
        self.assertEqual(config["intra_conv_axis"], "feature_types")
        self.assertEqual(config["feature_receptive_field"], 5)
        self.assertEqual(config["prefix_position"], "after_prompt")
        self.assertEqual(config["inner_adapter_type"], INNER_ADAPTER_TYPE)

    def test_checkpoint_rejects_pre_norm_only_inner_adapter(self) -> None:
        config = self.system.checkpoint_config()
        config["inner_adapter_type"] = "pre_ln_residual_mlp"
        payload = {
            "model_state_dict": self.system.state_dict(),
            "model_config": config,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old_inner_adapter.pt"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "incompatible Inner Adapter"):
                self.system.load_trainable_checkpoint(str(checkpoint))

    def test_training_places_prefix_after_prompt(self) -> None:
        executor_model = self.system.executor.model
        executor_model.explicit_embedding_inputs.clear()
        self.system.training_loss(
            [self.tasks[0]["prompt"]],
            [self.tasks[0]],
            [self.tasks[0]["target_text"]],
            intermediate_loss_weight=0.0,
        )

        actual = executor_model.explicit_embedding_inputs[-1][0]
        prompt_ids = self.system.executor.build_generation_prompt_ids(
            self.tasks[0]["prompt"],
            self.tasks[0],
        )
        target_ids = self.system.executor.tokenize_target(self.tasks[0]["target_text"])
        embedding = executor_model.get_input_embeddings()
        expected_prompt = embedding(prompt_ids)
        expected_target = embedding(target_ids)
        target_start = prompt_ids.numel() + self.system.prefix_tokens

        torch.testing.assert_close(actual[: prompt_ids.numel()], expected_prompt)
        torch.testing.assert_close(
            actual[target_start : target_start + target_ids.numel()],
            expected_target,
        )

    def test_language_model_loss_reaches_csi_parameters(self) -> None:
        self.system.set_inner_adapters_trainable(False)
        result = self.system.training_loss(
            [task["prompt"] for task in self.tasks],
            self.tasks,
            ["def add(a, b): return a + b", "def sub(a, b): return a - b"],
            intermediate_loss_weight=0.2,
        )
        self.assertTrue(torch.isfinite(result["loss"]).item())
        self.assertTrue(torch.isfinite(result["final_ce_loss"]).item())
        self.assertGreater(float(result["intermediate_loss"].item()), 0.0)
        result["loss"].backward()

        self.assertIsNotNone(self.system.projectors[0].weight.grad)
        self.assertIsNotNone(self.system.csi.intra.conv_blocks[0].depthwise.weight.grad)
        self.assertIsNotNone(self.system.csi.inter.projection.weight.grad)
        self.assertIsNotNone(self.system.prefix_mapper[-1].weight.grad)
        self.assertIsNone(self.system.inner_adapters[0].proj2.weight.grad)
        self.assertFalse(any("planner.model" in key for key in self.system.state_dict()))

    def test_manual_kv_generation_returns_text(self) -> None:
        executor_model = self.system.executor.model
        executor_model.explicit_embedding_inputs.clear()
        text = self.system.generate_code(
            self.tasks[0]["prompt"],
            self.tasks[0],
            max_new_tokens=4,
            do_sample=False,
        )
        self.assertIsInstance(text, str)
        initial_input = executor_model.explicit_embedding_inputs[0][0]
        prompt_ids = self.system.executor.build_generation_prompt_ids(
            self.tasks[0]["prompt"],
            self.tasks[0],
        )
        expected_prompt = executor_model.get_input_embeddings()(prompt_ids)
        torch.testing.assert_close(initial_input[: prompt_ids.numel()], expected_prompt)

    def test_generation_result_reports_tokens_and_finish_reason(self) -> None:
        result = self.system.generate_code(
            self.tasks[0]["prompt"],
            self.tasks[0],
            max_new_tokens=4,
            do_sample=False,
            return_result=True,
            profile_latency=True,
        )
        self.assertIsInstance(result, GenerationResult)
        assert isinstance(result, GenerationResult)
        self.assertEqual(result.generated_tokens, 4)
        self.assertEqual(result.finish_reason, "length")
        self.assertGreater(result.solver_prompt_tokens, 0)
        self.assertEqual(result.csi_prefix_tokens, self.system.prefix_tokens)
        self.assertEqual(
            set(result.timing_seconds),
            {"collaboration", "prefix_preparation", "solver_prefill", "solver_decode"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
