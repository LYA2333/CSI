"""Sequential RecursiveMAS code-generation adapter.

This reuses the official RecursiveMAS inference stages instead of reimplementing
their latent recursion. It targets the current public repository layout:

    RecursiveMAS/
      inference/system_loader.py
      inference/inference_utils/inference_mas.py

Important: the official stage helpers load and release the relevant LLM inside
each stage. Consequently ``generate`` below includes that lifecycle. This is a
functional adapter; use the same lifecycle in both systems, or refactor both to
persistent models, before drawing latency conclusions.
"""

from __future__ import annotations

from argparse import Namespace
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


class RecursiveMASCodegenRunner:
    def __init__(
        self,
        repo_root: str,
        style: str = "sequential_light",
        dataset: str = "mbppplus",
        device: str = "cuda:0",
        dtype: str = "auto",
        outer_dtype: str = "auto",
        recursive_rounds: int = 3,
        latent_steps: int = 48,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = -1,
        do_sample: bool = True,
        seed: int = 42,
        trust_remote_code: bool = True,
        default_task_type: str = "function",
    ) -> None:
        if style not in {"sequential_light", "sequential_scaled"}:
            raise ValueError("This example supports sequential_light/scaled only")
        if recursive_rounds < 1 or latent_steps < 1 or max_new_tokens < 1:
            raise ValueError("rounds, latent_steps and max_new_tokens must be positive")

        root = Path(repo_root).expanduser().resolve()
        inference_dir = root / "inference"
        if not (inference_dir / "system_loader.py").is_file():
            raise FileNotFoundError(
                f"Cannot find official RecursiveMAS inference code under {inference_dir}"
            )
        if str(inference_dir) not in sys.path:
            sys.path.insert(0, str(inference_dir))

        # Imports are intentionally delayed until the official repo is on sys.path.
        from system_loader import resolve_mas_paths
        from inference_utils import inference_mas

        self.rm = inference_mas
        self.device = torch.device(device)
        self.model_dtype = inference_mas.resolve_dtype(dtype)
        self.outer_dtype = inference_mas.resolve_dtype(outer_dtype)
        if self.model_dtype is None or self.outer_dtype is None:
            raise ValueError("Unsupported dtype or outer_dtype")

        print(f"Resolving official RecursiveMAS checkpoints: {style}/{dataset}")
        paths = resolve_mas_paths(style=style, dataset=dataset)
        if paths.family != "sequential":
            raise ValueError(f"Expected sequential family, got {paths.family}")

        self.planner_model = str(paths.repo_paths["planner"])
        self.critic_model = str(paths.repo_paths["critic"])
        self.solver_model = str(paths.repo_paths["solver"])
        self.planner_inner = str(paths.inner_adapter_paths["planner"])
        self.critic_inner = str(paths.inner_adapter_paths["critic"])
        self.solver_inner = str(paths.inner_adapter_paths["solver"])
        self.outer_12 = str(paths.outer_adapter_paths["outer_12"])
        self.outer_23 = str(paths.outer_adapter_paths["outer_23"])
        self.outer_31 = str(paths.outer_adapter_paths["outer_31"])

        self.recursive_rounds = recursive_rounds
        self.latent_steps = latent_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.seed = seed
        self.trust_remote_code = trust_remote_code
        self.default_task_type = default_task_type
        self.args = Namespace(
            mas_shape="chain",
            solver_pre_question=0,
            choice_old_prompt=0,
        )

        # These globals are used by the official generation-kwargs helper.
        inference_mas._GEN_TOP_K = top_k if top_k >= 0 else None
        inference_mas._GEN_MIN_P = None
        inference_mas._GEN_REPETITION_PENALTY = 1.0

    def _seed_for_task(self, task_id: str) -> None:
        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        task_seed = self.seed + int.from_bytes(digest[:4], "big")
        torch.manual_seed(task_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task_seed)

    def generate(
        self,
        prompt: str,
        task_id: str,
        task: Mapping[str, Any],
    ) -> str:
        """Run Planner -> Critic -> Solver latent recursion for one task."""
        self._seed_for_task(task_id)
        rm = self.rm
        questions = [prompt]
        task_type = str(task.get("task_type", self.default_task_type))
        entry_point = task.get("entry_point") or task.get("fn_name")
        task_types = [task_type]
        fn_names = [str(entry_point) if entry_point else None]

        feedback_to_planner = None
        final_critic_latents = None
        for round_index in range(self.recursive_rounds):
            if round_index == 0:
                planner_latents = rm.run_planner_latent_stage(
                    model_name_or_path=self.planner_model,
                    questions=questions,
                    agent1_inner_aligner_path=self.planner_inner,
                    outer_12_path=self.outer_12,
                    outer_12_type="outer_ln_res_adapter",
                    latent_steps=self.latent_steps,
                    batch_size=1,
                    device=self.device,
                    model_dtype=self.model_dtype,
                    outer_dtype=self.outer_dtype,
                    trust_remote_code=self.trust_remote_code,
                    inner_adapter_type_fallback="ln_res_adapter",
                    enable_thinking=False,
                    task_types=task_types,
                    fn_names=fn_names,
                )
            else:
                if feedback_to_planner is None:
                    raise RuntimeError("Missing solver feedback for recursive planner round")
                planner_latents = rm.run_planner_feedback_latent_stage(
                    model_name_or_path=self.planner_model,
                    questions=questions,
                    feedback_latents=feedback_to_planner,
                    agent1_inner_aligner_path=self.planner_inner,
                    outer_12_path=self.outer_12,
                    outer_12_type="outer_ln_res_adapter",
                    latent_steps=self.latent_steps,
                    batch_size=1,
                    device=self.device,
                    model_dtype=self.model_dtype,
                    outer_dtype=self.outer_dtype,
                    trust_remote_code=self.trust_remote_code,
                    inner_adapter_type_fallback="ln_res_adapter",
                    enable_thinking=False,
                    task_types=task_types,
                    fn_names=fn_names,
                )

            final_critic_latents = rm.run_refiner_latent_stage(
                model_name_or_path=self.critic_model,
                questions=questions,
                planner_latents=planner_latents,
                agent2_inner_aligner_path=self.critic_inner,
                outer_23_path=self.outer_23,
                outer_23_type="outer_ln_res_adapter",
                latent_steps=self.latent_steps,
                batch_size=1,
                device=self.device,
                model_dtype=self.model_dtype,
                outer_dtype=self.outer_dtype,
                trust_remote_code=self.trust_remote_code,
                inner_adapter_type_fallback="ln_res_adapter",
                enable_thinking=False,
                task_types=task_types,
                fn_names=fn_names,
            )

            if round_index < self.recursive_rounds - 1:
                feedback_to_planner = rm.run_solver_feedback_latent_stage(
                    model_name_or_path=self.solver_model,
                    questions=questions,
                    refiner_latents=final_critic_latents,
                    agent3_inner_aligner_path=self.solver_inner,
                    outer_31_path=self.outer_31,
                    outer_31_type="outer_ln_res_adapter",
                    latent_steps=self.latent_steps,
                    batch_size=1,
                    device=self.device,
                    model_dtype=self.model_dtype,
                    outer_dtype=self.outer_dtype,
                    trust_remote_code=self.trust_remote_code,
                    inner_adapter_type_fallback="ln_res_adapter",
                    enable_thinking=False,
                    args=self.args,
                    task_types=task_types,
                    fn_names=fn_names,
                )

        if final_critic_latents is None:
            raise RuntimeError("RecursiveMAS did not produce critic-to-solver latents")
        outputs = rm.run_solver_latent_stage(
            model_name_or_path=self.solver_model,
            questions=questions,
            refiner_latents=final_critic_latents,
            args=self.args,
            batch_size=1,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            device=self.device,
            dtype=self.model_dtype,
            trust_remote_code=self.trust_remote_code,
            enable_thinking=False,
            task_types=task_types,
            fn_names=fn_names,
        )
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one solver output, got {len(outputs)}")
        return outputs[0]


def build_runner(**kwargs) -> RecursiveMASCodegenRunner:
    return RecursiveMASCodegenRunner(**kwargs)
