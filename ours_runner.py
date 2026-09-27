"""CSI-Qwen runner compatible with compare_codegen.py.

The three model path defaults are intentionally empty. Fill them in below or
pass them through ``--ours-kwargs`` when running the comparison benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from csi_qwen import CSIQwenSystem, build_qwen_system


class CSICodegenRunner:
    def __init__(
        self,
        planner_model_path: str = "",
        critic_model_path: str = "",
        executor_model_path: str = "",
        checkpoint_path: str = "",
        planner_device: str = "cuda:0",
        critic_device: str = "cuda:0",
        executor_device: str = "cuda:0",
        csi_device: str = "",
        model_dtype: str = "bfloat16",
        csi_dtype: str = "float32",
        d_model: int = 512,
        latent_tokens: int = 32,
        prefix_tokens: int = 8,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        max_steps: int = 4,
        tolerance: float = 1e-3,
        max_feature_tokens: int = 512,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 0,
        seed: int = 42,
        trust_remote_code: bool = False,
        attn_implementation: str = "",
    ) -> None:
        self.system: CSIQwenSystem = build_qwen_system(
            planner_model_path=planner_model_path,
            critic_model_path=critic_model_path,
            executor_model_path=executor_model_path,
            planner_device=planner_device,
            critic_device=critic_device,
            executor_device=executor_device,
            csi_device=csi_device,
            model_dtype=model_dtype,
            csi_dtype=csi_dtype,
            d_model=d_model,
            latent_tokens=latent_tokens,
            prefix_tokens=prefix_tokens,
            kernel_size=kernel_size,
            dilations=dilations,
            max_steps=max_steps,
            tolerance=tolerance,
            max_feature_tokens=max_feature_tokens,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            checkpoint_path=checkpoint_path,
        )
        if not checkpoint_path:
            print(
                "WARNING: no CSI checkpoint was supplied; generation uses random "
                "CSI layers and is only suitable for an integration smoke test."
            )
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed

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
        self._seed_for_task(task_id)
        return self.system.generate_code(
            prompt,
            task,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

    def close(self) -> None:
        self.system.close()


def build_runner(**kwargs: Any) -> CSICodegenRunner:
    return CSICodegenRunner(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one program with CSI-Qwen")
    # Fill these defaults on your model machine, or pass CLI values.
    parser.add_argument("--planner-model-path", default="")
    parser.add_argument("--critic-model-path", default="")
    parser.add_argument("--executor-model-path", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--planner-device", default="cuda:0")
    parser.add_argument("--critic-device", default="cuda:0")
    parser.add_argument("--executor-device", default="cuda:0")
    parser.add_argument("--csi-device", default="")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--csi-dtype", default="float32")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--latent-tokens", type=int, default=32)
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-feature-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", default="manual_test")
    parser.add_argument("--entry-point", default="")
    parser.add_argument("--prompt", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runner_kwargs = vars(args).copy()
    task_id = runner_kwargs.pop("task_id")
    entry_point = runner_kwargs.pop("entry_point")
    prompt = runner_kwargs.pop("prompt")
    runner = CSICodegenRunner(**runner_kwargs)
    try:
        task = {"task_id": task_id, "prompt": prompt, "entry_point": entry_point}
        code = runner.generate(prompt, task_id, task)
        print(json.dumps({"task_id": task_id, "code": code}, ensure_ascii=False, indent=2))
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
