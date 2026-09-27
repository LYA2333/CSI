"""End-to-end CSI integration for three frozen Qwen-family agents.

The base LLMs are intentionally kept outside this module's registered PyTorch
submodules. Therefore ``CSIQwenSystem.state_dict()`` contains only the small
Inner Adapters, CSI projectors, Conv1D/GCN blocks, and Executor prefix mapper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from conv_graph_mas_demo import CSIGCN


AGENT_ROLES = ("planner", "critic", "executor")
FEATURE_NAMES = ("task", "role", "experience", "prompt")
INNER_ADAPTER_TYPE = "pre_post_ln_residual_mlp_v2"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int
    solver_prompt_tokens: int
    csi_prefix_tokens: int
    finish_reason: str
    timing_seconds: dict[str, float]

ROLE_DESCRIPTIONS = {
    "planner": (
        "You are the Planner. Analyze the programming problem, decompose it, "
        "and identify the algorithm, invariants, edge cases, and complexity."
    ),
    "critic": (
        "You are the Critic. Inspect a proposed solution conceptually, look for "
        "incorrect assumptions, missing edge cases, and implementation hazards."
    ),
    "executor": (
        "You are the Executor. Turn the task and collaborative guidance into "
        "correct, executable Python code that follows the required signature."
    ),
}

ROLE_INSTRUCTIONS = {
    "planner": "Produce a precise implementation plan for the task.",
    "critic": "Identify likely failure cases and requirements that must be verified.",
    "executor": "Implement the final solution. Return only Python source code.",
}

MATH_ROLE_DESCRIPTIONS = {
    "planner": (
        "You are the Planner. Analyze the mathematical problem, identify the "
        "relevant concepts, and construct a rigorous solution plan."
    ),
    "critic": (
        "You are the Critic. Check the mathematical reasoning for invalid "
        "assumptions, missing cases, and algebraic or logical errors."
    ),
    "executor": (
        "You are the Executor. Synthesize the collaborative guidance into a "
        "clear derivation and a correct final answer."
    ),
}

MATH_ROLE_INSTRUCTIONS = {
    "planner": "Produce a precise step-by-step plan for solving the problem.",
    "critic": "Identify likely reasoning errors and verify all necessary cases.",
    "executor": (
        "Write the complete mathematical solution and put the final answer "
        "inside \\boxed{}, for example \\boxed{1}."
    ),
}

CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Return only complete executable Python "
    "source code. Do not include Markdown fences or explanations."
)

MATH_SYSTEM_PROMPT = (
    "You are an expert mathematical problem solver. Give a rigorous, concise "
    "step-by-step solution and clearly state the final answer."
)


def resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[normalized]


def _complete_boxed_end(text: str) -> int | None:
    """Return the end offset of the first complete, non-empty ``\\boxed{...}``."""
    start = 0
    command = r"\boxed"
    while True:
        command_start = text.find(command, start)
        if command_start < 0:
            return None
        brace_start = command_start + len(command)
        while brace_start < len(text) and text[brace_start].isspace():
            brace_start += 1
        if brace_start >= len(text) or text[brace_start] != "{":
            start = command_start + len(command)
            continue

        depth = 0
        for index in range(brace_start, len(text)):
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    if text[brace_start + 1 : index].strip():
                        return index + 1
                    break
        start = command_start + len(command)


def _has_repeated_token_suffix(
    token_ids: Sequence[int],
    *,
    min_repeated_tokens: int,
    max_pattern_tokens: int,
) -> bool:
    """Detect a short token pattern repeated over a long generated suffix."""
    if min_repeated_tokens < 1 or max_pattern_tokens < 1:
        return False
    for pattern_size in range(1, max_pattern_tokens + 1):
        copies = max(4, (min_repeated_tokens + pattern_size - 1) // pattern_size)
        repeated_length = pattern_size * copies
        if len(token_ids) < repeated_length:
            continue
        pattern = list(token_ids[-pattern_size:])
        if list(token_ids[-repeated_length:]) == pattern * copies:
            return True
    return False


def _role_value(value: Any, role: str, default: str) -> str:
    if isinstance(value, Mapping):
        selected = value.get(role, default)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        selected = "\n".join(str(item) for item in value)
    elif value is None or value == "":
        selected = default
    else:
        selected = value
    return str(selected)


def build_agent_feature_texts(
    prompt: str,
    task: Mapping[str, Any],
    role: str,
) -> list[str]:
    """Build task/role/experience/prompt feature text for one agent."""
    if role not in AGENT_ROLES:
        raise ValueError(f"Unknown role: {role}")

    is_math = str(task.get("task_type", "code")).lower() == "math"
    default_descriptions = MATH_ROLE_DESCRIPTIONS if is_math else ROLE_DESCRIPTIONS
    default_instructions = MATH_ROLE_INSTRUCTIONS if is_math else ROLE_INSTRUCTIONS
    role_description = _role_value(task.get("roles"), role, default_descriptions[role])
    experience = _role_value(
        task.get("experience"), role, "No additional retrieved experience."
    )
    role_instruction = _role_value(
        task.get("agent_prompts"), role, default_instructions[role]
    )
    task_label = "Mathematical reasoning task" if is_math else "Programming task"
    task_text = f"{task_label}:\n{prompt}"
    role_text = f"Agent role:\n{role_description}"
    experience_text = f"Relevant experience:\n{experience}"
    prompt_text = f"Agent instruction:\n{role_instruction}\n\n{task_text}"
    return [task_text, role_text, experience_text, prompt_text]


def generation_prompt(prompt: str, task: Mapping[str, Any]) -> tuple[str, str]:
    if str(task.get("task_type", "code")).lower() == "math":
        user_prompt = (
            f"{prompt}\n\nSolve the problem step by step and put the final answer "
            "inside \\boxed{}, for example \\boxed{1}."
        )
        return MATH_SYSTEM_PROMPT, user_prompt

    entry_point = task.get("entry_point") or task.get("fn_name")
    suffix = (
        f"\n\nThe required entry point is `{entry_point}`."
        if entry_point
        else ""
    )
    user_prompt = f"{prompt}{suffix}\n\nReturn only the complete Python source code."
    return CODE_SYSTEM_PROMPT, user_prompt


def _required_training_text(task: Mapping[str, Any], key: str, role: str) -> str:
    value = task.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        task_id = task.get("task_id", "<unknown>")
        raise ValueError(
            f"Task {task_id!r} needs non-empty {key!r} for {role} Inner Adapter warm-up"
        )
    return text


def role_supervision_example(
    prompt: str,
    task: Mapping[str, Any],
    role: str,
) -> tuple[str, str, str]:
    """Build role-specific teacher-forcing text used only during inner warm-up."""
    if role not in AGENT_ROLES:
        raise ValueError(f"Unknown role: {role}")

    is_math = str(task.get("task_type", "code")).lower() == "math"
    descriptions = MATH_ROLE_DESCRIPTIONS if is_math else ROLE_DESCRIPTIONS
    plan = _required_training_text(task, "plan", role)
    refined_plan = _required_training_text(task, "refined_plan", role)
    answer = _required_training_text(task, "target_text", role)
    task_label = "Problem" if is_math else "Programming task"

    if role == "planner":
        user_prompt = (
            f"{task_label}:\n{prompt}\n\n"
            "Produce the detailed plan that should guide the other agents."
        )
        target = plan
    elif role == "critic":
        user_prompt = (
            f"{task_label}:\n{prompt}\n\nPlanner proposal:\n{plan}\n\n"
            "Critique the proposal and produce a corrected, refined plan."
        )
        target = refined_plan
    else:
        user_prompt = (
            f"{task_label}:\n{prompt}\n\nPlan:\n{plan}\n\n"
            f"Refined plan:\n{refined_plan}\n\nProduce the final answer."
        )
        target = answer

    return descriptions[role], user_prompt, target


class InnerRecursiveAdapter(nn.Module):
    """Pre/post-LN residual MLP for stable role-level warm-up."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.proj2(F.gelu(self.proj1(self.pre_norm(hidden_states))))
        return self.post_norm(hidden_states + residual)


class FrozenQwenAgent:
    """A frozen Qwen causal LM used as a contextual feature encoder/decoder."""

    def __init__(
        self,
        model_path: str,
        *,
        role: str,
        device: str,
        model_dtype: str = "bfloat16",
        max_feature_tokens: int = 512,
        trust_remote_code: bool = False,
        attn_implementation: str = "",
    ) -> None:
        if not model_path.strip():
            raise ValueError(f"{role}_model_path is empty")
        if max_feature_tokens < 1:
            raise ValueError("max_feature_tokens must be positive")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for real Qwen agents; install the "
                "RecursiveMAS/CSI environment first"
            ) from exc

        self.role = role
        self.model_path = model_path
        self.device = torch.device(device)
        self.max_feature_tokens = max_feature_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(f"Tokenizer for {role} has neither pad nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
            "dtype": model_dtype if model_dtype == "auto" else resolve_torch_dtype(model_dtype),
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except TypeError as exc:
            # Transformers 4.x used torch_dtype; Transformers 5.x prefers dtype.
            if "dtype" not in str(exc):
                raise
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.get_input_embeddings().embedding_dim)

    def _last_hidden_state(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        backbone = getattr(self.model, "model", None)
        if backbone is not None:
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            return outputs.last_hidden_state

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if not outputs.hidden_states:
            raise RuntimeError(f"{self.role} model did not return hidden states")
        return outputs.hidden_states[-1]

    def encode_feature_texts(
        self,
        batch_feature_texts: Sequence[Sequence[str]],
        *,
        latent_tokens: int,
    ) -> Tensor:
        """Return fixed-length features with shape [B, K, S, H_role]."""
        if latent_tokens < 1:
            raise ValueError("latent_tokens must be positive")
        if not batch_feature_texts:
            raise ValueError("batch_feature_texts cannot be empty")

        feature_count = len(batch_feature_texts[0])
        if feature_count != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} feature texts")
        if any(len(row) != feature_count for row in batch_feature_texts):
            raise ValueError("Every sample must have the same feature count")

        flat_texts = [text for row in batch_feature_texts for text in row]
        encoded = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.max_feature_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # no_grad freezes expensive feature extraction while still producing
        # ordinary tensors that trainable projection layers may consume.
        with torch.no_grad():
            hidden = self._last_hidden_state(input_ids, attention_mask)

        pooled_rows: list[Tensor] = []
        for row_index in range(hidden.shape[0]):
            valid = hidden[row_index][attention_mask[row_index].bool()]
            if valid.shape[0] == 0:
                raise RuntimeError("Tokenizer produced an empty feature sequence")
            pooled = F.adaptive_avg_pool1d(
                valid.transpose(0, 1).unsqueeze(0),
                latent_tokens,
            ).squeeze(0).transpose(0, 1)
            pooled_rows.append(pooled)

        batch_size = len(batch_feature_texts)
        return torch.stack(pooled_rows).reshape(
            batch_size,
            feature_count,
            latent_tokens,
            self.hidden_size,
        )

    def build_generation_prompt_ids(
        self,
        prompt: str,
        task: Mapping[str, Any],
    ) -> Tensor:
        system_prompt, user_prompt = generation_prompt(prompt, task)
        return self.build_chat_prompt_ids(system_prompt, user_prompt)

    def build_chat_prompt_ids(self, system_prompt: str, user_prompt: str) -> Tensor:
        """Tokenize a system/user exchange ending at the assistant generation slot."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }
            try:
                result = self.tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **template_kwargs,
                )
            except TypeError:
                result = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            if isinstance(result, Mapping):
                result = result["input_ids"]
            if not isinstance(result, Tensor):
                result = torch.tensor(result, dtype=torch.long)
            return result.reshape(-1).to(self.device)

        fallback = f"{system_prompt}\n\n{user_prompt}\n\nAssistant:\n"
        return self.tokenizer(
            fallback,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"].reshape(-1).to(self.device)

    def encode_supervision_pair(
        self,
        system_prompt: str,
        user_prompt: str,
        target_text: str,
    ) -> tuple[Tensor, Tensor]:
        """Return hidden predictors and their next-target embeddings for warm-up."""
        prompt_ids = self.build_chat_prompt_ids(system_prompt, user_prompt)
        target_ids = self.tokenize_target(target_text)

        # Bound the no-grad teacher-forcing pass while retaining at least one
        # prompt token immediately before the supervised target sequence.
        max_length = max(self.max_feature_tokens, 2)
        if target_ids.numel() >= max_length:
            target_ids = target_ids[: max_length - 1]
        prompt_budget = max(max_length - target_ids.numel(), 1)
        prompt_ids = prompt_ids[-prompt_budget:]
        full_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)

        with torch.no_grad():
            hidden = self._last_hidden_state(full_ids, attention_mask)[0]
            target_embeddings = self.model.get_input_embeddings()(target_ids)

        first_predictor = prompt_ids.numel() - 1
        predictors = hidden[
            first_predictor : first_predictor + target_ids.numel()
        ]
        if predictors.shape != target_embeddings.shape:
            raise RuntimeError(
                f"{self.role} warm-up alignment shape mismatch: "
                f"{tuple(predictors.shape)} vs {tuple(target_embeddings.shape)}"
            )
        return predictors, target_embeddings

    def tokenize_target(self, code: str) -> Tensor:
        result = self.tokenizer(
            code,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].reshape(-1)
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, Sequence):
            eos_token_id = eos_token_id[0]
        if eos_token_id is not None and (
            result.numel() == 0 or result[-1].item() != int(eos_token_id)
        ):
            result = torch.cat([result, torch.tensor([eos_token_id])])
        if result.numel() == 0:
            raise ValueError("Target code tokenized to an empty sequence")
        return result.to(self.device)


class CSIQwenSystem(nn.Module):
    """Three frozen Qwen agents connected by warmed adapters and CSI Conv1D+GCN."""

    def __init__(
        self,
        planner: FrozenQwenAgent,
        critic: FrozenQwenAgent,
        executor: FrozenQwenAgent,
        *,
        d_model: int = 512,
        latent_tokens: int = 32,
        prefix_tokens: int = 8,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        max_steps: int = 4,
        tolerance: float = 1e-3,
        csi_device: str = "",
        csi_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if d_model < 1 or latent_tokens < 1 or prefix_tokens < 1 or max_steps < 1:
            raise ValueError("CSI dimensions and max_steps must be positive")

        # These wrappers are plain Python objects, so their frozen LLM weights
        # are deliberately excluded from this module's state_dict/optimizer.
        self.planner = planner
        self.critic = critic
        self.executor = executor
        self.agents = (planner, critic, executor)
        self.d_model = d_model
        self.latent_tokens = latent_tokens
        self.prefix_tokens = prefix_tokens
        # The learned CSI tokens refine the fully encoded task prompt.  Keep
        # this semantic in checkpoints because changing the order also changes
        # every token's position and therefore is not weight-compatible.
        self.prefix_position = "after_prompt"
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.csi_device = torch.device(csi_device or str(executor.device))
        self.csi_dtype = resolve_torch_dtype(csi_dtype)

        self.inner_adapters = nn.ModuleList(
            InnerRecursiveAdapter(agent.hidden_size) for agent in self.agents
        )
        self.projectors = nn.ModuleList(
            nn.Linear(agent.hidden_size, d_model) for agent in self.agents
        )
        self.csi = CSIGCN(
            d_model=d_model,
            num_static_features=len(FEATURE_NAMES),
            kernel_size=kernel_size,
            dilations=dilations,
        )
        executor_hidden = executor.hidden_size
        self.prefix_mapper = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, prefix_tokens * executor_hidden),
        )
        self.prefix_norm = nn.LayerNorm(executor_hidden, elementwise_affine=False)
        embedding_sample = executor.model.get_input_embeddings().weight.detach()[:4096]
        initial_prefix_scale = float(embedding_sample.float().std().clamp_min(1e-3).item())
        self.prefix_scale = nn.Parameter(torch.tensor(initial_prefix_scale))

        # Row = receiver, column = sender:
        # Executor -> Planner, Planner -> Critic, Critic -> Executor.
        self.register_buffer(
            "collaboration_graph",
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        )
        self.to(device=self.csi_device, dtype=self.csi_dtype)

    def set_inner_adapters_trainable(
        self,
        enabled: bool,
        *,
        role: str | None = None,
    ) -> None:
        """Enable one role adapter, all adapters, or freeze every adapter."""
        if role is not None and role not in AGENT_ROLES:
            raise ValueError(f"Unknown role: {role}")

        self.inner_adapters.train(enabled)
        for adapter_role, adapter in zip(AGENT_ROLES, self.inner_adapters):
            role_enabled = enabled and (role is None or adapter_role == role)
            adapter.train(role_enabled)
            for parameter in adapter.parameters():
                parameter.requires_grad_(role_enabled)

    def inner_role_loss(
        self,
        tasks: Sequence[Mapping[str, Any]],
        role: str,
        *,
        cosine_weight: float = 1.0,
        mse_weight: float = 0.0,
    ) -> dict[str, Tensor]:
        """Train one role adapter against its own supervision target."""
        if not tasks:
            raise ValueError("Inner warm-up needs a non-empty task batch")
        if role not in AGENT_ROLES:
            raise ValueError(f"Unknown role: {role}")
        if cosine_weight < 0 or mse_weight < 0 or cosine_weight + mse_weight <= 0:
            raise ValueError("Inner warm-up loss weights must be non-negative and non-zero")

        role_index = AGENT_ROLES.index(role)
        agent = self.agents[role_index]
        adapter = self.inner_adapters[role_index]
        sample_cosine: list[Tensor] = []
        sample_mse: list[Tensor] = []
        for task in tasks:
            prompt = str(task.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("Inner warm-up task is missing a non-empty prompt")
            system_prompt, user_prompt, target = role_supervision_example(
                prompt,
                task,
                role,
            )
            hidden, target_embeddings = agent.encode_supervision_pair(
                system_prompt,
                user_prompt,
                target,
            )
            hidden = hidden.to(device=self.csi_device, dtype=self.csi_dtype)
            target_embeddings = target_embeddings.to(
                device=self.csi_device,
                dtype=self.csi_dtype,
            )
            predicted = adapter(hidden)
            sample_cosine.append(
                (1.0 - F.cosine_similarity(
                    predicted.float(),
                    target_embeddings.float(),
                    dim=-1,
                )).mean()
            )
            sample_mse.append(
                (predicted.float() - target_embeddings.float()).square().mean()
            )

        cosine = torch.stack(sample_cosine).mean()
        mse = torch.stack(sample_mse).mean()
        return {
            "loss": cosine_weight * cosine + mse_weight * mse,
            "cosine_loss": cosine.detach(),
            "mse_loss": mse.detach(),
        }

    def inner_warmup_loss(
        self,
        tasks: Sequence[Mapping[str, Any]],
        *,
        cosine_weight: float = 1.0,
        mse_weight: float = 0.0,
    ) -> dict[str, Tensor]:
        """Align each role adapter with plan/refined-plan/answer token embeddings."""
        role_losses: dict[str, Tensor] = {}
        role_cosine: dict[str, Tensor] = {}
        role_mse: dict[str, Tensor] = {}
        for role in AGENT_ROLES:
            result = self.inner_role_loss(
                tasks,
                role,
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
            )
            role_losses[role] = result["loss"]
            role_cosine[role] = result["cosine_loss"]
            role_mse[role] = result["mse_loss"]

        total = torch.stack(list(role_losses.values())).mean()
        return {
            "loss": total,
            "planner_loss": role_losses["planner"].detach(),
            "critic_loss": role_losses["critic"].detach(),
            "executor_loss": role_losses["executor"].detach(),
            "cosine_loss": torch.stack(list(role_cosine.values())).mean().detach(),
            "mse_loss": torch.stack(list(role_mse.values())).mean().detach(),
        }

    def encode_static_features(
        self,
        prompts: Sequence[str],
        tasks: Sequence[Mapping[str, Any]],
    ) -> Tensor:
        if len(prompts) != len(tasks) or not prompts:
            raise ValueError("prompts and tasks must have the same non-zero length")

        projected_agents: list[Tensor] = []
        for role, agent, inner_adapter, projector in zip(
            AGENT_ROLES,
            self.agents,
            self.inner_adapters,
            self.projectors,
        ):
            feature_texts = [
                build_agent_feature_texts(prompt, task, role)
                for prompt, task in zip(prompts, tasks)
            ]
            hidden = agent.encode_feature_texts(
                feature_texts,
                latent_tokens=self.latent_tokens,
            )
            hidden = hidden.to(device=self.csi_device, dtype=self.csi_dtype)
            projected_agents.append(projector(inner_adapter(hidden)))

        # [B, N=3, K=4, S, D]
        return torch.stack(projected_agents, dim=1)

    def collaborative_messages(
        self,
        prompts: Sequence[str],
        tasks: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor | int | list[Tensor]]:
        static_features = self.encode_static_features(prompts, tasks)
        return self.csi(
            static_features,
            max_steps=self.max_steps,
            tolerance=self.tolerance,
            adjacency=self.collaboration_graph,
        )

    def executor_prefix(self, messages: Tensor) -> Tensor:
        executor_message = messages[:, 2]
        prefix = self.prefix_mapper(executor_message)
        prefix = prefix.reshape(
            messages.shape[0],
            self.prefix_tokens,
            self.executor.hidden_size,
        )
        return self.prefix_scale * self.prefix_norm(prefix)

    def training_loss(
        self,
        prompts: Sequence[str],
        tasks: Sequence[Mapping[str, Any]],
        target_codes: Sequence[str],
        *,
        intermediate_loss_weight: float = 0.0,
    ) -> dict[str, Tensor | int]:
        if len(prompts) != len(target_codes):
            raise ValueError("Need one target code string for every prompt")
        if intermediate_loss_weight < 0:
            raise ValueError("intermediate_loss_weight must be non-negative")

        collaboration = self.collaborative_messages(prompts, tasks)
        prefix = self.executor_prefix(collaboration["messages"])
        embedding = self.executor.model.get_input_embeddings()
        executor_dtype = embedding.weight.dtype
        executor_device = self.executor.device
        prefix = prefix.to(device=executor_device, dtype=executor_dtype)

        sequences: list[Tensor] = []
        labels: list[Tensor] = []
        masks: list[Tensor] = []
        target_summaries: list[Tensor] = []
        for sample_index, (prompt, task, target_code) in enumerate(
            zip(prompts, tasks, target_codes)
        ):
            prompt_ids = self.executor.build_generation_prompt_ids(prompt, task)
            target_ids = self.executor.tokenize_target(target_code)
            with torch.no_grad():
                prompt_embeddings = embedding(prompt_ids)
                target_embeddings = embedding(target_ids)
                target_summaries.append(
                    target_embeddings.mean(dim=0).to(
                        device=self.csi_device,
                        dtype=self.csi_dtype,
                    )
                )

            sample_embeddings = torch.cat(
                [prompt_embeddings, prefix[sample_index], target_embeddings],
                dim=0,
            )
            ignored = self.prefix_tokens + prompt_ids.numel()
            sample_labels = torch.cat(
                [
                    torch.full(
                        (ignored,),
                        -100,
                        dtype=torch.long,
                        device=executor_device,
                    ),
                    target_ids,
                ]
            )
            sequences.append(sample_embeddings)
            labels.append(sample_labels)
            masks.append(
                torch.ones(sample_embeddings.shape[0], dtype=torch.long, device=executor_device)
            )

        max_length = max(sequence.shape[0] for sequence in sequences)
        padded_embeddings = torch.stack(
            [F.pad(sequence, (0, 0, 0, max_length - sequence.shape[0])) for sequence in sequences]
        )
        padded_labels = torch.stack(
            [F.pad(label, (0, max_length - label.shape[0]), value=-100) for label in labels]
        )
        attention_mask = torch.stack(
            [F.pad(mask, (0, max_length - mask.shape[0]), value=0) for mask in masks]
        )

        outputs = self.executor.model(
            inputs_embeds=padded_embeddings,
            attention_mask=attention_mask,
            labels=padded_labels,
            use_cache=False,
            return_dict=True,
        )

        message_history = collaboration["message_history"]
        if not isinstance(message_history, list) or not message_history:
            raise RuntimeError("CSI must return a non-empty message_history")
        intermediate_losses: list[Tensor] = []
        if intermediate_loss_weight > 0:
            target_summary = torch.stack(target_summaries).float()
            for round_messages in message_history[:-1]:
                round_prefix = self.executor_prefix(round_messages)
                round_summary = round_prefix.mean(dim=1).float()
                intermediate_losses.append(
                    (1.0 - F.cosine_similarity(
                        round_summary,
                        target_summary,
                        dim=-1,
                    )).mean()
                )
        if intermediate_losses:
            intermediate_loss = torch.stack(intermediate_losses).mean()
        else:
            intermediate_loss = collaboration["messages"].new_zeros(())

        total_loss = outputs.loss + intermediate_loss_weight * intermediate_loss.to(
            device=executor_device,
            dtype=outputs.loss.dtype,
        )
        return {
            "loss": total_loss,
            "final_ce_loss": outputs.loss.detach(),
            "intermediate_loss": intermediate_loss.detach(),
            "steps": collaboration["steps"],
            "relative_change": collaboration["relative_change"],
        }

    @staticmethod
    def _sample_token(
        logits: Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tensor:
        if not do_sample or temperature <= 0:
            return logits.argmax(dim=-1)

        filtered = logits / temperature
        if top_k > 0 and top_k < filtered.shape[-1]:
            threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
            filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            sampled_sorted = torch.multinomial(torch.softmax(sorted_logits, dim=-1), 1)
            return sorted_indices.gather(-1, sampled_sorted).squeeze(-1)
        return torch.multinomial(torch.softmax(filtered, dim=-1), 1).squeeze(-1)

    @torch.no_grad()
    def generate_code(
        self,
        prompt: str,
        task: Mapping[str, Any],
        *,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 0,
        stop_on_boxed: bool = False,
        stop_on_repetition: bool = False,
        repetition_min_tokens: int = 64,
        repetition_max_pattern_tokens: int = 16,
        return_result: bool = False,
        profile_latency: bool = False,
    ) -> str | GenerationResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_min_tokens < 1:
            raise ValueError("repetition_min_tokens must be positive")
        if repetition_max_pattern_tokens < 1:
            raise ValueError("repetition_max_pattern_tokens must be positive")

        self.eval()
        timing_seconds: dict[str, float] = {}
        if profile_latency:
            self._synchronize_generation_devices()
            stage_started = time.perf_counter()
        else:
            stage_started = 0.0

        collaboration = self.collaborative_messages([prompt], [task])
        if profile_latency:
            self._synchronize_generation_devices()
            now = time.perf_counter()
            timing_seconds["collaboration"] = now - stage_started
            stage_started = now

        prefix = self.executor_prefix(collaboration["messages"])

        prompt_ids = self.executor.build_generation_prompt_ids(prompt, task)
        embedding = self.executor.model.get_input_embeddings()
        prefix = prefix.to(
            device=self.executor.device,
            dtype=embedding.weight.dtype,
        )
        prompt_embeddings = embedding(prompt_ids).unsqueeze(0)
        inputs_embeds = torch.cat([prompt_embeddings, prefix], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=self.executor.device,
        )
        if profile_latency:
            self._synchronize_generation_devices()
            now = time.perf_counter()
            timing_seconds["prefix_preparation"] = now - stage_started
            stage_started = now

        outputs = self.executor.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        generated: list[int] = []
        if profile_latency:
            self._synchronize_generation_devices()
            now = time.perf_counter()
            timing_seconds["solver_prefill"] = now - stage_started
            stage_started = now

        configured_eos = getattr(
            getattr(self.executor.model, "generation_config", SimpleNamespace()),
            "eos_token_id",
            None,
        )
        if configured_eos is None:
            configured_eos = self.executor.tokenizer.eos_token_id
        if isinstance(configured_eos, Sequence):
            eos_ids = {int(value) for value in configured_eos}
        elif configured_eos is None:
            eos_ids = set()
        else:
            eos_ids = {int(configured_eos)}

        stopped_text: str | None = None
        finish_reason = "length"
        for _ in range(max_new_tokens):
            next_token = self._sample_token(
                outputs.logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            token_id = int(next_token.item())
            if token_id in eos_ids:
                finish_reason = "eos"
                break
            generated.append(token_id)

            # Decoding the complete sequence on every step is unnecessarily
            # expensive. A boxed answer can only become complete when the new
            # token contains a closing brace, so perform the full check then.
            if stop_on_boxed:
                token_text = self.executor.tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                if "}" in token_text:
                    decoded = self.executor.tokenizer.decode(
                        generated, skip_special_tokens=True
                    )
                    boxed_end = _complete_boxed_end(decoded)
                    if boxed_end is not None:
                        stopped_text = decoded[:boxed_end].strip()
                        finish_reason = "boxed"
                        break

            if stop_on_repetition and _has_repeated_token_suffix(
                generated,
                min_repeated_tokens=repetition_min_tokens,
                max_pattern_tokens=repetition_max_pattern_tokens,
            ):
                stopped_text = self.executor.tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip()
                finish_reason = "repetition"
                break

            if past_key_values is None:
                raise RuntimeError("Executor model did not return a KV cache")

            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((1, 1))],
                dim=1,
            )
            outputs = self.executor.model(
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

        if profile_latency:
            self._synchronize_generation_devices()
            timing_seconds["solver_decode"] = time.perf_counter() - stage_started

        text = (
            stopped_text
            if stopped_text is not None
            else self.executor.tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()
        )
        result = GenerationResult(
            text=text,
            generated_tokens=len(generated),
            solver_prompt_tokens=int(prompt_ids.numel()),
            csi_prefix_tokens=int(prefix.shape[1]),
            finish_reason=finish_reason,
            timing_seconds=timing_seconds,
        )
        return result if return_result else result.text

    def _synchronize_generation_devices(self) -> None:
        if not torch.cuda.is_available():
            return
        devices = {agent.device for agent in self.agents}
        devices.add(self.csi_device)
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def load_trainable_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSI checkpoint not found: {path}")
        try:
            payload = torch.load(path, map_location=self.csi_device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.csi_device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model_config = payload.get("model_config")
            checkpoint_axis = (
                model_config.get("intra_conv_axis")
                if isinstance(model_config, Mapping)
                else None
            )
            if checkpoint_axis != self.csi.intra.conv_axis:
                raise RuntimeError(
                    "This checkpoint predates feature-axis Conv1D or uses a "
                    f"different convolution axis ({checkpoint_axis!r}). Retrain "
                    "CSI instead of reusing token-axis convolution weights."
                )
            checkpoint_prefix_position = (
                model_config.get("prefix_position")
                if isinstance(model_config, Mapping)
                else None
            )
            if checkpoint_prefix_position != self.prefix_position:
                raise RuntimeError(
                    "This checkpoint places the CSI prefix before the prompt "
                    "or does not record its position. Retrain CSI with the "
                    "current prompt-then-prefix input order."
                )
            checkpoint_inner_adapter_type = (
                model_config.get("inner_adapter_type")
                if isinstance(model_config, Mapping)
                else None
            )
            if checkpoint_inner_adapter_type != INNER_ADAPTER_TYPE:
                raise RuntimeError(
                    "This checkpoint uses an incompatible Inner Adapter "
                    f"architecture ({checkpoint_inner_adapter_type!r}). Retrain "
                    "with the current post-LayerNorm residual adapter."
                )
        state_dict = (
            payload.get("model_state_dict", payload)
            if isinstance(payload, dict)
            else payload
        )
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible CSI checkpoint: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return payload if isinstance(payload, dict) else {"model_state_dict": payload}

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "latent_tokens": self.latent_tokens,
            "prefix_tokens": self.prefix_tokens,
            "kernel_size": self.csi.intra.kernel_size,
            "dilations": list(self.csi.intra.dilations),
            "intra_conv_axis": self.csi.intra.conv_axis,
            "feature_receptive_field": self.csi.intra.receptive_field,
            "prefix_position": self.prefix_position,
            "max_steps": self.max_steps,
            "tolerance": self.tolerance,
            "hidden_sizes": [agent.hidden_size for agent in self.agents],
            "inner_adapter_type": INNER_ADAPTER_TYPE,
        }

    def close(self) -> None:
        for agent in self.agents:
            if hasattr(agent, "model"):
                del agent.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_qwen_system(
    *,
    planner_model_path: str = "",
    critic_model_path: str = "",
    executor_model_path: str = "",
    planner_device: str = "cuda:0",
    critic_device: str = "cuda:0",
    executor_device: str = "cuda:0",
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
    trust_remote_code: bool = False,
    attn_implementation: str = "",
    csi_device: str = "",
    checkpoint_path: str = "",
) -> CSIQwenSystem:
    """Load the three Qwen agents and construct the trainable CSI system."""
    shared_agent_kwargs = {
        "model_dtype": model_dtype,
        "max_feature_tokens": max_feature_tokens,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }
    planner = FrozenQwenAgent(
        planner_model_path,
        role="planner",
        device=planner_device,
        **shared_agent_kwargs,
    )
    critic = FrozenQwenAgent(
        critic_model_path,
        role="critic",
        device=critic_device,
        **shared_agent_kwargs,
    )
    executor = FrozenQwenAgent(
        executor_model_path,
        role="executor",
        device=executor_device,
        **shared_agent_kwargs,
    )
    system = CSIQwenSystem(
        planner,
        critic,
        executor,
        d_model=d_model,
        latent_tokens=latent_tokens,
        prefix_tokens=prefix_tokens,
        kernel_size=kernel_size,
        dilations=dilations,
        max_steps=max_steps,
        tolerance=tolerance,
        csi_device=csi_device,
        csi_dtype=csi_dtype,
    )
    if checkpoint_path:
        system.load_trainable_checkpoint(checkpoint_path)
    return system
