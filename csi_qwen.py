"""Interleave each CSI graph update with a complete RecursiveMAS pass.

Training and inference support padded minibatches; each question retains its own messages.
Frozen backbones remain differentiable with respect to their soft inputs.
Inner pretrains each agent's local CSI prefix with CE and its detached InnerLink alignment.
Each agent projects native hidden states before convolution by default.
Legacy native-width checkpoints retain their original computation path.
Backbones are deliberately outside the adapter state_dict.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from vendor.recursivemas.model import Adapter, CrossModelAdapter
from vendor.recursivemas import prompts as rmas
from conv_graph_mas_demo import (IntraAgentConv, GraphConvolutionUpdate,
                                 parse_conv_padding, validate_conv_layout)

ROLES = ("planner", "critic", "solver")
AGENT_ROLES = ("planner", "critic", "executor")
SEQUENTIAL = "sequential"
MIXTURE = "mixture"
MIXTURE_ROLES = ("math_expert", "code_expert", "science_expert", "summarizer")
MAS_ROLES = {SEQUENTIAL: ROLES, MIXTURE: MIXTURE_ROLES}
MAS_MODEL_ROLES = {SEQUENTIAL: AGENT_ROLES, MIXTURE: MIXTURE_ROLES}
MIXTURE_MODEL_PATHS = (
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "BioMistral/BioMistral-7B",
    "Qwen/Qwen3.5-2B",
)
MIXTURE_EXPERT_SLOTS = (rmas.HIE_MATH_EXPERT_SLOT, rmas.HIE_CODE_EXPERT_SLOT,
                        rmas.HIE_SCIENCE_EXPERT_SLOT)
MIXTURE_TASK_FAMILIES = ("math", "choice", "code")
MIXTURE_TARGET_FIELDS = (
    ("hie_math_expert", "math_expert", "expert_math"),
    ("hie_code_expert", "code_expert", "expert_code"),
    ("hie_science_expert", "science_expert", "expert_science"),
    ("answer", "hie_summarizer", "summary"),
)
# Legacy order for existing checkpoints. Active order is system.features;
# new CLI training enables EXPERIENCE_FEATURES by default.
FEATURES = ("task", "role", "system", "message")
EXPERIENCE_FEATURES = ("task", "role", "system", "experience", "message")
# Shared general experience; keep examples, question identifiers and answers out.
DEFAULT_EXPERIENCE_TEXT = """General experience for mathematical reasoning:

- Identify the exact requested quantity and distinguish it from intermediate results. Preserve the given conditions, units, and the meaning of geometric labels; do not infer measurements from appearance.
- Check that the model and formulas match the actual assumptions, including timing, repeated contributions, scaling, and transformations.
- Use exact arithmetic when practical. Check decisive fraction simplifications, remainders, expansions, and coefficients by an inverse operation or an independent calculation.
- In counting, define the sample space, distinguish identical from distinct objects, and justify symmetry division. Use a small case to check the counting rule when appropriate.
- Carry domain restrictions, excluded values, strict boundaries, and special cases through to the final answer.
- Treat guessed patterns as conjectures until justified. Check consistency with known values, polynomial degrees, and previously derived identities.
- Verify the final candidate against the original conditions using substitution or another concrete check. A statement that a result is verified is not itself evidence.
- Keep reasoning focused. Repair the affected step after an error instead of repeating an entire derivation. Finish the requested result before adding optional commentary; final answers should use one complete boxed expression.
""".strip()
SLOT = "<<CSI_PREVIOUS_AGENT>>"
ARCHITECTURE = "rmas_single_pass_four_part_native_hidden_conv_single_fc_csi_v4"
TWO_FC_ARCHITECTURE = "rmas_single_pass_four_part_native_hidden_conv_csi_v3"
SEQUENCE_ARCHITECTURE = "rmas_single_pass_four_part_sequence_slots_csi_v5"
PROJECTED_ARCHITECTURE = "rmas_single_pass_four_part_pre_conv_projection_csi_v6"
EXPERIENCE_ARCHITECTURE = "rmas_single_pass_five_part_experience_csi_v8"
RECURSIVE_ARCHITECTURE = "rmas_interleaved_conv_gcn_then_pcs_final_solver_v9"
MIXTURE_ARCHITECTURE = "rmas_interleaved_conv_gcn_mixture_final_summarizer_v1"
RECURSIVE_ARCHITECTURES = (RECURSIVE_ARCHITECTURE, MIXTURE_ARCHITECTURE)
LEGACY_ARCHITECTURES = (ARCHITECTURE,TWO_FC_ARCHITECTURE,SEQUENCE_ARCHITECTURE,
                        PROJECTED_ARCHITECTURE,EXPERIENCE_ARCHITECTURE)
SINGLE_PASS = 'single_pass'
INTERLEAVED = 'interleaved'
LEGACY_RECURSIVE_MESSAGE_UPDATE = 'cached_static_parts_previous_agent_output_slots'
RECURSIVE_MESSAGE_UPDATE = 'cached_static_parts_previous_gcn_message'
MESSAGE_SOURCE_POLICIES = {'gcn': RECURSIVE_MESSAGE_UPDATE,
                           'agent_output': LEGACY_RECURSIVE_MESSAGE_UPDATE}
RECURSIVE_ROUND_FEEDBACKS = {
    RECURSIVE_MESSAGE_UPDATE: 'gcn_message_and_solver_to_planner_outerlink',
    LEGACY_RECURSIVE_MESSAGE_UPDATE: 'native_output_slots_and_solver_to_planner_outerlink',
}
MIXTURE_ROUND_FEEDBACKS = {
    RECURSIVE_MESSAGE_UPDATE: 'gcn_message_and_summarizer_to_experts_outerlinks',
    LEGACY_RECURSIVE_MESSAGE_UPDATE: 'native_output_slots_and_summarizer_to_experts_outerlinks',
}
LEGACY_RECURSIVE_SOLVER_POLICY = 'latent_feedback_without_reference_answer_final_solver_ce_only'
RECURSIVE_SOLVER_POLICY = 'teacher_forced_reference_answer_feedback_final_solver_ce_only'
RECURSIVE_SOLVER_POLICIES = (RECURSIVE_SOLVER_POLICY, LEGACY_RECURSIVE_SOLVER_POLICY)
MIXTURE_SUMMARIZER_POLICY = 'teacher_forced_reference_answer_feedback_final_summarizer_ce_only'
CONV_RESIDUAL_POLICY = 'broadcast_singleton_or_same_length_v1'
NO_CONV_RESIDUAL_POLICY = 'no_residual_v1'
CHAIN_ADJACENCY = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
LEGACY_CYCLE_ADJACENCY = ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))
MIXTURE_ADJACENCY = ((0., 0., 0., 1.), (0., 0., 0., 1.), (0., 0., 0., 1.), (1., 1., 1., 0.))
SEQUENTIAL_EDGES = ((0, 1), (1, 2), (2, 0))
MIXTURE_EDGES = ((0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2))


def validate_gcn_adjacency(value, num_agents=3):
    adjacency = torch.as_tensor(value, dtype=torch.float32, device='cpu').detach().clone()
    if (adjacency.shape != (num_agents, num_agents) or not torch.isfinite(adjacency).all()
            or (adjacency < 0).any()):
        raise ValueError(f'GCN adjacency must be a finite non-negative {num_agents}x{num_agents} matrix')
    return adjacency


def require_original_training_policy(payload):
    """Reject silent reuse of last night's changed prompts/objective on resume/eval."""
    config = payload.get('model_config', {})
    args = payload.get('training_args', {})
    changed = (config.get('solver_answer_check',args.get('solver_answer_check',False))
               or config.get('execution_plan_format',args.get('training_plan_format','original')) != 'original'
               or config.get('boxed_loss_weight',args.get('boxed_loss_weight',1.0)) != 1.0)
    if changed:
        raise ValueError('This checkpoint uses the removed Solver-check, target-first or weighted-CE experiment. '
                         'ConvRSI v3 restores the pre-experiment prompts and original targets; '
                         'use --init-checkpoint CHECKPOINT for a fresh training run, or evaluate a pre-experiment checkpoint')


def normalize_padding_config(config):
    if config.get('architecture') not in (*LEGACY_ARCHITECTURES, *RECURSIVE_ARCHITECTURES):
        return config
    config = dict(config)
    mixture = config['architecture'] == MIXTURE_ARCHITECTURE
    mas_system = MIXTURE if mixture else SEQUENTIAL
    if config.setdefault('mas_system', mas_system) != mas_system:
        raise ValueError('Checkpoint architecture disagrees with mas_system')
    # Checkpoints predating this field always had flattened cross-feature FCs.
    fusion = config.setdefault('cross_feature_fusion', True)
    if not isinstance(fusion, bool):
        raise ValueError('Checkpoint cross_feature_fusion must be a boolean')
    # Earlier graph updates always added the untransformed input as a skip.
    if not isinstance(config.setdefault('gcn_residual', True), bool):
        raise ValueError('Checkpoint gcn_residual must be a boolean')
    if not isinstance(config.setdefault('conv_pointwise', True), bool):
        raise ValueError('Checkpoint conv_pointwise must be a boolean')
    recursive = config['architecture'] in RECURSIVE_ARCHITECTURES
    mode = INTERLEAVED if recursive else SINGLE_PASS
    if config.get('collaboration_mode',mode) != mode:
        raise ValueError('Checkpoint architecture disagrees with collaboration_mode')
    config['collaboration_mode'] = mode
    # Historical checkpoints used a directed cycle. Fresh interleaved runs use a chain.
    graph = MIXTURE_ADJACENCY if mixture else LEGACY_CYCLE_ADJACENCY
    config['adjacency'] = validate_gcn_adjacency(config.get('adjacency', graph), len(MAS_ROLES[mas_system])).tolist()
    expected_features = EXPERIENCE_FEATURES if (config['architecture'] == EXPERIENCE_ARCHITECTURE
                                                or recursive and config.get('experience_text')) else FEATURES
    if config.get('feature_order',list(expected_features)) != list(expected_features):
        raise ValueError('Checkpoint has incompatible feature_order')
    padding = parse_conv_padding(config.get('conv_padding','same'))
    residual = config.get('conv_residual',True)
    count = len(config.get('feature_order',expected_features))
    lengths = validate_conv_layout(count,config['kernel_size'],config['dilations'],padding,residual)
    if config.get('conv_output_lengths',lengths) != lengths:
        raise ValueError('Checkpoint conv_output_lengths disagrees with its padding/kernel/dilations')
    policy = CONV_RESIDUAL_POLICY if residual else NO_CONV_RESIDUAL_POLICY
    if config.get('conv_residual_policy',policy) != policy:
        raise ValueError('Checkpoint has a different convolution residual policy')
    config.update(conv_padding=padding,conv_output_lengths=lengths,conv_residual=residual,conv_residual_policy=policy)
    if recursive:
        if config.get('rmas_passes') != config.get('csi_steps'):
            raise ValueError('Recursive checkpoint rmas_passes must match csi_steps')
        feedbacks = MIXTURE_ROUND_FEEDBACKS if mixture else RECURSIVE_ROUND_FEEDBACKS
        if config.get('message_update') not in feedbacks:
            raise ValueError('Recursive checkpoint has an incompatible feedback readout')
        feedback = feedbacks[config['message_update']]
        if config.get('round_feedback',feedback) != feedback:
            raise ValueError('Recursive checkpoint round_feedback disagrees with message_update')
        config['round_feedback'] = feedback
        if mixture and config.get('intermediate_summarizer') != MIXTURE_SUMMARIZER_POLICY:
            raise ValueError('Mixture checkpoint has an incompatible intermediate Summarizer policy')
        if not mixture and config.get('intermediate_solver') not in RECURSIVE_SOLVER_POLICIES:
            raise ValueError('Recursive checkpoint has an incompatible intermediate Solver policy')
    return config


class TokenUsage:
    """Logical LLM input positions; checkpoint recomputation is not counted twice.

    Device-local scalar counters are converted to Python integers only when logged
    or saved, avoiding an extra GPU synchronization on every model forward.
    """
    PHASES = ('inner', 'outer', 'validation')
    FIELDS = ('processed_token_positions', 'masked_padding_positions', 'forward_calls', 'feature_processed_token_positions',
              'generation_processed_token_positions', 'alignment_target_tokens', 'ce_target_tokens',
              'generated_tokens')
    LEGACY_FIELDS = {'processed_token_positions': 'input_tokens',
                     'masked_padding_positions': 'padding_tokens',
                     'feature_processed_token_positions': 'feature_input_tokens',
                     'generation_processed_token_positions': 'generation_input_tokens'}

    @classmethod
    def normalize_counter(cls, values):
        """Rename v1 counters without changing counts or epoch baselines."""
        return {key: int(values[key] if key in values else values[cls.LEGACY_FIELDS.get(key,key)])
                for key in cls.FIELDS}

    def __init__(self, roles=ROLES):
        self.roles = roles
        self.phase = None
        self.counts = {phase: {role: {key: 0 for key in self.FIELDS} for role in roles}
                       for phase in self.PHASES}
        self.coverage = dict(complete_from_run_start=True, unmeasured_before=None)

    @contextmanager
    def scope(self, phase):
        if phase not in self.PHASES:
            raise ValueError(f'Unknown token accounting phase: {phase}')
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def add(self, index, **values):
        if self.phase is None:
            return
        counter = self.counts[self.phase][self.roles[index]]
        for key, value in values.items():
            counter[key] += value.detach() if isinstance(value, Tensor) else value

    def record_forward(self, index, inputs, attention_mask=None, *, feature=False,
                       generation=False, ce_labels=None):
        if self.phase is None:
            return
        slots = inputs.shape[0]*inputs.shape[1]
        valid = slots if attention_mask is None else attention_mask.detach().sum(dtype=torch.int64)
        self.add(index, processed_token_positions=valid, masked_padding_positions=slots-valid, forward_calls=1,
                 feature_processed_token_positions=valid if feature else 0,
                 generation_processed_token_positions=valid if generation else 0)
        if ce_labels is not None:
            self.add(index, ce_target_tokens=ce_labels[:, 1:].ne(-100).sum())

    @contextmanager
    def generation(self, index, model):
        """Count actual prefill/cache input lengths inside Transformers generate()."""
        if self.phase is None:
            yield
            return

        def record(module, args, kwargs):
            inputs = kwargs.get('inputs_embeds')
            if inputs is None:
                inputs = kwargs.get('input_ids')
            if inputs is None:
                inputs = args[0]
            mask = kwargs.get('attention_mask')
            # A cached decode receives the whole context mask but only new inputs.
            if mask is not None:
                mask = mask[:, -inputs.shape[1]:]
            self.record_forward(index, inputs, mask, generation=True)

        handle = model.register_forward_pre_hook(record, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    def phase_totals(self, phase=None):
        phase = self.phase if phase is None else phase
        agents = self.counts[phase]
        for values in agents.values():
            for key in self.FIELDS:
                values[key] = int(values[key])
        return {key: sum(values[key] for values in agents.values()) for key in self.FIELDS}

    def snapshot(self):
        phases = {phase: dict(totals=self.phase_totals(phase),
                             agents=copy.deepcopy(self.counts[phase])) for phase in self.PHASES}
        inner, outer, validation = (phases[phase]['totals'] for phase in self.PHASES)
        training = {key: inner[key]+outer[key] for key in self.FIELDS}
        final_role = self.roles[-1].capitalize()
        return dict(schema_version=2, metric_name='processed_token_positions',
                    metric_label='处理的 token 位置数', coverage=copy.deepcopy(self.coverage),
                    counting=dict(
                        processed_token_positions='Non-padding input positions summed over logical LLM forwards; '
                                     'includes text, teacher-forced answers, CSI soft prefixes and incoming latents.',
                        feature_processed_token_positions='Subset of processed_token_positions used to encode independent CSI parts.',
                        generation_processed_token_positions=f'Subset used by {final_role} generate: prefill plus new cached inputs; '
                                                'includes batch slots processed after an individual row finishes.',
                        alignment_target_tokens='Inner next-token embedding alignment targets.',
                        ce_target_tokens=f'Per-agent Inner and final {final_role} Outer causal CE targets after the one-token shift.',
                        generated_tokens=f'Returned {final_role} continuation tokens through EOS, excluding trailing padding.',
                        excludes='Padding masked out by attention_mask, backward passes, gradient-checkpoint '
                                 'recomputation, FC/Conv/GCN positions and discarded work before a resume checkpoint.'),
                    totals=dict(training_processed_token_positions=training['processed_token_positions'],
                                validation_processed_token_positions=validation['processed_token_positions'],
                                all_processed_token_positions=training['processed_token_positions']+validation['processed_token_positions']),
                    training=training, phases=phases)

    def load_state_dict(self, saved, *, progress=None):
        if saved is None:
            # Older checkpoints did not measure tokens; do not invent prior totals.
            states = ([progress['outer']]+list(progress['inner'].values())) if progress else []
            if any(state['update'] or state['cursor'] for state in states):
                self.coverage = dict(complete_from_run_start=False,
                                     unmeasured_before=copy.deepcopy(progress))
            return
        if saved.get('schema_version') not in (1,2):
            raise ValueError('Unsupported token usage schema')
        self.coverage = copy.deepcopy(saved['coverage'])
        self.counts = {phase: {role: self.normalize_counter(saved['phases'][phase]['agents'][role])
                              for role in self.roles} for phase in self.PHASES}
        if progress is not None:
            for state in [progress['outer']]+list(progress['inner'].values()):
                if 'epoch_token_start' in state:
                    state['epoch_token_start'] = self.normalize_counter(state['epoch_token_start'])


def active_inner_losses(ce_weight, alignment_weight, cosine_weight, mse_weight):
    """Validate weights and return the effective CE, cosine and MSE switches."""
    weights = (ce_weight, alignment_weight, cosine_weight, mse_weight)
    if not all(math.isfinite(value) and value >= 0 for value in weights):
        raise ValueError('Inner loss weights must be finite and nonnegative')
    active = (ce_weight > 0, alignment_weight > 0 and cosine_weight > 0,
              alignment_weight > 0 and mse_weight > 0)
    if not any(active):
        raise ValueError('Inner loss needs an active CE or alignment term')
    return active


def resolve_csi_prefix_tokens(mode, tokens):
    if mode not in ('last_token', 'full_sequence'):
        raise ValueError('csi_feature_mode must be last_token or full_sequence')
    tokens = (8 if mode == 'full_sequence' else 1) if tokens is None else tokens
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError('csi_prefix_tokens must be a positive integer')
    if mode == 'last_token' and tokens != 1:
        raise ValueError('last_token baseline requires --csi-prefix-tokens 1')
    return tokens


def validate_projection_dims(local_dims, shared_dim, num_agents=3):
    # Accept the old spelling only when it describes the single shared width.
    if local_dims is not None and list(local_dims) != [shared_dim]*num_agents:
        raise ValueError('--local-dims no longer sets independent widths; omit it and use --d-model')


def convert_projection_checkpoint(payload):
    """Fold the v3 affine pair for weight loading, never its optimizer state.

    LN(H) -> (W1,b1) -> (W2,b2) -> LN(D) becomes
    LN(H) -> (W2@W1, W2@b1+b2) -> LN(D).
    Floating-point rounding can differ from the original two matmuls.
    """
    config = normalize_padding_config(payload.get('model_config', {}))
    if config.get('architecture') in (*LEGACY_ARCHITECTURES, *RECURSIVE_ARCHITECTURES):
        stored = payload.get('model_state_dict', {}).get('adjacency')
        num_agents = len(MAS_ROLES[config['mas_system']])
        if stored is not None and not torch.equal(validate_gcn_adjacency(stored, num_agents),
                                                 validate_gcn_adjacency(config['adjacency'], num_agents)):
            raise ValueError('Checkpoint GCN adjacency disagrees between model_config and model_state_dict')
    payload = dict(payload,model_config=config)
    if config.get('architecture') in RECURSIVE_ARCHITECTURES:
        if config.get('conv_projection') not in ('pre_conv','native_hidden'):
            raise ValueError('Recursive checkpoint requires a valid conv_projection')
        resolve_csi_prefix_tokens(config['csi_feature_mode'],config['csi_prefix_tokens'])
        return payload
    if config.get('architecture') != TWO_FC_ARCHITECTURE:
        if config.get('architecture') in (ARCHITECTURE, SEQUENCE_ARCHITECTURE, PROJECTED_ARCHITECTURE,
                                          EXPERIENCE_ARCHITECTURE):
            config = dict(config)
            if config['architecture'] == ARCHITECTURE:
                config.setdefault('csi_feature_mode', 'last_token')
                config.setdefault('csi_prefix_tokens', 1)
            if config['architecture'] == EXPERIENCE_ARCHITECTURE:
                projection = config.get('conv_projection')
                if projection not in ('pre_conv', 'native_hidden'):
                    raise ValueError('Experience checkpoint requires a valid conv_projection')
                if not isinstance(config.get('experience_text'), str) or not config['experience_text'].strip():
                    raise ValueError('Experience checkpoint requires nonempty experience_text')
                if config.get('feature_order') != list(EXPERIENCE_FEATURES):
                    raise ValueError('Experience checkpoint has incompatible feature_order')
            else:
                projection = 'pre_conv' if config['architecture'] == PROJECTED_ARCHITECTURE else 'native_hidden'
            if config.get('conv_projection', projection) != projection:
                raise ValueError('Checkpoint architecture disagrees with conv_projection')
            config['conv_projection'] = projection
            return dict(payload, model_config=config)
        return payload
    state = payload['model_state_dict'].copy()
    for index in range(3):
        post, shared = f'post_conv_fc.{index}.', f'to_shared.{index}.'
        w1, b1 = state[post+'1.weight'], state[post+'1.bias']
        w2, b2 = state.pop(shared+'0.weight'), state.pop(shared+'0.bias')
        # CPU half-precision checkpoints are folded in fp32, then cast back.
        dtype = torch.float64 if w1.dtype == torch.float64 else torch.float32
        state[post+'1.weight'] = (w2.to(dtype) @ w1.to(dtype)).to(w1.dtype)
        state[post+'1.bias'] = (w2.to(dtype) @ b1.to(dtype)+b2.to(dtype)).to(b1.dtype)
        state[post+'2.weight'] = state.pop(shared+'1.weight')
        state[post+'2.bias'] = state.pop(shared+'1.bias')
    config = dict(config, architecture=ARCHITECTURE,
                  feature_projection_order='native_hidden_conv_then_single_fc',
                  projection_fc_count_per_agent=1,
                  csi_feature_mode='last_token', csi_prefix_tokens=1, conv_projection='native_hidden')
    config.pop('local_dims', None)
    converted = dict(payload, model_state_dict=state, model_config=config,
                     weight_conversion=dict(source_architecture=TWO_FC_ARCHITECTURE,
                                            optimizer_state_compatible=False))
    # Adam moments for two factors cannot be continued as moments of their product.
    for key in ('optimizer_state_dict', 'inner_optimizer_state_dicts',
                'lr_scheduler_state_dict', 'inner_lr_scheduler_state_dicts'):
        converted.pop(key, None)
    return converted


class MaskedTokenPooler(nn.Module):
    """Read all valid hidden states into K content-dependent slots at native H.

    A per-token FC produces K scores; each slot softmaxes over input positions.
    The same slot reader is shared by task/role/system within one agent.
    """
    def __init__(self, hidden_size, tokens):
        super().__init__()
        self.score_norm = nn.LayerNorm(hidden_size)
        self.score = nn.Linear(hidden_size, tokens, bias=False)
        self.slot_embedding = nn.Parameter(torch.randn(tokens, hidden_size)*0.02)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden, mask):
        if hidden.ndim != 3 or tuple(mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError('Token pooler needs [B,L,H] hidden states and a [B,L] mask')
        mask = mask.to(device=self.score.weight.device, dtype=torch.bool)
        if not mask.any(-1).all():
            raise ValueError('Each pooled sequence needs at least one valid token')
        hidden = hidden.to(self.score.weight).masked_fill(~mask[:, :, None], 0)
        scores = self.score(self.score_norm(hidden)).transpose(1, 2).float()
        scores = scores.masked_fill(~mask[:, None], float('-inf'))
        weights = scores.softmax(dim=-1).to(hidden.dtype)
        pooled = torch.matmul(weights, hidden)
        return self.output_norm(pooled+self.slot_embedding)


def stage_prompt(question: str, index: int, step: int = 1) -> str:
    if SLOT in question or any(s in question for s in
                              (rmas.PLANNER_SLOT, rmas.REFINED_SLOT, rmas.FEEDBACK_SLOT)):
        raise ValueError("Question contains a reserved latent marker")
    if step < 1:
        raise ValueError("RMAS round must be positive")
    if index == 0:
        if step > 1:
            return (rmas.build_math_planner_prompt_with_feedback_slot(question)
                    .replace(rmas.FEEDBACK_SLOT,SLOT).replace('This is round 2.',f'This is round {step}.'))
        return rmas.build_math_planner_prompt(question)
    if index == 1:
        return rmas.build_math_refiner_prompt_with_slot(question).replace(rmas.PLANNER_SLOT, SLOT)
    if index == 2:
        return rmas.build_math_solver_prompt_with_slots(question).replace(rmas.REFINED_SLOT, SLOT)
    raise ValueError("Unknown agent index")


def mixture_stage_prompt(question, index, step=1, *, row=None, expert_outputs=None):
    row = {} if row is None else row
    family = row.get('task_family')
    family = family.strip().lower() if isinstance(family, str) and family.strip() else 'math'
    if family not in MIXTURE_TASK_FAMILIES:
        raise ValueError(f'Unsupported Mixture task_family: {family}')
    fn_name = row.get('fn_name')
    fn_name = fn_name.strip() if isinstance(fn_name, str) else None
    if any(marker in question or marker in (fn_name or '')
           for marker in (*MIXTURE_EXPERT_SLOTS, rmas.HIE_FEEDBACK_SLOT)):
        raise ValueError('Question or function name contains a reserved latent marker')
    kwargs = dict(mas_task=family, task_type=row.get('type') or 'complete', fn_name=fn_name)
    if index == len(MIXTURE_ROLES)-1:
        if expert_outputs is not None:
            return rmas.build_hie_summarizer_prompt(question, *expert_outputs, **kwargs)
        return rmas.build_hie_summarizer_prompt_with_slots(question, **kwargs)
    role = 'hie_' + MIXTURE_ROLES[index]
    if step > 1:
        return rmas.build_hie_expert_prompt_with_feedback_slot(question, role, **kwargs)
    return rmas.build_hie_expert_prompt(question, role, **kwargs)


@dataclass
class Pack:
    inputs: Tensor                    # [1, T, H_i]
    labels: Tensor                    # [1, T], prompt and soft inputs = -100
    prompt_tokens: int
    incoming_tokens: int
    message_tokens: int


@dataclass
class BatchPack:
    inputs: Tensor
    labels: Tensor
    attention_mask: Tensor
    rows: list[Pack]


class Backbone:
    def __init__(self, model, tokenizer, name: str = ""):
        self.model, self.tokenizer, self.name = model, tokenizer, name
        self.model.requires_grad_(False).eval()
        self.hidden_size = model.get_input_embeddings().weight.shape[-1]
        self.device = model.get_input_embeddings().weight.device

    @classmethod
    def load(cls, name, device="cuda:0", dtype="bfloat16", trust_remote_code=False,
             attn_implementation=""):
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        if not tok.chat_template:
            raise ValueError(f"Model needs an explicit chat template: {name}")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None:
            raise ValueError("Tokenizer needs an EOS or pad token")
        kwargs = dict(dtype='auto' if dtype == 'auto' else getattr(torch, dtype),
                      trust_remote_code=trust_remote_code)
        if attn_implementation:
            kwargs['attn_implementation'] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs).to(device)
        return cls(model, tok, name)


def answer_ce(logits: Tensor, labels: Tensor) -> Tensor:
    # Give each question the same weight even when answer lengths differ.
    if labels.shape[0] > 1:
        return torch.stack([answer_ce(scores[None], target[None])
                            for scores, target in zip(logits, labels)]).mean()
    valid = labels[:, 1:].ne(-100)
    if not valid.any():
        raise ValueError("No supervised answer tokens")
    scores, targets = logits[:, :-1][valid], labels[:, 1:][valid]
    # Avoid casting prompt * vocabulary to fp32; chunk only supervised tokens.
    total = sum(F.cross_entropy(s.float(), t, reduction="sum")
                for s, t in zip(scores.split(128), targets.split(128)))
    return total / targets.numel()


def generation_end_metadata(tokens, *, max_new_tokens, eos_token_id):
    """Observe returned token IDs; never add a stopping rule or infer one from text."""
    token_ids = tokens.tolist() if isinstance(tokens, Tensor) else list(tokens)
    if max_new_tokens < 1 or len(token_ids) > max_new_tokens:
        raise ValueError('Invalid returned generation length or max_new_tokens')
    if isinstance(eos_token_id, Tensor):
        eos_token_id = eos_token_id.tolist()
    eos_ids = set([] if eos_token_id is None else
                  [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)
    stop = next((i for i,token in enumerate(token_ids) if token in eos_ids), None)
    count = len(token_ids) if stop is None else stop+1
    # EOS at the exact token budget is still reported as EOS. The independent
    # hit_token_limit flag records that both conditions hold on that last step.
    reason = 'eos' if stop is not None else 'length' if count == max_new_tokens else 'other'
    return dict(generated_tokens=count, max_new_tokens=max_new_tokens, finish_reason=reason,
                stop_token_id=None if stop is None else int(token_ids[stop]),
                hit_token_limit=count == max_new_tokens)


class CSIQwenSystem(nn.Module):
    def __init__(self, backbones: list[Backbone], *, shared_dim=256,
                 local_dims=None, csi_steps=3, latent_steps=32,
                 max_latent_tokens=80, max_length=4096, max_feature_tokens=2048,
                 kernel_size=3, dilations=(1, 2), gradient_checkpointing=True,
                 csi_device=None, csi_dtype='float32',
                 csi_feature_mode='full_sequence', csi_prefix_tokens=None, conv_projection='pre_conv',
                 experience_text='', conv_padding='same', conv_residual=True,
                 collaboration_mode=INTERLEAVED, gcn_adjacency=None,
                 intermediate_solver_policy=RECURSIVE_SOLVER_POLICY,
                 message_update=RECURSIVE_MESSAGE_UPDATE, cross_feature_fusion=False,
                 gcn_residual=False, conv_pointwise=False, mas_system=SEQUENTIAL):
        super().__init__()
        if mas_system not in MAS_ROLES:
            raise ValueError(f'Unknown MAS system: {mas_system}')
        self.mas_system, self.roles = mas_system, MAS_ROLES[mas_system]
        self.final_index = len(self.roles)-1
        self.token_usage = TokenUsage(self.roles)
        if len(backbones) != len(self.roles):
            raise ValueError(f'{mas_system} requires {len(self.roles)} backbones in order {self.roles}')
        if collaboration_mode not in (SINGLE_PASS,INTERLEAVED):
            raise ValueError('Unknown collaboration_mode')
        self.collaboration_mode = collaboration_mode
        if mas_system == MIXTURE and (collaboration_mode != INTERLEAVED
                                      or intermediate_solver_policy != RECURSIVE_SOLVER_POLICY):
            raise ValueError('Mixture requires interleaved rounds with teacher-forced Summarizer feedback')
        if intermediate_solver_policy not in RECURSIVE_SOLVER_POLICIES:
            raise ValueError('Unknown intermediate Solver training policy')
        self.intermediate_solver_policy = intermediate_solver_policy
        if message_update not in RECURSIVE_ROUND_FEEDBACKS:
            raise ValueError('Unknown recursive Message update policy')
        self.message_update = message_update
        validate_projection_dims(local_dims, shared_dim, len(self.roles))
        if conv_projection not in ('pre_conv', 'native_hidden'):
            raise ValueError('conv_projection must be pre_conv or native_hidden')
        self.conv_projection = conv_projection
        if not isinstance(cross_feature_fusion, bool):
            raise ValueError('cross_feature_fusion must be a boolean')
        self.cross_feature_fusion = cross_feature_fusion
        if not isinstance(gcn_residual, bool):
            raise ValueError('gcn_residual must be a boolean')
        self.gcn_residual = gcn_residual
        if not isinstance(experience_text, str):
            raise ValueError('experience_text must be a string')
        self.experience_text = experience_text.strip()
        self.features = EXPERIENCE_FEATURES if self.experience_text else FEATURES
        self.conv_padding = parse_conv_padding(conv_padding)
        self.conv_residual = conv_residual
        self.conv_pointwise = conv_pointwise
        self.conv_output_lengths = validate_conv_layout(len(self.features),kernel_size,dilations,self.conv_padding,conv_residual)
        self.conv_residual_policy = CONV_RESIDUAL_POLICY if conv_residual else NO_CONV_RESIDUAL_POLICY
        self.csi_feature_mode = csi_feature_mode
        self.csi_prefix_tokens = resolve_csi_prefix_tokens(csi_feature_mode, csi_prefix_tokens)
        if min(shared_dim, csi_steps, latent_steps, max_latent_tokens,
               max_length, max_feature_tokens) < 1:
            raise ValueError("Dimensions, rounds and limits must be positive")
        # These plain objects keep frozen LLM weights out of checkpoints.
        self.backbones = backbones
        self.options = dict(shared_dim=shared_dim, csi_steps=csi_steps,
                            latent_steps=latent_steps, max_latent_tokens=max_latent_tokens,
                            max_length=max_length, max_feature_tokens=max_feature_tokens,
                            kernel_size=kernel_size, dilations=list(dilations),
                            gradient_checkpointing=gradient_checkpointing,
                            csi_feature_mode=csi_feature_mode, csi_prefix_tokens=self.csi_prefix_tokens,
                            conv_projection=conv_projection,conv_padding=self.conv_padding,conv_residual=conv_residual,
                            collaboration_mode=collaboration_mode,cross_feature_fusion=cross_feature_fusion,
                            gcn_residual=gcn_residual,conv_pointwise=conv_pointwise)
        if self.experience_text:
            self.options['experience_text'] = self.experience_text
        self.csi_steps, self.latent_steps = csi_steps, latent_steps
        self.max_latent_tokens, self.max_length = max_latent_tokens, max_length
        self.max_feature_tokens = max_feature_tokens
        self.gradient_checkpointing = gradient_checkpointing
        sizes = [b.hidden_size for b in backbones]
        self.inner_links = nn.ModuleList(Adapter(h, "ln_res_adapter") for h in sizes)
        # Keep the historical Sequential link ordering for its checkpoints.
        self.outer_edges = (MIXTURE_EDGES if mas_system == MIXTURE else
                            SEQUENTIAL_EDGES if collaboration_mode == INTERLEAVED else SEQUENTIAL_EDGES[:2])
        self.outer_links = nn.ModuleList(CrossModelAdapter(sizes[src], sizes[dst], "outer_ln_res_adapter")
                                         for src, dst in self.outer_edges)
        # Project each part/slot independently H_i -> D before convolution.
        # No new random parameters are created on the legacy native-width path.
        self.feature_fc = (nn.ModuleList(nn.Sequential(nn.LayerNorm(h), nn.Linear(h, shared_dim))
                                         for h in sizes) if conv_projection == 'pre_conv' else None)
        widths = [shared_dim]*len(self.roles) if self.feature_fc is not None else sizes
        self.local_conv = nn.ModuleList(IntraAgentConv(width, len(self.features), kernel_size, dilations,
                                                       padding=self.conv_padding,residual=conv_residual,
                                                       cross_feature_fusion=cross_feature_fusion,
                                                       conv_pointwise=conv_pointwise)
                                        for width in widths)
        # Keep the pooled-message transform: D -> D on the projected path.
        self.post_conv_fc = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, shared_dim),
                                                       nn.LayerNorm(shared_dim)) for width in widths)
        self.gcn = GraphConvolutionUpdate(shared_dim,residual=gcn_residual)
        self.from_shared = nn.ModuleList(nn.Sequential(nn.Linear(shared_dim, h), nn.LayerNorm(h))
                                         for h in sizes)
        # A[receiver, sender]: P->C->S; GCN adds self-loops. S->P OuterLink is separate.
        graph = (MIXTURE_ADJACENCY if mas_system == MIXTURE else
                 CHAIN_ADJACENCY if collaboration_mode == INTERLEAVED else LEGACY_CYCLE_ADJACENCY)
        self.register_buffer("adjacency", validate_gcn_adjacency(
            graph if gcn_adjacency is None else gcn_adjacency, len(self.roles)))
        self.sequence_poolers = (nn.ModuleList(MaskedTokenPooler(h, self.csi_prefix_tokens) for h in sizes)
                                 if csi_feature_mode == 'full_sequence' else None)
        self.to(device=csi_device or backbones[0].device, dtype=getattr(torch, csi_dtype))
        self.set_inner_trainable(False)

    def train(self, mode=True):
        super().train(mode)
        for backbone in self.backbones:
            backbone.model.eval()
        return self

    def set_inner_trainable(self, enabled):
        self.inner_links.requires_grad_(enabled)

    def inner_modules(self, index):
        """Independent per-agent modules used by Inner, excluding GCN and OuterLinks."""
        modules = [self.inner_links[index], self.local_conv[index],
                   self.post_conv_fc[index], self.from_shared[index]]
        if self.feature_fc is not None:
            modules.append(self.feature_fc[index])
        if self.sequence_poolers is not None:
            modules.append(self.sequence_poolers[index])
        return modules

    def inner_parameters(self, index):
        """The selected agent's InnerLink and local CSI parameters."""
        for module in self.inner_modules(index):
            yield from module.parameters()

    def configure_training_stage(self, stage, index=None):
        """Freeze inactive modules even when Inner and Outer optimizers coexist."""
        if stage == 'outer':
            self.requires_grad_(True)
            self.set_inner_trainable(False)
        elif stage == 'inner' and index in range(len(self.roles)):
            self.requires_grad_(False)
            for parameter in self.inner_parameters(index):
                parameter.requires_grad_(True)
        else:
            raise ValueError('Training stage must be outer or inner with an agent index')

    def _ids(self, index, text):
        b = self.backbones[index]
        return b.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(b.device)

    def _render(self, index, user, answer=None):
        messages = [{"role": "system", "content": rmas.SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        tokenizer = self.backbones[index].tokenizer
        kwargs = dict(tokenize=False, add_generation_prompt=answer is None, enable_thinking=False)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as exc:
            error = str(exc).lower()
            if 'system' not in error and 'conversation roles must alternate' not in error:
                raise
            # Templates such as BioMistral's only accept user/assistant turns.
            merged = [{'role':'user', 'content':rmas.SYSTEM_PROMPT+'\n\n'+user}]
            if answer is not None:
                merged.append({'role':'assistant', 'content':answer})
            return tokenizer.apply_chat_template(merged, **kwargs)

    def pack(self, index, user, *, incoming=None, soft=None, answer=None) -> Pack:
        """Render role text with ordered upstream latent slots and a CSI soft prefix."""
        b = self.backbones[index]
        embedding = b.model.get_input_embeddings()
        prompt = self._render(index, user)
        full = prompt if answer is None else self._render(index, user, answer)
        pieces = []
        if soft is not None:
            if tuple(soft.shape) != (self.csi_prefix_tokens, b.hidden_size):
                raise ValueError(f'CSI soft prefix must have {self.csi_prefix_tokens} tokens')
            pieces.append(soft.to(embedding.weight))
        incoming_length = 0
        markers = (SLOT,)
        if self.mas_system == MIXTURE:
            markers = MIXTURE_EXPERT_SLOTS if index == self.final_index else (rmas.HIE_FEEDBACK_SLOT,)
        prompt_tail, full_tail = prompt, full
        if any(marker in prompt for marker in markers):
            sequences = (incoming,) if len(markers) == 1 else incoming
            if incoming is None or len(sequences) != len(markers) or any(
                prompt.count(marker) != 1 or full.count(marker) != 1 for marker in markers
            ):
                raise ValueError('Latent slots require one sequence per marker')
            for marker, sequence in zip(markers, sequences):
                before, found, prompt_tail = prompt_tail.partition(marker)
                full_before, full_found, full_tail = full_tail.partition(marker)
                if not found or not full_found or before != full_before:
                    raise ValueError('Chat template changed context or order before a latent slot')
                pieces.extend([embedding(self._ids(index, before)), sequence.to(embedding.weight)])
                incoming_length += len(sequence)
        elif incoming is not None:
            raise ValueError("Incoming sequence has no slot")
        prompt_ids, full_ids = self._ids(index, prompt_tail), self._ids(index, full_tail)
        common = 0
        for x, y in zip(prompt_ids.tolist(), full_ids.tolist()):
            if x != y:
                break
            common += 1
        prefix_length = sum(len(p) for p in pieces)
        inputs = torch.cat([*pieces, embedding(full_ids)], dim=0)
        if len(inputs) > self.max_length:
            raise ValueError(f"{self.roles[index]} has {len(inputs)} tokens > max_length={self.max_length}; no truncation")
        labels = torch.full((len(inputs),), -100, dtype=torch.long, device=b.device)
        if answer is not None:
            if common == len(full_ids):
                raise ValueError("Chat template produced no assistant target")
            labels[prefix_length+common:] = full_ids[common:]
        return Pack(inputs[None], labels[None], prefix_length+common,
                    incoming_length, 0 if soft is None else len(soft))

    @staticmethod
    def _pad_sequences(sequences):
        if not sequences or any(len(sequence) == 0 for sequence in sequences):
            raise ValueError('Cannot batch an empty sequence')
        width = max(len(sequence) for sequence in sequences)
        inputs = torch.stack([F.pad(sequence, (0, 0, 0, width-len(sequence))) for sequence in sequences])
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=inputs.device)
        mask = (torch.arange(width, device=inputs.device)[None] < lengths[:, None]).long()
        return inputs, mask

    def batch_pack(self, index, users, *, incoming=None, soft=None, answers=None):
        batch = len(users)
        if not batch or (incoming is not None and len(incoming) != batch):
            raise ValueError('Batch prompts and incoming sequences must match')
        if answers is not None and len(answers) != batch:
            raise ValueError('Batch prompts and targets must match')
        if soft is not None and tuple(soft.shape) != (batch, self.csi_prefix_tokens, self.backbones[index].hidden_size):
            raise ValueError(f'Each question needs {self.csi_prefix_tokens} CSI soft tokens')
        packs = [self.pack(index, user,
                           incoming=None if incoming is None else incoming[b],
                           soft=None if soft is None else soft[b],
                           answer=None if answers is None else answers[b]) for b, user in enumerate(users)]
        inputs, mask = self._pad_sequences([pack.inputs[0] for pack in packs])
        labels = torch.stack([F.pad(pack.labels[0], (0, inputs.shape[1]-pack.inputs.shape[1]), value=-100)
                              for pack in packs])
        return BatchPack(inputs, labels, mask, packs)

    def hidden(self, index, inputs, *, ce_labels=None, feature=False, attention_mask=None,
               use_cache=False, past_key_values=None, position_ids=None):
        b = self.backbones[index]
        limit = self.max_feature_tokens if feature else self.max_length
        if not inputs.shape[1] or inputs.shape[1] > limit:
            raise ValueError(f"{self.roles[index]} forward length {inputs.shape[1]} exceeds limit {limit}")
        if use_cache and (feature or ce_labels is not None or torch.is_grad_enabled() and inputs.requires_grad):
            raise ValueError('KV cache is only enabled for no-grad latent inference')
        if attention_mask is not None:
            valid_shape = (attention_mask.shape[0] == inputs.shape[0] and
                           attention_mask.shape[1] >= inputs.shape[1] if use_cache else
                           tuple(attention_mask.shape) == tuple(inputs.shape[:2]))
            if not valid_shape:
                raise ValueError('Attention mask must match the batch and sequence axes')
            if use_cache and attention_mask.shape[1] > limit:
                raise ValueError('Cached latent context exceeds max_length')
        # Outside run(): backward checkpoint recomputation must not inflate usage.
        measured_mask = (attention_mask[:, -inputs.shape[1]:]
                         if use_cache and attention_mask is not None else attention_mask)
        self.token_usage.record_forward(index, inputs, measured_mask, feature=feature, ce_labels=ce_labels)

        def run(x):
            mask = (torch.ones(x.shape[:2], device=x.device, dtype=torch.long)
                    if attention_mask is None else attention_mask.to(device=x.device, dtype=torch.long))
            kwargs = dict(inputs_embeds=x, attention_mask=mask,
                          use_cache=use_cache, return_dict=True)
            if position_ids is not None:
                kwargs['position_ids'] = position_ids
            elif attention_mask is not None:
                kwargs['position_ids'] = (mask.cumsum(-1)-1).clamp_min(0)
                if use_cache:
                    kwargs['position_ids'] = kwargs['position_ids'][:, -x.shape[1]:]
            if past_key_values is not None:
                kwargs['past_key_values'] = past_key_values
            if ce_labels is not None:
                out = b.model(**kwargs, output_hidden_states=True)
                return out.hidden_states[-1], answer_ce(out.logits, ce_labels)
            try:
                decoder = b.model.get_decoder()
            except (AttributeError, NotImplementedError):
                decoder = getattr(b.model, 'model', None)
            if decoder is not None:
                outputs = decoder(**kwargs)
                hidden = outputs.last_hidden_state
            else:
                outputs = b.model(**kwargs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
            if use_cache:
                cache = getattr(outputs, 'past_key_values', None)
                if cache is None:
                    raise ValueError(f'{self.roles[index]} did not return past_key_values with use_cache=True')
                return hidden, x.new_zeros(()), cache
            return hidden, x.new_zeros(())

        if self.gradient_checkpointing and torch.is_grad_enabled() and inputs.requires_grad:
            return checkpoint(run, inputs, use_reentrant=False)
        return run(inputs)

    def _encode_text(self, index, text):
        # No chat wrapper or assistant suffix: read the actual part's last token.
        embedding = self.backbones[index].model.get_input_embeddings()
        return self.hidden(index, embedding(self._ids(index, text))[None], feature=True)[0][0, -1]

    def _role_prompt(self, index):
        if self.mas_system == MIXTURE:
            return mixture_stage_prompt('', index, expert_outputs=('', '', ''))
        return stage_prompt('', index).replace(SLOT, '')

    def encode_parts(self, index, question, soft=None):
        """Encode independent parts with full-sequence slots or last-token readout.

        The first CSI round has no message, so its last state is masked zero.
        Later rounds use the previous GCN output mapped to native hidden width.
        Legacy interleaved checkpoints can explicitly restore LLM-output pooling.
        """
        if self.csi_feature_mode == 'full_sequence':
            parts, mask = self.encode_parts_batch(index, [question], None if soft is None else soft[None])
            return parts[0], mask[0]
        b = self.backbones[index]
        role = self._role_prompt(index)
        states = [self._encode_text(index, question), self._encode_text(index, role),
                  self._encode_text(index, rmas.SYSTEM_PROMPT)]
        if self.experience_text:
            states.append(self._encode_text(index, self.experience_text))
        absent = b.model.get_input_embeddings().weight.new_zeros(b.hidden_size)
        states.append(absent if soft is None else
                      self.hidden(index, soft.to(absent)[None], feature=True)[0][0, -1])
        mask = torch.tensor([True]*(len(self.features)-1)+[soft is not None], device=b.device)
        return torch.stack(states), mask

    def _encode_sequences_batch(self, index, sequences):
        inputs, mask = self._pad_sequences(sequences)
        hidden, _ = self.hidden(index, inputs, feature=True, attention_mask=mask)
        last = mask.sum(-1)-1
        return hidden[torch.arange(len(sequences), device=hidden.device), last]

    def _encode_sequence_slots(self, index, sequences):
        inputs, mask = self._pad_sequences(sequences)
        # Static text comes from frozen embeddings. Cache the entire last layer
        # without its LLM graph, then train the pooler OUTSIDE no_grad.
        with torch.set_grad_enabled(torch.is_grad_enabled() and inputs.requires_grad):
            hidden, _ = self.hidden(index, inputs, feature=True, attention_mask=mask)
        return self.sequence_poolers[index](hidden, mask)

    def encode_parts_batch(self, index, questions, soft=None):
        """Batch questions within each independent initial part, never across parts."""
        b = self.backbones[index]
        embedding = b.model.get_input_embeddings()
        batch = len(questions)
        role = self._role_prompt(index)
        encode = self._encode_sequence_slots if self.csi_feature_mode == 'full_sequence' else self._encode_sequences_batch
        text_batch = lambda texts: encode(
            index, [embedding(self._ids(index, text)) for text in texts])
        states = [text_batch(questions), text_batch([role]*batch),
                  text_batch([rmas.SYSTEM_PROMPT]*batch)]
        if self.experience_text:
            states.append(text_batch([self.experience_text]*batch))
        absent = torch.zeros_like(states[0])
        if soft is not None and tuple(soft.shape) != (batch, self.csi_prefix_tokens, b.hidden_size):
            raise ValueError(f'Each initial message must have {self.csi_prefix_tokens} soft tokens')
        states.append(absent if soft is None else encode(
            index, [sequence.to(embedding.weight) for sequence in soft.unbind(0)]))
        mask = torch.tensor([[True]*(len(self.features)-1)+[soft is not None]]*batch,
                            device=states[0].device)
        if self.csi_feature_mode == 'full_sequence':
            mask = mask[:, :, None].expand(-1, -1, self.csi_prefix_tokens)
        return torch.stack(states, dim=1), mask

    def local_message_batch(self, index, states, mask):
        ref = self.post_conv_fc[index][1].weight
        features, mask = states.to(ref), mask.to(device=ref.device, dtype=torch.bool)
        if self.feature_fc is not None:
            features = self.feature_fc[index](features)
            # A biased projection must not turn the missing initial message into a state.
            features = features.masked_fill(~mask[..., None], 0)
        if self.csi_feature_mode == 'full_sequence':
            pooled = self.local_conv[index](features, return_sequence=True, feature_mask=mask)
        else:
            pooled = self.local_conv[index](features[:, :, None, :], feature_mask=mask[:, :, None])
        return self.post_conv_fc[index](pooled)  # Retain K; output width is always D.

    def exchange_batch(self, messages):
        shared = torch.stack(messages, dim=1)
        return self.gcn(shared, self.adjacency)

    def local_message(self, index, states, mask):
        return self.local_message_batch(index, states[None], mask[None])[0]

    def exchange(self, messages):
        shared = torch.stack(messages)[None]
        result, normalized = self.gcn(shared, self.adjacency)
        return result[0], normalized[0]

    def message_prefix(self, index, message):
        prefix = self.from_shared[index](message)
        return prefix if self.csi_feature_mode == 'full_sequence' else prefix[:, None]

    def transfer(self, index, states, *, training):
        return self._transfer_links(index, states, links=(index,), training=training)[0]

    def _transfer_links(self, index, states, *, links, training):
        ref = self.inner_links[index].proj1.weight
        latent = self.inner_links[index](states.to(ref))
        outputs = [self.outer_links[edge](latent) for edge in links]
        return [out[:self.max_latent_tokens] for out in outputs] if training else outputs

    def latent_rollout(self, index, pack):
        if not torch.is_grad_enabled() or not pack.inputs.requires_grad:
            mask = torch.ones(pack.inputs.shape[:2], dtype=torch.long, device=pack.inputs.device)
            return self._cached_latent_rollout(index, pack.inputs, mask)[0]
        collected, inputs = [], pack.inputs
        if inputs.shape[1] + self.latent_steps > self.max_length:
            raise ValueError("Latent rollout exceeds max_length")
        for _ in range(self.latent_steps):
            hidden, _ = self.hidden(index, inputs)
            last = hidden[0, -1]
            collected.append(last)
            ref = self.inner_links[index].proj1.weight
            next_token = self.inner_links[index](last.to(ref)).to(inputs)
            inputs = torch.cat([inputs, next_token[None, None]], dim=1)
        return torch.stack(collected)

    @torch.no_grad()
    def _cached_latent_rollout(self, index, inputs, mask):
        """One full prefill, then one new soft token per step; cache is local to this call."""
        width = inputs.shape[1]
        if width+self.latent_steps > self.max_length:
            raise ValueError('Latent rollout exceeds max_length')
        lengths = mask.sum(-1)
        # Align valid sequence ends before creating a shared-length batched cache.
        # Position IDs still follow each question's unpadded sequence length.
        positions = torch.arange(width, device=inputs.device)[None]
        gather = (positions-(width-lengths[:, None])) % width
        inputs = inputs.gather(1, gather[:, :, None].expand(-1, -1, inputs.shape[-1]))
        mask = mask.gather(1, gather)
        position_ids = (mask.cumsum(-1)-1).clamp_min(0)
        cache, collected = None, []
        for step in range(self.latent_steps):
            hidden, _, cache = self.hidden(index, inputs, attention_mask=mask,
                position_ids=position_ids, use_cache=True, past_key_values=cache)
            last = hidden[:, -1].clone()
            collected.append(last)
            ref = self.inner_links[index].proj1.weight
            inputs = self.inner_links[index](last.to(ref)).to(inputs)[:, None]
            mask = F.pad(mask, (0, 1), value=1)
            position_ids = (lengths+step)[:, None]
        return torch.stack(collected, dim=1)

    def latent_rollout_batch(self, index, pack, *, differentiable=False):
        """Cached inference; legacy differentiable training keeps full-context forwards."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and differentiable):
            if not differentiable:
                return self._cached_latent_rollout(index, pack.inputs, pack.attention_mask)
            inputs, mask = pack.inputs, pack.attention_mask
            if inputs.shape[1] + self.latent_steps > self.max_length:
                raise ValueError("Latent rollout exceeds max_length")
            lengths = mask.sum(-1)
            batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
            collected = []
            for _ in range(self.latent_steps):
                hidden, _ = self.hidden(index, inputs, attention_mask=mask)
                last = hidden[batch_indices, lengths-1]
                collected.append(last)
                ref = self.inner_links[index].proj1.weight
                next_token = self.inner_links[index](last.to(ref)).to(inputs)
                # Append immediately after each question's valid tokens, before its padding.
                inputs = F.pad(inputs, (0, 0, 0, 1))
                inputs[batch_indices, lengths] = next_token
                mask = F.pad(mask, (0, 1))
                mask[batch_indices, lengths] = 1
                lengths = lengths+1
            return torch.stack(collected, dim=1)  # [B, latent_steps, H_i]
    def inner_loss(self, row, index, *, ce_weight=1., alignment_weight=1.,
                   cosine_weight=1., mse_weight=0.1, return_components=False):
        return self.inner_loss_batch([row], index, ce_weight=ce_weight,
                                     alignment_weight=alignment_weight, cosine_weight=cosine_weight,
                                     mse_weight=mse_weight, return_components=return_components)

    def inner_loss_batch(self, rows, index, *, ce_weight=1., alignment_weight=1.,
                         cosine_weight=1., mse_weight=0.1, return_components=False):
        """One local CSI pass, then one teacher-forced LLM forward per agent.

        Static parts never contain reference outputs. The Message part is masked
        zero, and GCN/OuterLinks are not called. CE updates the local CSI prefix;
        alignment sees detached hidden states and updates only this InnerLink.
        Zero-weight terms are not evaluated or included in returned components.
        """
        if not rows:
            raise ValueError('Inner batch must be nonempty')
        ce_active, cosine_active, mse_active = active_inner_losses(
            ce_weight, alignment_weight, cosine_weight, mse_weight)
        # Only CE needs a graph through the prefix and frozen LLM. Alignment
        # still uses the identical prefix-conditioned trajectory, without a graph.
        with torch.set_grad_enabled(torch.is_grad_enabled() and ce_active):
            questions = [question_of(row) for row in rows]
            parts, feature_mask = self.encode_parts_batch(index, questions)
            message = self.local_message_batch(index, parts, feature_mask)
            soft = self.message_prefix(index, message)
            if self.mas_system == MIXTURE:
                users = [mixture_stage_prompt(
                    question, index, row=row,
                    expert_outputs=([mixture_target_of(row, i) for i in range(self.final_index)]
                                    if index == self.final_index else None),
                ) for question, row in zip(questions, rows)]
                answers = [mixture_target_of(row, index) for row in rows]
            else:
                users = [stage_prompt(question, index, 1) for question in questions]
                if index:
                    users = [user.replace(SLOT, target_of(row, index-1)) for user, row in zip(users, rows)]
                answers = [target_of(row, index) for row in rows]
            pack = self.batch_pack(index, users, soft=soft, answers=answers)
            # With no CE labels, hidden() calls the decoder without an LM head.
            hidden, ce = self.hidden(index, pack.inputs, ce_labels=pack.labels if ce_active else None,
                                     attention_mask=pack.attention_mask)
        terms, components = [], {}
        if ce_active:
            terms.append(ce_weight*ce.to(self.post_conv_fc[index][1].weight.device))
            components['ce_loss'] = ce.detach()
        if cosine_active or mse_active:
            with torch.no_grad():
                valid = pack.labels[:, 1:].ne(-100)
                self.token_usage.add(index, alignment_target_tokens=valid.sum())
                counts = valid.sum(-1).tolist()
                if min(counts) < 1:
                    raise ValueError('Each question needs supervised Inner tokens')
                source = hidden.detach()[:, :-1][valid]
                targets = self.backbones[index].model.get_input_embeddings()(pack.labels[:, 1:][valid])
            ref = self.inner_links[index].proj1.weight
            predicted = self.inner_links[index](source.to(ref)).float()
            targets = targets.to(device=predicted.device, dtype=torch.float32)
            alignment_terms = []
            if cosine_active:
                alignment_terms.append(cosine_weight*(1-F.cosine_similarity(predicted, targets, dim=-1)))
            if mse_active:
                alignment_terms.append(mse_weight*F.mse_loss(predicted, targets, reduction='none').mean(-1))
            losses = sum(alignment_terms)
            alignment = torch.stack([sample.mean() for sample in losses.split(counts)]).mean()
            terms.append(alignment_weight*alignment)
            components['alignment_loss'] = alignment.detach()
        loss = sum(terms)
        if return_components:
            return loss, components
        return loss

    def refine_messages_batch(self, questions, *, trace=False):
        """Encode once, then refine with the configured FC/convolution/GCN path."""
        if self.collaboration_mode != SINGLE_PASS:
            raise ValueError('Interleaved CSI needs each pass output; use collaborate/generate instead')
        if not questions:
            raise ValueError('CSI batch must be nonempty')
        # There is no previous-agent information or gold output in these features.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.csi_feature_mode == 'full_sequence'):
            cached = [self.encode_parts_batch(index, questions) for index in range(3)]
        cached = [(states.to(self.post_conv_fc[i][1].weight), mask.to(self.adjacency.device))
                  for i, (states, mask) in enumerate(cached)]
        messages, history = None, []  # local state is reset for every question/batch
        for step in range(1, self.csi_steps+1):
            local, parts, masks = [], [], []
            for index, (base, base_mask) in enumerate(cached):
                if messages is None:
                    states, mask = base, base_mask
                else:
                    # The recurrent message is already a representation. Mapping it
                    # back to H_i replaces the last slot without another LLM call.
                    message = self.from_shared[index](messages[:, index])[:, None]
                    states = torch.cat([base[:, :-1], message], dim=1)
                    mask = torch.cat([base_mask[:, :-1], torch.ones_like(base_mask[:, -1:])], dim=1)
                parts.append(states); masks.append(mask)
                local.append(self.local_message_batch(index, states, mask))
            messages, normalized = self.exchange_batch(local)
            event = dict(phase='csi', csi_step=step, batch_size=len(questions),
                         feature_masks=torch.stack(masks, dim=1).tolist())
            if trace:
                for role, states in zip(ROLES, parts):
                    event[role+'_parts'] = states.detach().cpu()
                event['next_messages'] = messages.detach().cpu()
                event['normalized_adjacency'] = normalized.detach().cpu()
            history.append(event)
        return messages, history

    def _row_history(self, history, index):
        per_question = {'feature_masks', 'message_tokens', 'incoming_tokens',
                        'next_messages', 'normalized_adjacency',
                        *(role+'_parts' for role in self.roles)}
        return [{key: value[index] if key in per_question else value
                 for key, value in event.items() if key != 'batch_size'} for event in history]

    def _single_pass_collaborate_batch(self, rows, *, training, trace=False):
        if not rows:
            raise ValueError('Collaboration batch must be nonempty')
        questions = [question_of(row) for row in rows]
        messages, history = self.refine_messages_batch(questions, trace=trace)
        incoming = None
        # Exactly one Planner -> Critic -> Solver pass, after ALL CSI updates.
        for index in range(3):
            soft = self.message_prefix(index, messages[:, index])
            pack = self.batch_pack(index, [stage_prompt(q, index) for q in questions],
                                   incoming=incoming, soft=soft,
                                   answers=[target_of(row, index) for row in rows] if training else None)
            history.append(dict(phase='rmas', rmas_pass=1, agent=ROLES[index], batch_size=len(rows),
                                message_tokens=[p.message_tokens for p in pack.rows],
                                incoming_tokens=[p.incoming_tokens for p in pack.rows]))
            final = index == 2
            if training:
                hidden, loss = self.hidden(index, pack.inputs, attention_mask=pack.attention_mask,
                                           ce_labels=pack.labels if final else None)
                if final:
                    return loss, history
                # Keep upstream teacher-forced latent transfer and its training cap.
                incoming = [self.transfer(index, hidden[b, pack.labels[b].ne(-100)], training=True)
                            for b in range(len(rows))]
            elif final:
                return pack, history
            else:
                states = self.latent_rollout_batch(index, pack)
                incoming = list(self.transfer(index, states, training=False).unbind(0))
        raise RuntimeError('Missing final solver stage')

    def _pool_output_states(self, index, states):
        """Restore output-state Message pooling only for legacy checkpoints."""
        if not states or any(state.ndim != 2 or len(state) == 0 for state in states):
            raise ValueError('Round feedback needs one nonempty output sequence per question')
        if self.csi_feature_mode == 'last_token':
            return torch.stack([state[-1] for state in states])
        inputs, mask = self._pad_sequences(states)
        return self.sequence_poolers[index](inputs,mask)

    def _recursive_csi_update(self, cached, feedback, step, *, trace=False):
        local, parts, masks = [], [], []
        for index,(base,base_mask) in enumerate(cached):
            if feedback is None:
                states,mask = base,base_mask
            else:
                message = (self.from_shared[index](feedback[:,index])
                           if self.message_update == RECURSIVE_MESSAGE_UPDATE else feedback[index])
                message = message.to(base)[:,None]
                states = torch.cat([base[:,:-1],message],dim=1)
                mask = torch.cat([base_mask[:,:-1],torch.ones_like(base_mask[:,-1:])],dim=1)
            parts.append(states);masks.append(mask)
            local.append(self.local_message_batch(index,states,mask))
        messages,normalized = self.exchange_batch(local)
        event = dict(phase='csi',csi_step=step,batch_size=cached[0][0].shape[0],
                     message_source=('masked_zero' if feedback is None else
                                     'previous_gcn' if self.message_update == RECURSIVE_MESSAGE_UPDATE
                                     else 'previous_llm_outputs'),
                     feature_masks=torch.stack(masks,dim=1).tolist())
        if trace:
            for role,states in zip(self.roles,parts):
                event[role+'_parts'] = states.detach().cpu()
            event['next_messages'] = messages.detach().cpu()
            event['normalized_adjacency'] = normalized.detach().cpu()
        return messages,event

    def _cache_recursive_parts(self, questions):
        # Four static parts are encoded once. No reference output enters them.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.csi_feature_mode == 'full_sequence'):
            cached = [self.encode_parts_batch(i, questions) for i in range(len(self.roles))]
        return [(states.to(self.post_conv_fc[i][1].weight),mask.to(self.adjacency.device))
                  for i,(states,mask) in enumerate(cached)]

    def _interleaved_collaborate_batch(self, rows, *, training, trace=False):
        if not rows:
            raise ValueError('Collaboration batch must be nonempty')
        questions = [question_of(row) for row in rows]
        cached = self._cache_recursive_parts(questions)
        feedback,solver_to_planner,history = None,None,[]
        for step in range(1,self.csi_steps+1):
            messages,event = self._recursive_csi_update(cached,feedback,step,trace=trace)
            history.append(event)
            incoming = solver_to_planner
            output_states = [] if self.message_update == LEGACY_RECURSIVE_MESSAGE_UPDATE else None
            for index in range(3):
                final_solver = index == 2 and step == self.csi_steps
                # All agents use teacher forcing in the current Outer schedule.
                # The previous rollout policy is retained for legacy checkpoint restoration.
                targets = ([target_of(row,index) for row in rows]
                           if training and (index < 2 or final_solver or
                               self.intermediate_solver_policy == RECURSIVE_SOLVER_POLICY) else None)
                pack = self.batch_pack(index,[stage_prompt(q,index,step) for q in questions],
                                       incoming=incoming,soft=self.message_prefix(index,messages[:,index]),
                                       answers=targets)
                history.append(dict(phase='rmas',rmas_pass=step,csi_step=step,agent=ROLES[index],
                    batch_size=len(rows),final_solver=final_solver,
                    execution='teacher_forcing' if targets is not None else 'decode' if final_solver else 'latent',
                    message_tokens=[p.message_tokens for p in pack.rows],
                    incoming_tokens=[p.incoming_tokens for p in pack.rows]))
                if final_solver:
                    if training:
                        _,loss = self.hidden(index,pack.inputs,attention_mask=pack.attention_mask,ce_labels=pack.labels)
                        return loss,history
                    return pack,history
                if targets is not None:
                    hidden,_ = self.hidden(index,pack.inputs,attention_mask=pack.attention_mask)
                    states = [hidden[b,pack.labels[b].ne(-100)] for b in range(len(rows))]
                else:
                    rollout = self.latent_rollout_batch(index,pack,differentiable=training)
                    states = list(rollout.unbind(0))
                if output_states is not None:
                    output_states.append(states)
                incoming = [self.transfer(index,state,training=training) for state in states]
            # Solver output still enters the next Planner through its OuterLink.
            # The fifth CSI feature independently recurs from this round's GCN.
            solver_to_planner = incoming
            feedback = (messages if output_states is None else
                        [self._pool_output_states(i,states) for i,states in enumerate(output_states)])
        raise RuntimeError('Missing final Solver')

    def _mixture_collaborate_batch(self, rows, *, training, trace=False):
        if not rows:
            raise ValueError('Collaboration batch must be nonempty')
        questions = [question_of(row) for row in rows]
        targets = [[mixture_target_of(row, i) for row in rows] if training else None
                   for i in range(len(self.roles))]
        cached = self._cache_recursive_parts(questions)
        history, feedback = [], None
        expert_feedback = [None]*self.final_index
        feedback_links = range(self.final_index, len(self.outer_edges))

        for step in range(1, self.csi_steps+1):
            messages, event = self._recursive_csi_update(cached, feedback, step, trace=trace)
            history.append(event)
            expert_outputs = []
            output_states = [] if self.message_update == LEGACY_RECURSIVE_MESSAGE_UPDATE else None
            for index, role in enumerate(self.roles):
                summarizing = index == self.final_index
                final = summarizing and step == self.csi_steps
                # Each expert receives its own feedback from the previous round.
                # The Summarizer receives three independent slots per question.
                incoming = list(zip(*expert_outputs)) if summarizing else expert_feedback[index]
                users = [mixture_stage_prompt(question, index, step, row=row)
                         for question, row in zip(questions, rows)]
                pack = self.batch_pack(index, users, incoming=incoming, answers=targets[index],
                                       soft=self.message_prefix(index, messages[:, index]))
                history.append(dict(phase='rmas', rmas_pass=step, csi_step=step, agent=role,
                    batch_size=len(rows), final_summarizer=final,
                    execution='teacher_forcing' if training else 'decode' if final else 'latent',
                    message_tokens=[p.message_tokens for p in pack.rows],
                    incoming_tokens=[p.incoming_tokens for p in pack.rows]))
                if final:
                    if training:
                        _, loss = self.hidden(index, pack.inputs, attention_mask=pack.attention_mask,
                                              ce_labels=pack.labels)
                        return loss, history
                    return pack, history

                if training:
                    hidden, _ = self.hidden(index, pack.inputs, attention_mask=pack.attention_mask)
                    states = [hidden[b, pack.labels[b].ne(-100)] for b in range(len(rows))]
                else:
                    states = list(self.latent_rollout_batch(index, pack).unbind(0))
                if output_states is not None:
                    output_states.append(states)
                if summarizing:
                    mapped = [self._transfer_links(index, state, links=feedback_links, training=training)
                              for state in states]
                    expert_feedback = list(zip(*mapped))
                else:
                    expert_outputs.append([self.transfer(index, state, training=training) for state in states])
            feedback = (messages if output_states is None else
                        [self._pool_output_states(i, states) for i, states in enumerate(output_states)])
        raise RuntimeError('Missing final Summarizer')

    def _collaborate_batch(self, rows, *, training, trace=False):
        if self.mas_system == MIXTURE:
            return self._mixture_collaborate_batch(rows, training=training, trace=trace)
        if self.collaboration_mode == SINGLE_PASS:
            return self._single_pass_collaborate_batch(rows,training=training,trace=trace)
        return self._interleaved_collaborate_batch(rows,training=training,trace=trace)

    def collaborate(self, row, *, training, trace=False):
        result, history = self._collaborate_batch([row], training=training, trace=trace)
        return (result if training else result.rows[0]), self._row_history(history, 0)

    def training_loss(self, row, *, trace=False):
        return self.collaborate(row, training=True, trace=trace)

    def training_loss_batch(self, rows, *, trace=False):
        """Roundwise CSI and MAS collaboration; only the final agent has Outer CE."""
        if len(rows) == 1:
            return self.training_loss(rows[0], trace=trace)
        return self._collaborate_batch(rows, training=True, trace=trace)

    @torch.no_grad()
    def collaborate_batch(self, rows, *, trace=False):
        pack, history = self._collaborate_batch(rows, training=False, trace=trace)
        return pack, [self._row_history(history, b) for b in range(len(rows))]

    @torch.no_grad()
    def generate(self, row, *, max_new_tokens=2048, trace=False):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.eval()
        pack, history = self.collaborate(row, training=False, trace=trace)
        index = self.final_index
        if pack.inputs.shape[1] + max_new_tokens > self.max_length:
            raise ValueError(f"Final {self.roles[index]} prompt plus generation budget exceeds max_length")
        b = self.backbones[index]
        with self.token_usage.generation(index, b.model):
            generated = b.model.generate(inputs_embeds=pack.inputs,
                                         attention_mask=torch.ones(pack.inputs.shape[:2], device=b.device, dtype=torch.long),
                                         max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                         pad_token_id=b.tokenizer.pad_token_id, return_dict_in_generate=False)
        # Transformers 5.3, decoder-only inputs_embeds returns continuation IDs.
        if generated.shape[1] > max_new_tokens:
            raise RuntimeError("Unexpected generate() contract: received prompt IDs in continuation")
        metadata = generation_end_metadata(generated[0], max_new_tokens=max_new_tokens,
                                           eos_token_id=b.model.generation_config.eos_token_id)
        history[-1].update(metadata)
        count = metadata['generated_tokens']
        self.token_usage.add(index, generated_tokens=count)
        return b.tokenizer.decode(generated[0, :count], skip_special_tokens=True), history

    @torch.no_grad()
    def generate_batch(self, rows, *, max_new_tokens=2048, trace=False):
        """Batch the latent collaboration and a single final-agent generate call."""
        if not rows or max_new_tokens < 1:
            raise ValueError('Inference needs nonempty rows and positive max_new_tokens')
        if len(rows) == 1:
            text, history = self.generate(rows[0], max_new_tokens=max_new_tokens, trace=trace)
            return [text], [history]
        self.eval()
        pack, histories = self.collaborate_batch(rows, trace=trace)
        width = pack.inputs.shape[1]
        index = self.final_index
        if width+max_new_tokens > self.max_length:
            raise ValueError(f'Final {self.roles[index]} prompt plus generation budget exceeds max_length')
        # Decoder-only generate samples the last position, so prompts must be left-padded.
        inputs = torch.stack([F.pad(row.inputs[0], (0, 0, width-row.inputs.shape[1], 0))
                              for row in pack.rows])
        lengths = pack.attention_mask.sum(-1)
        mask = (torch.arange(width, device=inputs.device)[None] >= width-lengths[:, None]).long()
        b = self.backbones[index]
        with self.token_usage.generation(index, b.model):
            generated = b.model.generate(inputs_embeds=inputs, attention_mask=mask,
                                         max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                         pad_token_id=b.tokenizer.pad_token_id, return_dict_in_generate=False)
        if generated.shape[0] != len(rows) or generated.shape[1] > max_new_tokens:
            raise RuntimeError('Unexpected batched generate() continuation shape')
        responses = []
        for tokens, row_pack, history in zip(generated, pack.rows, histories):
            # Include the stopping EOS, but exclude padding after an early-finished row.
            metadata = generation_end_metadata(tokens, max_new_tokens=max_new_tokens,
                                               eos_token_id=b.model.generation_config.eos_token_id)
            count = metadata['generated_tokens']
            responses.append(b.tokenizer.decode(tokens[:count], skip_special_tokens=True))
            history[-1].update(metadata)
            self.token_usage.add(index, generated_tokens=count)
        return responses, histories

    def checkpoint_config(self):
        projected = self.conv_projection == 'pre_conv'
        architecture = (EXPERIENCE_ARCHITECTURE if self.experience_text else
                        PROJECTED_ARCHITECTURE if projected else
                        SEQUENCE_ARCHITECTURE if self.csi_feature_mode == 'full_sequence' else ARCHITECTURE)
        recursive = self.collaboration_mode == INTERLEAVED
        mixture = self.mas_system == MIXTURE
        if recursive:
            architecture = MIXTURE_ARCHITECTURE if mixture else RECURSIVE_ARCHITECTURE
        config = dict(architecture=architecture, mas_system=self.mas_system,
                    hidden_sizes=[b.hidden_size for b in self.backbones],
                    **self.options, feature_order=list(self.features), rmas_passes=self.csi_steps if recursive else 1,
                    conv_output_lengths=list(self.conv_output_lengths),conv_residual_policy=self.conv_residual_policy,
                    conv_hidden_sizes=([self.options['shared_dim']]*len(self.roles) if projected else
                                       [b.hidden_size for b in self.backbones]),
                    feature_projection_order=('fc_then_projected_conv_then_message_fc' if projected else
                                              'native_hidden_conv_then_single_fc'),
                    projection_fc_count_per_agent=2 if projected else 1,
                    message_update=self.message_update if recursive else 'cached_parts_direct_recurrent_message',
                    initial_message='masked_zero', adjacency=self.adjacency.cpu().tolist())
        if recursive:
            feedbacks = MIXTURE_ROUND_FEEDBACKS if mixture else RECURSIVE_ROUND_FEEDBACKS
            config.update(round_feedback=feedbacks[self.message_update],
                          outer_link_edges=[[self.roles[src], self.roles[dst]] for src, dst in self.outer_edges])
            if mixture:
                config['intermediate_summarizer'] = MIXTURE_SUMMARIZER_POLICY
            else:
                config['intermediate_solver'] = self.intermediate_solver_policy
        return config

    def load_trainable_checkpoint(self, path, *, strict_runtime=False, initialize=False):
        payload = torch.load(Path(path).expanduser(), map_location='cpu', weights_only=False)
        if not initialize:
            require_original_training_policy(payload)
        if strict_runtime and payload.get('model_config', {}).get('architecture') == TWO_FC_ARCHITECTURE:
            raise ValueError('v3 two-FC optimizer state cannot be resumed after merging; '
                             'use --init-checkpoint CHECKPOINT --inner-epochs 0 to load weights with fresh optimizers')
        payload = convert_projection_checkpoint(payload)
        expected = self.checkpoint_config()
        saved = payload.get('model_config', {})
        if saved.get('mas_system', SEQUENTIAL) != self.mas_system:
            raise ValueError(f'Checkpoint MAS system must match --mas-system {self.mas_system}')
        remove_fusion = (initialize and saved.get('cross_feature_fusion') is True
                         and not self.cross_feature_fusion)
        if ('cross_feature_fusion' in saved
                and saved['cross_feature_fusion'] != self.cross_feature_fusion and not remove_fusion):
            raise ValueError('Incompatible cross_feature_fusion; removing the flattened feature FC changes '
                             'the model and optimizer layout. Use --init-checkpoint CHECKPOINT for fresh training; '
                             'evaluation restores the checkpoint structure automatically')
        if ('gcn_residual' in saved and saved['gcn_residual'] != self.gcn_residual
                and not initialize):
            raise ValueError('Incompatible gcn_residual; --resume requires the same graph update. '
                             'Use --init-checkpoint CHECKPOINT for fresh training without the GCN input skip; '
                             'evaluation restores the checkpoint setting automatically')
        remove_pointwise = (initialize and saved.get('conv_pointwise') is True and not self.conv_pointwise)
        if ('conv_pointwise' in saved and saved['conv_pointwise'] != self.conv_pointwise
                and not remove_pointwise):
            raise ValueError('Incompatible conv_pointwise. --resume requires matching convolution layers. '
                             'Use --init-checkpoint CHECKPOINT to reuse weights without PWConv')
        if strict_runtime and saved.get('collaboration_mode') != self.collaboration_mode:
            raise ValueError('Single-pass and interleaved training cannot share optimizer progress; '
                             'use --init-checkpoint CHECKPOINT for a new Outer run')
        if (initialize and self.experience_text and saved.get('architecture') in
                (ARCHITECTURE, SEQUENCE_ARCHITECTURE, PROJECTED_ARCHITECTURE)):
            if saved.get('hidden_sizes') != expected['hidden_sizes']:
                raise ValueError('InnerLink initialization requires matching backbone hidden sizes')
            inner = {key.removeprefix('inner_links.'): value
                     for key, value in payload['model_state_dict'].items() if key.startswith('inner_links.')}
            self.inner_links.load_state_dict(inner, strict=True)
            print('Initialize experience CSI: reused three InnerLinks; CSI and OuterLinks start fresh', flush=True)
            return dict(payload, initialization={'reused': 'inner_links', 'fresh': ['CSI', 'outer_links']})
        if (initialize and self.conv_projection == 'pre_conv'
                and saved.get('architecture') in (ARCHITECTURE, SEQUENCE_ARCHITECTURE)):
            if saved.get('hidden_sizes') != expected['hidden_sizes']:
                raise ValueError('InnerLink initialization requires matching backbone hidden sizes')
            inner = {key.removeprefix('inner_links.'): value
                     for key, value in payload['model_state_dict'].items() if key.startswith('inner_links.')}
            self.inner_links.load_state_dict(inner, strict=True)
            print('Initialize projected CSI: reused three InnerLinks; CSI and OuterLinks start fresh', flush=True)
            return dict(payload, initialization={'reused': 'inner_links', 'fresh': ['CSI', 'outer_links']})
        warm_sequence = (initialize and self.csi_feature_mode == 'full_sequence'
                         and payload.get('model_config', {}).get('architecture') == ARCHITECTURE)
        warm_recursive = (initialize and self.collaboration_mode == INTERLEAVED
                          and saved.get('collaboration_mode') == SINGLE_PASS)
        initialized_keys = {'architecture', 'csi_feature_mode', 'csi_prefix_tokens'} if warm_sequence else set()
        if remove_fusion:
            initialized_keys.add('cross_feature_fusion')
        if remove_pointwise:
            initialized_keys.add('conv_pointwise')
        if initialize:
            initialized_keys.update(('adjacency','intermediate_solver','gcn_residual'))
            if self.collaboration_mode == INTERLEAVED:
                initialized_keys.update(('message_update','round_feedback'))
        if warm_recursive:
            initialized_keys |= {'architecture','collaboration_mode','rmas_passes','message_update',
                                 'intermediate_solver','round_feedback','outer_link_edges'}
        runtime = {'latent_steps', 'max_latent_tokens', 'max_length', 'max_feature_tokens', 'gradient_checkpointing',
                   'conv_padding','conv_output_lengths','conv_residual','conv_residual_policy','csi_steps','rmas_passes'}
        for key, value in expected.items():
            if key in initialized_keys:
                continue
            if not strict_runtime and key in runtime:
                continue
            if payload.get('model_config', {}).get(key) != value:
                if key == 'intermediate_solver':
                    raise ValueError('Incompatible intermediate Solver training policy; optimizer resume requires the same policy. '
                                     'Use --init-checkpoint CHECKPOINT --inner-epochs 0 for fresh teacher-forced Outer training')
                if key in ('message_update','round_feedback'):
                    raise ValueError('Incompatible CSI Message recurrence; --resume requires the same Message source. '
                                     'Use --init-checkpoint CHECKPOINT --inner-epochs 0 for a fresh GCN-message Outer run')
                if key == 'adjacency':
                    raise ValueError('Incompatible GCN adjacency; --resume requires the same graph. '
                                     'Use --init-checkpoint CHECKPOINT for fresh training with the new graph')
                if key in ('conv_padding','conv_output_lengths','conv_residual','conv_residual_policy'):
                    raise ValueError(f'Incompatible CSI checkpoint: {key}; --resume requires identical padding and residual settings. '
                                     'Use --init-checkpoint for a fresh convolution experiment')
                raise ValueError(f'Incompatible CSI checkpoint: {key}; use matching --conv-projection and CSI settings. '
                                 'Legacy native-hidden Conv->single-FC checkpoints need --conv-projection native_hidden; '
                                 'use --init-checkpoint CHECKPOINT --inner-epochs 0 to reuse their InnerLinks in a new pre_conv run')
        state = payload['model_state_dict']
        if remove_fusion:
            removed = {f'local_conv.{i}.{name}.{parameter}' for i in range(len(self.roles))
                       for name in ('fusion_norm','feature_fusion') for parameter in ('weight','bias')}
            if not removed.issubset(state):
                raise ValueError('Legacy checkpoint is missing flattened feature-fusion parameters')
            state = {key:value for key,value in state.items() if key not in removed}
            print('Initialize without cross-feature FC: discarded its FC and LayerNorm; '
                  'reused compatible remaining weights with fresh optimizers',flush=True)
        if remove_pointwise:
            removed = {
                f'local_conv.{agent_index}.conv_blocks.{block_index}.pointwise.{parameter}'
                for agent_index, conv in enumerate(self.local_conv)
                for block_index in range(len(conv.conv_blocks))
                for parameter in ('weight', 'bias')
            }
            if not removed.issubset(state):
                raise ValueError('Legacy checkpoint is missing PWConv parameters')
            state = {key: value for key, value in state.items() if key not in removed}
        if warm_sequence or warm_recursive:
            current = self.state_dict()
            missing, extra = set(current)-set(state), set(state)-set(current)
            allowed = tuple(name for enabled,name in ((warm_sequence,'sequence_poolers.'),
                                                       (warm_recursive,'outer_links.2.')) if enabled)
            if extra or any(not key.startswith(allowed) for key in missing):
                raise ValueError('Weight initialization may only add sequence poolers and the S->P OuterLink')
            state = dict(current, **state)
            print('Initialize CSI weights with fresh optimizers: added '+', '.join(allowed),flush=True)
        if initialize:
            # The graph is configuration, not a learned weight: retain the target topology.
            state = dict(state, adjacency=self.adjacency.detach().clone())
        self.load_state_dict(state, strict=True)
        return payload

    def close(self):
        for backbone in self.backbones:
            if hasattr(backbone,'model'):
                del backbone.model


# Public entry point: interleaved by default; checkpoint loading restores legacy mode.
CSISystem = CSIQwenSystem


def build_qwen_system(*, planner_model_path=None, critic_model_path=None, executor_model_path=None,
                      planner_device='cuda:0', critic_device='cuda:0', executor_device='cuda:0',
                      csi_device=None, model_dtype='bfloat16', csi_dtype='float32',
                      d_model=512, local_dims=None, prefix_tokens=80, kernel_size=5,
                      dilations=(1,), max_steps=3, latent_length=32, max_length=4096,
                      max_feature_tokens=2048, gradient_checkpointing=True,
                      trust_remote_code=False, attn_implementation='', checkpoint_path='',
                      csi_feature_mode='full_sequence', csi_prefix_tokens=None, conv_projection='pre_conv',
                      experience_text='', conv_padding='same', conv_residual=True,
                      collaboration_mode=INTERLEAVED, gcn_adjacency=None,
                      intermediate_solver_policy=RECURSIVE_SOLVER_POLICY,
                      message_update=RECURSIVE_MESSAGE_UPDATE,cross_feature_fusion=False,
                      gcn_residual=False, conv_pointwise=False, mas_system=SEQUENTIAL,
                      mixture_model_paths=None, mixture_devices=None):
    if mas_system not in MAS_ROLES:
        raise ValueError(f'Unknown MAS system: {mas_system}')
    roles = MAS_ROLES[mas_system]
    if mas_system == MIXTURE:
        paths = MIXTURE_MODEL_PATHS if mixture_model_paths is None else mixture_model_paths
        devices = ('cuda:0',)*len(roles) if mixture_devices is None else mixture_devices
        if collaboration_mode != INTERLEAVED or intermediate_solver_policy != RECURSIVE_SOLVER_POLICY:
            raise ValueError('Mixture requires interleaved rounds with teacher-forced Summarizer feedback')
    else:
        paths = (planner_model_path, critic_model_path, executor_model_path)
        devices = (planner_device, critic_device, executor_device)
    if len(paths) != len(roles) or not all(paths) or len(devices) != len(roles) or not all(devices):
        raise ValueError(f'{mas_system} needs one model path and device for each of {roles}')
    validate_projection_dims(local_dims, d_model, len(roles))
    if not isinstance(gcn_residual, bool):
        raise ValueError('gcn_residual must be a boolean')
    if collaboration_mode not in (SINGLE_PASS,INTERLEAVED) or max_steps < 1:
        raise ValueError('Invalid collaboration_mode or number of rounds')
    if intermediate_solver_policy not in RECURSIVE_SOLVER_POLICIES:
        raise ValueError('Unknown intermediate Solver training policy')
    if message_update not in RECURSIVE_ROUND_FEEDBACKS:
        raise ValueError('Unknown recursive Message update policy')
    if conv_projection not in ('pre_conv', 'native_hidden'):
        raise ValueError('conv_projection must be pre_conv or native_hidden')
    csi_prefix_tokens = resolve_csi_prefix_tokens(csi_feature_mode, csi_prefix_tokens)
    conv_padding = parse_conv_padding(conv_padding)
    validate_conv_layout(len(EXPERIENCE_FEATURES if experience_text.strip() else FEATURES),
                         kernel_size,dilations,conv_padding,conv_residual)
    backbones = []
    for role, path, device in zip(roles, paths, devices):
        print(f'Loading {role}: {path} on {device}', flush=True)
        backbones.append(Backbone.load(path, device, model_dtype, trust_remote_code, attn_implementation))
    projected = conv_projection == 'pre_conv'
    widths = [d_model]*len(roles) if projected else [b.hidden_size for b in backbones]
    print(f'CSI dimensions: conv_hidden_sizes={widths} '
          f'post_conv_dims={[d_model]*len(roles)} projection_fc_count_per_agent={2 if projected else 1} gcn_dim={d_model} '
          f'feature_mode={csi_feature_mode} csi_prefix_tokens={csi_prefix_tokens} '
          f'conv_projection={conv_projection} conv_padding={conv_padding} conv_residual={conv_residual} '
          f'cross_feature_fusion={cross_feature_fusion} conv_pointwise={conv_pointwise}',flush=True)
    system = CSIQwenSystem(backbones, shared_dim=d_model,
                           csi_steps=max_steps, latent_steps=latent_length, max_latent_tokens=prefix_tokens,
                           max_length=max_length, max_feature_tokens=max_feature_tokens,
                           kernel_size=kernel_size, dilations=dilations,
                           gradient_checkpointing=gradient_checkpointing,
                           csi_device=csi_device, csi_dtype=csi_dtype,
                           csi_feature_mode=csi_feature_mode, csi_prefix_tokens=csi_prefix_tokens,
                           conv_projection=conv_projection, experience_text=experience_text,
                           conv_padding=conv_padding,conv_residual=conv_residual,collaboration_mode=collaboration_mode,
                           gcn_adjacency=gcn_adjacency,intermediate_solver_policy=intermediate_solver_policy,
                           message_update=message_update,cross_feature_fusion=cross_feature_fusion,
                           gcn_residual=gcn_residual,conv_pointwise=conv_pointwise,mas_system=mas_system)
    print(f'Conv raw output lengths: {system.conv_output_lengths}; '
          +(f'residual keeps {len(system.features)} feature positions' if conv_residual else
            f'no residual; gated pooling reads {system.conv_output_lengths[-1]} positions'),flush=True)
    print(f'CSI feature order: {list(system.features)}', flush=True)
    print(f'GCN adjacency [receiver, sender]: {system.adjacency.detach().cpu().tolist()} (self-loops added by GCN)',flush=True)
    print(f'GCN input residual: {system.gcn_residual}',flush=True)
    print(f'Collaboration: {mas_system}, {system.collaboration_mode}; passes='
          f'{system.csi_steps if system.collaboration_mode == INTERLEAVED else 1}',flush=True)
    print(f'Inference KV cache: latent=True, final_{roles[-1]}=True; training/feature encoding=False',flush=True)
    if system.collaboration_mode == INTERLEAVED:
        source = next(name for name,policy in MESSAGE_SOURCE_POLICIES.items() if policy == system.message_update)
        print(f'CSI Message source: {source}; update={system.message_update}',flush=True)
        policy = MIXTURE_SUMMARIZER_POLICY if mas_system == MIXTURE else system.intermediate_solver_policy
        print(f'Outer intermediate {roles[-1].capitalize()} training policy: {policy}',flush=True)
    if checkpoint_path:
        system.load_trainable_checkpoint(checkpoint_path)
    return system


def question_of(row):
    q = str(row.get("question") or row.get("problem") or row.get("prompt") or "").strip()
    if not q:
        raise ValueError("Record has no question/problem/prompt")
    return q


def target_of(row, index):
    key = ("plan", "refined_plan", "answer")[index]
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Training requires {key}")
    return value


def mixture_target_of(row, index):
    fields = MIXTURE_TARGET_FIELDS[index]
    for field in fields:
        value = row.get(field)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            text = str(value).strip()
            if text:
                return text
    raise ValueError(f'Mixture training requires {MIXTURE_ROLES[index]} target ({", ".join(fields)})')
