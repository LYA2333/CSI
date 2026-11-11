"""Pretrain local CSI prefixes and InnerLinks, then train CSI/Outer links jointly."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch

from csi_qwen import (build_qwen_system, validate_projection_dims,
                      resolve_csi_prefix_tokens, active_inner_losses, DEFAULT_EXPERIENCE_TEXT,INTERLEAVED,
                      RECURSIVE_SOLVER_POLICY,LEGACY_RECURSIVE_SOLVER_POLICY,
                      RECURSIVE_MESSAGE_UPDATE,MESSAGE_SOURCE_POLICIES,
                      SEQUENTIAL, MIXTURE, MAS_MODEL_ROLES, MIXTURE_ROLES, MIXTURE_MODEL_PATHS)
from data_utils import CodeTrainingDataset, load_records
from conv_graph_mas_demo import parse_conv_padding, validate_conv_layout

ROOT = Path(__file__).resolve().parent
TRAINING_SCHEDULE = 'inner_ce_align_local_csi_then_native_hidden_single_rmas_final_ce_v7'
SEQUENCE_TRAINING_SCHEDULE = 'inner_ce_align_local_csi_then_sequence_slots_single_rmas_final_ce_v7'
PROJECTED_TRAINING_SCHEDULE = 'inner_ce_align_local_csi_then_pre_conv_single_rmas_final_ce_v7'
EXPERIENCE_TRAINING_SCHEDULE = 'inner_ce_align_local_csi_then_experience_single_rmas_final_ce_v8'
INNER_TRAINING_OBJECTIVE = 'local_csi_ce_plus_detached_innerlink_alignment_v1'
LEGACY_RECURSIVE_TRAINING_SCHEDULE = 'local_csi_ce_align_then_interleaved_csi_pcs_no_gold_solver_feedback_final_ce_v9'
OUTPUT_MESSAGE_TRAINING_SCHEDULE = 'local_csi_ce_align_then_interleaved_pcs_teacher_forcing_final_solver_ce_v10'
RECURSIVE_TRAINING_SCHEDULE = 'local_csi_ce_align_then_interleaved_gcn_message_pcs_teacher_forcing_final_solver_ce_v11'
MIXTURE_TRAINING_SCHEDULE = 'local_csi_ce_align_then_interleaved_mixture_teacher_forcing_final_summarizer_ce_v1'
# Exact file-based predecessor: same experience text, network and optimizer layout.
INLINE_EXPERIENCE_PARENT_SOURCES = {
    'csi_qwen.py': '7f6ad65f509e04f45a8a8f19f6152ebb775ef27b6e0a78ff4b10ab3cc6ba4f69',
    'train_csi.py': 'cd57d28382657c9d8b9df7236821b866a83a38872d457a85b27fdd579b0ab062',
    'experience.txt': 'd28fdfa2ec0493eddaad5753fbb7d41262fb3eea81ba9af6ec41293a58344be1',
}
# Verified v7 parent: zero-weight terms were evaluated but excluded from loss.
# Parameter layout, objective and optimizer states are unchanged.
ZERO_WEIGHT_LOSS_PARENT_SOURCES = {
    'csi_qwen.py': '824be5aae37bb4cb1a7e60bf12e998ea5de2c374bc9803d1482d95e26dc47418',
    'train_csi.py': '02ff0172aa9214a15af106f374892a1c33dd4b5cd0cba6f78e9776fe23107c79',
}
# Immediate predecessor: position-counter naming only; training is unchanged.
TOKEN_POSITION_NAMING_PARENT_SOURCES = {
    'csi_qwen.py': '0c5ca6cc322d5b2860a248c44eff63790f0c8c23dc41cb076bc624f05ff26db4',
    'train_csi.py': '6694e975f7efc995630902fe727ca4c6996520647e3f7da235ed5491a21f583d',
}
# Verified pre-accounting sources: only usage instrumentation/logging changed.
TOKEN_USAGE_PARENT_SOURCES = {
    'csi_qwen.py': '4619f5e8f78f1a548d9b8c24462692511e796c9f215510f1bbe31a88ed7a6cc0',
    'train_csi.py': 'a0e1d513b5f0b8bbef422f40af5b68994349979349c61b52970b0ca119f41a17',
}
V5_PARENT_SOURCES = {
    'csi_qwen.py': 'f50d2e9ad813bc5c9dfd30ece4d26c09d0020e55a4b77094577a65f1cbeb73b2',
    'train_csi.py': '33dbc5613b33e693fee94b2a4dd999b41684db63621c02143ae4fe905ef332c9',
    'evaluate_math500_csi.py': 'bbea78b03a906768f5c5d825f211eb23813cc2a5449045e808aa39e143c28bdb',
}
# This exact predecessor differs only in JSONL output; all model/data checks remain.
MASK_LOGGING_PARENT_SOURCES = {
    'csi_qwen.py': TOKEN_USAGE_PARENT_SOURCES['csi_qwen.py'],
    'train_csi.py': '7140582406a3155d6e56a928e0ba711685e609cffe698ab1e95461f5fb9c2bf3',
    'evaluate_math500_csi.py': '391350753b6a81d794e9c2ec3bbb535205db3fe3394b4bba1b0b2ef0e8e4d3e5',
}
# Verified immediate v4 predecessor: only last_token mode may restore its state.
V4_PARENT_SOURCES = {
    'csi_qwen.py': '3b27fff7129b24e5e631800cd9bf8645389a4dc414112365eea8489ad0b74011',
    'train_csi.py': '0a981154057b7cf6923b8b086d02ac4305f469815145d454cb004caa0a416b40',
    'evaluate_math500_csi.py': '08f86074c65bbfcf0b13babeffdc2630b47b7f8c85dbfa5a6774cdab7d9abcba',
}
# This exact predecessor differs only in epoch-local logging. Other sources
# and all existing architecture/data/argument checks must still match.
EPOCH_LOGGING_PARENT_TRAINER_SHA256 = frozenset({
    '405972431f7307f54687baa894068d9bea1978a2a14d54b8002aad341ba16866',
})
# Removing an unused evaluation CLI option leaves training and model computation unchanged.
EVAL_CLI_PARENT_SOURCES = {
    'train_csi.py': '71ef54ef12687a51a6c8c08e3b28b4f730394687b2c98ce6a3eff8ede1993d17',
    'evaluate_math500_csi.py': 'fba5bc8b5ff7b6e480c274ba501cf32a476bee7cf93db22118bd5e478d1ab0bd',
}
# Standalone evaluation defaults do not change training or saved model weights.
EVAL_DEFAULTS_PARENT_SOURCES = {
    'train_csi.py': 'd861e32a6176482d6ecef180bffda3c00dfd564117385654c352fce76299eca6',
    'evaluate_math500_csi.py': '53b2c3764ae8562da563cfd4d3ea6c4a61f58d80c8ee2fd2b4d4fa30fd6e6f28',
}


# Exact verified pre-experiment snapshot; new settings/data/config guards still apply.
V3_BASE_SOURCE_SHA256 = {'csi_qwen.py': 'fbfc97538dcc04fe90187f8810631ff32a4be7c34fbc4b60343436a28320ee33', 'conv_graph_mas_demo.py': '703d6db0574c35722586d73a3911cb0f25a21691deddd619300c20b0f79b7e6d', 'train_csi.py': '6f68771a335dbda14f9272a8397c0c5ba32058a18e66ef623c67e52d24d00eec', 'data_utils.py': 'ca2d83c7c0e6b8685a131347f9742d86cfe8e00e38a0d381a237580a8e912c81', 'math_grading.py': '8ed6dcb1a63fe81aa7893fe8c84e59fd3ead7c5a062f820670f83235c4d12d6c', 'evaluate_math500_csi.py': '0f0c8ee3e6b6f78f5edb320038570682cf85422acd148a50f652791151cf3fe2', 'vendor/recursivemas/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'vendor/recursivemas/answer_utils.py': '7e429afa46a2b447a7e05bef91fb567d799004fa7648880a7c1d31db7569eee0', 'vendor/recursivemas/model.py': 'e37161f7a31fc6c40ff096b3b05565ab4ae58e24e9def61853560dabd42390f4', 'vendor/recursivemas/prompts.py': '5e73260be7fcc77ca7ec089367271c257a700900aee8e29c67afc6e250c60e34'}


# Only the total last_loss display changed; exact predecessor checkpoints remain compatible.
LOSS_LOGGING_PARENT_SOURCES = {'train_csi.py': 'a67b7c88e25de1b5554b83a1decf64b965bcbddb8e3a4c3b398ed899da4f3a94'}


# Exact pre-switch sources: enabled residuals retain identical parameters and computation.
CONV_RESIDUAL_PARENT_SOURCES = {'csi_qwen.py': 'e94a8164ddb0536674fb90bd037b9ec251ff820901d0ec776bbdeda255aa9608', 'conv_graph_mas_demo.py': 'e14440e8f761cd909dcb52f6138e7971dc61308bd96377c4ff6ccaff1fe1d5df', 'train_csi.py': '638d9dbdaf6415b462db2549cb0d1db4bce6ad4fbff3ef33cb30bccce2884c55', 'data_utils.py': '0b866d34839f6e972e5be2990e24d89e8c094451fa78b31acc36456a4bafce32', 'math_grading.py': '7a8036cd31092e86827d17cf4a4a7fbf8a99177397e906284a34eab68040e56e', 'evaluate_math500_csi.py': '896c3f518e9b11b031a8fb8329d172e52225c45ad6ac40c712a10587f1505888', 'vendor/recursivemas/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'vendor/recursivemas/answer_utils.py': '7e429afa46a2b447a7e05bef91fb567d799004fa7648880a7c1d31db7569eee0', 'vendor/recursivemas/model.py': 'e37161f7a31fc6c40ff096b3b05565ab4ae58e24e9def61853560dabd42390f4', 'vendor/recursivemas/prompts.py': '5e73260be7fcc77ca7ec089367271c257a700900aee8e29c67afc6e250c60e34'}
CONV_RESIDUAL_PARENT_SOURCE_SETS = (CONV_RESIDUAL_PARENT_SOURCES,
    dict(CONV_RESIDUAL_PARENT_SOURCES, **LOSS_LOGGING_PARENT_SOURCES))
PARALLEL_INNER_PARENT_SOURCES = {
    'csi_qwen.py': 'b339bb0fc69368636b2a14a3fbcdb32d4d469f163ca5022ede0cc1d894f0c5d8',
    'train_csi.py': 'fc3d4a478c768f920066e0c2a9e7555b76d3bf876ac7e31f0865c870e8571a1d',
}
NO_KV_PARENT_SOURCES = {
    'csi_qwen.py': 'e7ca292f6b231c0737ac81fd488bdc088ada93817fd39819fa6f9f1d430b3052',
    'evaluate_math500_csi.py': '6486ab03409eba6b285586474314040b4fa30071cfe3ee3c829197e260dc835e',
    'train_csi.py': '3f9d9d76244a45a330d0f409d406262ca5dcc8f922b95fe4a59473d1aeb2c65c',
}
ALL_KV_PARENT_SOURCES = {
    'csi_qwen.py': 'c4d7d3fab802a502d0474b49f7e1466e9b0aaa85bb78b743fdad450ac489f2c8',
    'train_csi.py': '29a8c0f7ff45dce244ac3a1dfe1a35f1629aa087938dec7acde1a6817963b0b3',
}
GCN_MESSAGE_PARENT_SOURCES = {
    'csi_qwen.py': '6ef6fc51c8ddb62213c31d6c8e4ea9554bd53400ea5b5913cd33036c9d4a411d',
    'train_csi.py': '415feabd398bba92f361e24282e496ed2e16bf8e8bc3573a8736b808fd8851ca',
    'evaluate_math500_csi.py': '061c21333168f262c0f7a612bca5859ec7d0e57fff937430ec6dbca9ec6740ed',
}
# Exact Sequential sources in the v3 directory used to build v4.
V3_MAS_PARENT_SOURCES = {
    'csi_qwen.py': '40e13deb57dc3309f8dd53aef2d964f7d52d10a476f274c8917da58aa6a16986',
    'train_csi.py': 'b6148b69de38fa3185b36aa3415d280a518b69c97312c443b1b2be502fb2cae1',
    'data_utils.py': '0b866d34839f6e972e5be2990e24d89e8c094451fa78b31acc36456a4bafce32',
    'evaluate_math500_csi.py': 'd9c0b7e76fb5ec46d49e11ebaec0eac47da774f07f52b35e6da41a5b0d2effe3',
}


def add_model_arguments(parser, *, checkpoint_defaults=False):
    # Evaluation restores saved model paths/devices unless explicitly overridden.
    parser.add_argument('--mas-system', '--multi-agent-system', choices=tuple(MAS_MODEL_ROLES),
                        default=None if checkpoint_defaults else SEQUENTIAL,
                        help=('Restore the system from the checkpoint; an explicit value must match'
                              if checkpoint_defaults else 'Agent topology (default: %(default)s)'))
    sequential_paths = ('pre-trained/Qwen3-1.7B', 'pre-trained/Llama-3.2-1B-Instruct',
                        'pre-trained/Qwen2.5-Math-1.5B-Instruct')
    for mas_system, paths in ((SEQUENTIAL, sequential_paths), (MIXTURE, MIXTURE_MODEL_PATHS)):
        for role, path in zip(MAS_MODEL_ROLES[mas_system], paths):
            flag = role.replace('_', '-')
            parser.add_argument(f'--{flag}-model-path', default=None if checkpoint_defaults else path)
            parser.add_argument(f'--{flag}-device', default=None if checkpoint_defaults else 'cuda:0')
    parser.add_argument('--csi-device', default=None if checkpoint_defaults else 'cuda:0')
    parser.add_argument('--model-dtype', choices=['auto','float32','bfloat16','float16'],
                        default=None if checkpoint_defaults else 'bfloat16')
    parser.add_argument('--csi-dtype', choices=['float32','bfloat16','float16'],
                        default=None if checkpoint_defaults else 'float32')
    parser.add_argument('--trust-remote-code', action=argparse.BooleanOptionalAction,
                        default=None if checkpoint_defaults else False)
    parser.add_argument('--attn-implementation', default=None if checkpoint_defaults else '')


def build_parser():
    parser = argparse.ArgumentParser(description='Train ConvRSI with Sequential or Mixture multi-agent collaboration')
    add_model_arguments(parser)
    data = parser.add_mutually_exclusive_group()
    data.add_argument('--train-jsonl', default='')
    data.add_argument('--train-parquet', default='')
    parser.add_argument('--output-dir', type=Path, default=Path('csi_checkpoints'))
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument('--resume', default='', help='Matching architecture checkpoint with optimizer state, e.g. csi_latest.pt')
    restart.add_argument('--init-checkpoint', default='',
                         help='Fresh optimizers/progress; architecture/experience changes reuse only InnerLinks; use --inner-epochs 0 to skip Inner training')
    parser.add_argument('--d-model', type=int, default=512,
                        help='Projected convolution and shared GCN width; default 512, independent of LLM hidden sizes')
    parser.add_argument('--conv-projection', choices=['pre_conv','native_hidden'], default='pre_conv',
                        help='pre_conv: project H_i to d-model before convolution; native_hidden: legacy v4/v5 computation')
    parser.add_argument('--csi-feature-mode', choices=['full_sequence','last_token'], default='full_sequence',
                        help='full_sequence: trainable pooling of all hidden states; last_token: original one-token baseline')
    parser.add_argument('--csi-prefix-tokens', type=int, default=8,
                        help='Fixed slots per part, message and CSI soft prefix; default 8; last_token requires 1')
    parser.add_argument('--no-experience', action='store_true',
                        help='Disable the built-in general experience and use the legacy four CSI features')
    parser.add_argument('--local-dims', type=int, nargs='+', default=None,
                        help='Legacy alias: one --d-model value per active agent; omit to use the shared convolution width')
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--dilations', type=int, nargs='+', default=[1])
    parser.add_argument('--conv-padding','--padding',dest='conv_padding',type=parse_conv_padding,default='same',
                        help='same keeps sequence length; integer sets symmetric padding, e.g. 0. Residuals are controlled by --conv-residual')
    parser.add_argument('--conv-residual',action=argparse.BooleanOptionalAction,default=True,
                        help='Keep the convolution input skip (default); --no-conv-residual pools the raw convolution output')
    parser.add_argument('--max-steps', type=int, default=3,
                        help='Number of complete CSI and multi-agent collaboration rounds')
    parser.add_argument('--message-source',choices=list(MESSAGE_SOURCE_POLICIES),default='gcn',
                        help='Fifth feature: previous GCN output (default); agent_output restores the legacy experiment')
    parser.add_argument('--latent-length', type=int, default=32,
                        help='Inference latent steps per non-final stage; current Outer training uses teacher forcing for all agents')
    parser.add_argument('--max-prefix-tokens','--prefix-tokens', dest='prefix_tokens', type=int, default=80,
                        help='Original RMAS TRAIN transfer cap')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--max-feature-tokens', type=int, default=1024)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--inner-epochs','--inner-warmup-epochs', dest='inner_epochs', type=int, default=1)
    parser.add_argument('--inner-parallel', action=argparse.BooleanOptionalAction, default=False,
                        help='Train independent Inner agents concurrently; one CUDA stream per agent on its --*-device')
    parser.add_argument('--inner-learning-rate', type=float, default=5e-4)
    parser.add_argument('--inner-max-length', type=int, default=2048)
    parser.add_argument('--inner-ce-weight', type=float, default=0.,
                        help='Weight of agent target CE; 0 skips CE/LM head and prefix backward graph')
    parser.add_argument('--inner-align-weight', type=float, default=1.,
                        help='Weight of detached-hidden InnerLink alignment; 0 skips the entire alignment branch')
    parser.add_argument('--inner-cos-weight', type=float, default=1.,help='Cosine alignment weight; 0 skips cosine')
    parser.add_argument('--inner-mse-weight', type=float, default=0.0,help='MSE alignment weight; 0 skips MSE')
    parser.add_argument('--inner-max-updates', type=int, default=-1)
    for roles in MAS_MODEL_ROLES.values():
        for role in roles:
            parser.add_argument(f'--{role.replace("_", "-")}-inner-path', default='',
                                help='Optional compatible ln_res_adapter adapter.pt')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Outer questions per true minibatch; default 2')
    parser.add_argument('--inner-batch-size', type=int, default=2,
                        help='Inner minibatch size, independent of --batch-size; default 2')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--lr-scheduler', choices=['cosine','constant'], default='cosine')
    parser.add_argument('--warmup-updates', type=int, default=10)
    parser.add_argument('--min-lr-ratio', type=float, default=0.01,
                        help='Cosine final LR / initial LR for each parameter group; default 0.01; 0 restores zero floor')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.)
    parser.add_argument('--max-updates', type=int, default=-1)
    parser.add_argument('--log-every', type=int, default=16,
                        help='Optimizer updates between logs within each epoch; also log the epoch tail')
    parser.add_argument('--save-every', type=int, default=10000, help='Optimizer updates between csi_latest.pt saves; 0=epoch/stage only')
    parser.add_argument('--outer-validation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--validation-jsonl', type=Path, default=Path('data/math500.jsonl'))
    parser.add_argument('--validation-batch-size', type=int, default=16,
                        help='Questions per true generation batch during Outer validation')
    parser.add_argument('--validation-max-new-tokens', type=int, default=2048)
    parser.add_argument('--validation-limit', type=int, default=0, help='0 evaluates every validation record')
    parser.add_argument('--validation-grader', choices=['auto','math_verify','normalized'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def validate_args(args):
    model_roles = MAS_MODEL_ROLES[args.mas_system]
    if not args.train_jsonl and not args.train_parquet:
        if args.mas_system == MIXTURE:
            raise ValueError('Mixture requires --train-jsonl or --train-parquet with expert and Summarizer targets')
        args.train_jsonl = str(ROOT/'data/sequential_math.jsonl')
    if args.inner_batch_size is None:
        args.inner_batch_size = args.batch_size
    args.csi_prefix_tokens = resolve_csi_prefix_tokens(args.csi_feature_mode, args.csi_prefix_tokens)
    validate_projection_dims(args.local_dims, args.d_model, len(model_roles))
    args.conv_padding = parse_conv_padding(args.conv_padding)
    validate_conv_layout(4 if args.no_experience else 5,args.kernel_size,args.dilations,args.conv_padding,args.conv_residual)
    if args.init_checkpoint and any(getattr(args,role+'_inner_path') for role in model_roles):
        raise ValueError('Choose --init-checkpoint or individual --*-inner-path adapters')
    for key in ('epochs','d_model','max_steps','latent_length','prefix_tokens','max_length','max_feature_tokens',
                'inner_max_length','gradient_accumulation_steps','log_every','validation_max_new_tokens',
                'batch_size','inner_batch_size','validation_batch_size'):
        if getattr(args,key) < 1:
            raise ValueError(f'{key} must be positive')
    if min(args.inner_epochs,args.warmup_updates,args.save_every,args.validation_limit) < 0:
        raise ValueError('Epochs, warmup, save interval and validation limit must be non-negative')
    if not 0. <= args.min_lr_ratio <= 1.:
        raise ValueError('min_lr_ratio must be finite and between 0 and 1')
    if args.model_dtype == 'float16' or args.csi_dtype == 'float16':
        raise ValueError('Training supports bfloat16/float32; unscaled float16 is unsupported')
    for key in ('max_updates','inner_max_updates'):
        if getattr(args,key) == 0 or getattr(args,key) < -1:
            raise ValueError(f'{key} must be -1 or positive')
    if min(args.learning_rate,args.inner_learning_rate,args.max_grad_norm) <= 0:
        raise ValueError('Learning rates and gradient clipping must be positive')
    active_inner_losses(args.inner_ce_weight,args.inner_align_weight,args.inner_cos_weight,args.inner_mse_weight)
    for device in {args.csi_device, *(getattr(args, role+'_device') for role in model_roles)}:
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; pass cpu for the selected agents and --csi-device')
            with torch.cuda.device(device):
                if 'bfloat16' in (args.model_dtype,args.csi_dtype) and not torch.cuda.is_bf16_supported():
                    raise ValueError(f'{device} lacks bfloat16 support; use float32')


def construct_system(args):
    names = ('csi_device','model_dtype','csi_dtype','d_model','local_dims','prefix_tokens',
             'kernel_size','dilations','max_steps','latent_length','max_length','max_feature_tokens',
             'gradient_checkpointing','trust_remote_code','attn_implementation',
             'csi_feature_mode','csi_prefix_tokens','conv_projection')
    mas_system = getattr(args, 'mas_system', SEQUENTIAL)
    if mas_system == MIXTURE:
        models = dict(mixture_model_paths=tuple(getattr(args, role+'_model_path') for role in MIXTURE_ROLES),
                      mixture_devices=tuple(getattr(args, role+'_device') for role in MIXTURE_ROLES))
    else:
        models = {role+suffix: getattr(args, role+suffix)
                  for role in MAS_MODEL_ROLES[SEQUENTIAL] for suffix in ('_model_path', '_device')}
    # Inference supplies the exact checkpoint text, including empty legacy text.
    # Fresh training uses the code constant; no external file or data row is read.
    experience_text = getattr(args,'experience_text',None)
    if experience_text is None:
        experience_text = '' if getattr(args,'no_experience',False) else DEFAULT_EXPERIENCE_TEXT
    return build_qwen_system(**{name:getattr(args,name) for name in names}, **models,
                             mas_system=mas_system, experience_text=experience_text,
                             conv_padding=getattr(args,'conv_padding','same'),
                             conv_residual=getattr(args,'conv_residual',True),
                             conv_pointwise=getattr(args,'conv_pointwise',False),
                             cross_feature_fusion=getattr(args,'cross_feature_fusion',False),
                             gcn_residual=getattr(args,'gcn_residual',False),
                             collaboration_mode=getattr(args,'collaboration_mode',INTERLEAVED),
                             gcn_adjacency=getattr(args,'gcn_adjacency',None),
                             intermediate_solver_policy=getattr(args,'intermediate_solver_policy',RECURSIVE_SOLVER_POLICY),
                             message_update=MESSAGE_SOURCE_POLICIES[getattr(args,'message_source',None) or 'gcn'])


def planned_optimizer_updates(samples, epochs, accumulation, cap=-1, *, batch_size=1):
    total = epochs*math.ceil(samples/(batch_size*accumulation))
    return min(total,cap) if cap > 0 else total


def build_lr_scheduler(optimizer, total, warmup, kind='cosine', min_lr_ratio=0.01):
    if not 0. <= min_lr_ratio <= 1.:
        raise ValueError('min_lr_ratio must be finite and between 0 and 1')
    warmup = min(warmup,max(0,total-1))
    def multiplier(update):
        if warmup and update < warmup:
            return (update+1)/warmup
        if kind == 'constant':
            return 1.
        progress = min(1.,max(0.,(update-warmup)/max(1,total-warmup)))
        return min_lr_ratio+(1-min_lr_ratio)*0.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [ROOT/name for name in ('csi_qwen.py','conv_graph_mas_demo.py','train_csi.py',
                                  'data_utils.py','math_grading.py','evaluate_math500_csi.py')]
    paths += sorted((ROOT/'vendor/recursivemas').glob('*.py'))
    return {str(p.relative_to(ROOT)):_digest(p) for p in paths}


def resume_sources_compatible(saved, current, *, mas_system=SEQUENTIAL):
    if not isinstance(saved,dict) or not isinstance(current,dict) or 'train_csi.py' not in current:
        return False
    if saved == current:
        return True
    return mas_system == SEQUENTIAL and (
        _sequential_sources_compatible(saved, current)
        or _sequential_sources_compatible(saved, dict(current, **V3_MAS_PARENT_SOURCES))
    )


def _sequential_sources_compatible(saved, current):
    if saved == current or saved == V3_BASE_SOURCE_SHA256 or saved in CONV_RESIDUAL_PARENT_SOURCE_SETS:
        return True
    if saved == dict(current, **PARALLEL_INNER_PARENT_SOURCES):
        return True
    if saved == dict(current, **NO_KV_PARENT_SOURCES):
        return True
    if saved == dict(current, **ALL_KV_PARENT_SOURCES):
        return True
    if saved == dict(current, **GCN_MESSAGE_PARENT_SOURCES):
        return True
    if saved == dict(current,**INLINE_EXPERIENCE_PARENT_SOURCES):
        return True
    for parent in (V4_PARENT_SOURCES, V5_PARENT_SOURCES, MASK_LOGGING_PARENT_SOURCES,
                   TOKEN_USAGE_PARENT_SOURCES, TOKEN_POSITION_NAMING_PARENT_SOURCES,
                   EVAL_CLI_PARENT_SOURCES, EVAL_DEFAULTS_PARENT_SOURCES, ZERO_WEIGHT_LOSS_PARENT_SOURCES,
                   LOSS_LOGGING_PARENT_SOURCES):
        if saved.keys() == current.keys() and all(saved[name] == parent.get(name,digest)
                                                 for name,digest in current.items()):
            return True
    return (saved.keys() == current.keys()
            and saved.get('train_csi.py') in EPOCH_LOGGING_PARENT_TRAINER_SHA256
            and all(saved[name] == TOKEN_USAGE_PARENT_SOURCES.get(name,digest)
                    for name,digest in current.items() if name != 'train_csi.py'))


def write_json(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path,system,outer_optimizer,*,inner_optimizers,outer_scheduler,
                    inner_schedulers,progress,args,training_data_sha256):
    phase = progress['phase']
    active = progress['outer'] if phase == 'outer' else progress['inner'][system.roles[progress['role_index']]]
    payload = dict(model_state_dict={k:v.detach().cpu() for k,v in system.state_dict().items()},
                   model_config=system.checkpoint_config(),training_schedule=training_schedule(system),
                   inner_training_objective=INNER_TRAINING_OBJECTIVE,
                   optimizer_state_dict=outer_optimizer.state_dict(),
                   inner_optimizer_state_dicts={r:o.state_dict() for r,o in inner_optimizers.items()},
                   lr_scheduler_state_dict=outer_scheduler.state_dict(),
                   inner_lr_scheduler_state_dicts={r:s.state_dict() for r,s in inner_schedulers.items()},
                   training_phase=phase,epoch=active['epoch'],update=progress['outer']['update'],
                   inner_updates={r:v['update'] for r,v in progress['inner'].items()},progress=progress,
                   token_usage=system.token_usage.snapshot(),
                   training_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                   training_data_sha256=training_data_sha256,source_sha256=source_hashes(),
                   rng_state=torch.get_rng_state(),python_rng_state=random.getstate(),
                   cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp'); torch.save(payload,temp); temp.replace(path)


def training_schedule(system):
    if getattr(system, 'mas_system', SEQUENTIAL) == MIXTURE:
        return MIXTURE_TRAINING_SCHEDULE
    if getattr(system,'collaboration_mode',None) == INTERLEAVED:
        if system.intermediate_solver_policy == LEGACY_RECURSIVE_SOLVER_POLICY:
            return LEGACY_RECURSIVE_TRAINING_SCHEDULE
        return (RECURSIVE_TRAINING_SCHEDULE if system.message_update == RECURSIVE_MESSAGE_UPDATE
                else OUTPUT_MESSAGE_TRAINING_SCHEDULE)
    if getattr(system,'experience_text',''):
        return EXPERIENCE_TRAINING_SCHEDULE
    if system.conv_projection == 'pre_conv':
        return PROJECTED_TRAINING_SCHEDULE
    return SEQUENCE_TRAINING_SCHEDULE if system.csi_feature_mode == 'full_sequence' else TRAINING_SCHEDULE


def _progress():
    return dict(epoch=0,cursor=0,update=0,loss_sum=0.,samples=0,last_loss=0.)


@contextmanager
def inner_context(system,max_length):
    previous = system.max_length
    system.max_length = max_length
    try:
        yield
    finally:
        system.max_length = previous


def optimize_epoch(system,rows,optimizer,scheduler,progress,args,*,epochs,cap,salt,label,loss_fn,on_save,
                   batch_size=1,batch_loss_fn=None,on_log=None,csi_steps=None,
                   max_updates_per_call=None,manage_mode=True,report_usage=True,recent_losses=None):
    """One epoch/partial epoch; cursors are saved only at optimizer boundaries."""
    order = list(range(len(rows)))
    random.Random(args.seed+salt+progress['epoch']).shuffle(order)
    cursor = progress['cursor']; accum = args.gradient_accumulation_steps
    window_size = batch_size*accum
    message_steps = system.csi_steps if csi_steps is None else csi_steps
    if batch_size < 1 or (batch_size > 1 and batch_loss_fn is None):
        raise ValueError('Batch size must be positive; batched training requires a true batch loss function')
    if cursor < 0 or cursor > len(rows) or (cursor < len(rows) and cursor % window_size):
        raise ValueError('Checkpoint cursor is not an optimizer boundary')
    if max_updates_per_call is not None and max_updates_per_call < 1:
        raise ValueError('max_updates_per_call must be positive')
    if manage_mode:
        system.train()
    # Recover the epoch-local count from the saved optimizer-boundary cursor.
    # The global update remains continuous for schedulers, caps and checkpoints.
    epoch_update = math.ceil(cursor/window_size)
    usage = getattr(system,'token_usage',None)
    usage = usage if report_usage and usage is not None and usage.phase is not None else None
    if usage is not None:
        progress.setdefault('epoch_token_start',usage.phase_totals())
    recent = [] if recent_losses is None else recent_losses
    updates_this_call = 0
    while cursor < len(rows) and (cap < 0 or progress['update'] < cap):
        group = order[cursor:cursor+window_size]
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.
        update_components = {}
        for start in range(0,len(group),batch_size):
            micro_indices = group[start:start+batch_size]
            micro_rows = [rows[index] for index in micro_indices]
            result = loss_fn(micro_rows[0]) if len(micro_rows) == 1 else batch_loss_fn(micro_rows)
            loss, components = result if isinstance(result,tuple) else (result,{})
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{label}: non-finite loss at original rows {micro_indices}; no row was skipped')
            # A short final microbatch must not receive the weight of a full one.
            weight = len(micro_rows)/len(group)
            (loss*weight).backward()
            update_loss += float(loss.detach())*weight
            for name, value in components.items():
                update_components[name] = update_components.get(name,0.)+float(value.detach())*weight
        parameters = [p for g in optimizer.param_groups for p in g['params']]
        torch.nn.utils.clip_grad_norm_(parameters,args.max_grad_norm,error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        cursor += len(group)
        epoch_update += 1
        progress.update(cursor=cursor,update=progress['update']+1,last_loss=update_loss,
                        loss_sum=progress['loss_sum']+update_loss*len(group),samples=progress['samples']+len(group))
        if update_components:
            progress['last_loss_components'] = update_components
        recent.append(update_loss)
        if epoch_update % args.log_every == 0 or cursor == len(rows) or (cap > 0 and progress['update'] == cap):
            tokens = usage.snapshot() if usage is not None else None
            token_log = (f" training_processed_token_positions={tokens['totals']['training_processed_token_positions']} "
                         f"ce_tokens={tokens['training']['ce_target_tokens']}" if tokens is not None else '')
            component_log = ''.join(f' last_{name}={value:.4f}' for name,value in update_components.items())
            print(f"{label}: epoch={progress['epoch']+1} update={progress['update']} "
                  f"epoch_update={epoch_update} "
                  f"loss={sum(recent)/len(recent):.4f} mean_loss={progress['loss_sum']/progress['samples']:.4f} "
                  f"samples={cursor}/{len(rows)} csi_steps={message_steps} "
                  f"batch_size={batch_size} "
                  f"lr={optimizer.param_groups[0]['lr']:.6g}{component_log}{token_log}",flush=True)
            recent.clear()
            if on_log is not None:
                on_log()
        if args.save_every and progress['update'] % args.save_every == 0:
            on_save()
        updates_this_call += 1
        if max_updates_per_call is not None and updates_this_call >= max_updates_per_call:
            break
    completed = cursor == len(rows)
    summary = dict(epoch=progress['epoch']+1,mean_loss=progress['loss_sum']/max(1,progress['samples']),
                   update=progress['update'],samples=progress['samples'],complete=completed)
    if usage is not None:
        summary['token_usage'] = {key:value-progress['epoch_token_start'][key]
                                  for key,value in usage.phase_totals().items()}
    if completed:
        progress.update(epoch=progress['epoch']+1,cursor=0,loss_sum=0.,samples=0,last_loss=0.)
        progress.pop('epoch_token_start',None)
    return summary


def _move_inner_optimizer_state(optimizer):
    # Adam's non-capturable step counter stays on the CPU. Moments follow parameters.
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if name != 'step' and isinstance(value, torch.Tensor):
                state[name] = value.to(device=parameter.device)


@contextmanager
def parallel_inner_layout(system, inner_optimizers):
    """Temporarily colocate each agent's local CSI and InnerLink with its frozen LLM."""
    placements = [(module, next(module.parameters()).device)
                  for i in range(len(system.roles)) for module in system.inner_modules(i)]
    try:
        system.requires_grad_(False)
        for i, role in enumerate(system.roles):
            for module in system.inner_modules(i):
                module.to(system.backbones[i].device)
                module.requires_grad_(True)
            _move_inner_optimizer_state(inner_optimizers[role])
        yield
    finally:
        # The pool joins and CUDA workers synchronize before this context exits.
        for module, device in placements:
            module.to(device)
        for optimizer in inner_optimizers.values():
            _move_inner_optimizer_state(optimizer)


def run_inner_parallel(system, rows, inner_optimizers, inner_schedulers, progress, args, *, on_save, on_log):
    """One independent optimizer window per active role, then a coordinated checkpoint boundary.

    Only the coordinator writes files or snapshots token counters. Workers never
    toggle global trainability, model mode, token-accounting phase or max_length.
    """
    roles = system.roles

    def pending():
        return [i for i, role in enumerate(roles)
                if progress['inner'][role]['epoch'] < args.inner_epochs and
                (args.inner_max_updates < 0 or progress['inner'][role]['update'] < args.inner_max_updates)]

    def checkpoint(name='csi_latest.pt'):
        # Save the regular Outer input budget, as in the sequential entry point.
        with inner_context(system, args.max_length):
            on_save(name)

    def save_finished(i):
        # A parallel snapshot may contain several unfinished roles. Sequential
        # resume must start at the earliest of them, regardless of the filename.
        progress['role_index'] = (pending() or [system.final_index])[0]
        checkpoint(f'csi_inner_{roles[i]}.pt')
        print(f"Inner {roles[i]}: complete updates={progress['inner'][roles[i]]['update']}", flush=True)

    active = pending()
    finished = set(range(len(roles))) - set(active)
    for i in sorted(finished):
        save_finished(i)
    if not active:
        return
    recent = {i: [] for i in range(len(roles))}
    with parallel_inner_layout(system, inner_optimizers), inner_context(system, args.inner_max_length), \
         system.token_usage.scope('inner'):
        system.train()
        streams = {}
        for i in active:
            device = system.backbones[i].device
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    streams[i] = stream
        print('Inner parallel: enabled; '+', '.join(f'{roles[i]}={system.backbones[i].device}' for i in active), flush=True)

        def update(i, signals):
            role = roles[i]
            def optimize():
                with torch.enable_grad():
                    return optimize_epoch(system, rows, inner_optimizers[role], inner_schedulers[role],
                        progress['inner'][role], args, epochs=args.inner_epochs, cap=args.inner_max_updates,
                        salt=10000*i, label=f'Inner {role}',
                        loss_fn=lambda row: system.inner_loss(row, i, ce_weight=args.inner_ce_weight,
                            alignment_weight=args.inner_align_weight, cosine_weight=args.inner_cos_weight,
                            mse_weight=args.inner_mse_weight, return_components=True),
                        batch_loss_fn=lambda group: system.inner_loss_batch(group, i, ce_weight=args.inner_ce_weight,
                            alignment_weight=args.inner_align_weight, cosine_weight=args.inner_cos_weight,
                            mse_weight=args.inner_mse_weight, return_components=True),
                        batch_size=args.inner_batch_size, csi_steps=1, max_updates_per_call=1,
                        manage_mode=False, report_usage=False, recent_losses=recent[i],
                        on_save=lambda: signals.__setitem__('save', True),
                        on_log=lambda: signals.__setitem__('log', True))
            if i not in streams:
                return optimize()
            stream = streams[i]
            with torch.cuda.device(system.backbones[i].device), torch.cuda.stream(stream):
                try:
                    return optimize()
                finally:
                    # Commit all GPU writes before the coordinator reads weights/counters.
                    stream.synchronize()

        with ThreadPoolExecutor(max_workers=len(roles), thread_name_prefix='csi-inner') as pool:
            while active:
                signals = {i: dict(save=False, log=False) for i in active}
                futures = {i: pool.submit(update, i, signals[i]) for i in active}
                results = {}
                for i, future in futures.items():
                    try:
                        results[i] = future.result()
                    except Exception as exc:
                        raise RuntimeError(f'Inner parallel {roles[i]} failed: {exc}') from exc
                # Every future is now at an optimizer boundary; snapshots are coherent.
                next_active = pending()
                newly_finished = [i for i in active if i not in next_active]
                for i in newly_finished:
                    save_finished(i)
                    finished.add(i)
                progress['role_index'] = next_active[0] if next_active else system.final_index
                if any(flag['log'] for flag in signals.values()) and on_log is not None:
                    on_log()
                if (any(flag['save'] for flag in signals.values()) or newly_finished
                        or any(summary['complete'] for summary in results.values())):
                    checkpoint()
                active = next_active


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    data_path = Path(args.train_parquet or args.train_jsonl).expanduser().resolve()
    dataset = CodeTrainingDataset(data_path, mas_system=args.mas_system)
    if dataset.restored_original_targets:
        print(f'Restored original plan/refined_plan fields in {dataset.restored_original_targets} imported records',flush=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    system = construct_system(args)
    roles = system.roles
    if args.init_checkpoint:
        system.load_trainable_checkpoint(args.init_checkpoint,initialize=True)
        print(f'Initialize weights: {args.init_checkpoint}; fresh optimizers, schedulers and progress',flush=True)
    system.configure_training_stage('outer')
    outer_optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],
                                        lr=args.learning_rate,weight_decay=args.weight_decay)
    inner_optimizers = {role:torch.optim.AdamW(system.inner_parameters(i),lr=args.inner_learning_rate)
                        for i,role in enumerate(roles)}
    outer_total = planned_optimizer_updates(len(dataset),args.epochs,args.gradient_accumulation_steps,args.max_updates,
                                            batch_size=args.batch_size)
    inner_total = planned_optimizer_updates(len(dataset),args.inner_epochs,args.gradient_accumulation_steps,args.inner_max_updates,
                                            batch_size=args.inner_batch_size)
    outer_scheduler = build_lr_scheduler(outer_optimizer,outer_total,args.warmup_updates,args.lr_scheduler,args.min_lr_ratio)
    inner_schedulers = {r:build_lr_scheduler(o,max(1,inner_total),args.warmup_updates,args.lr_scheduler,args.min_lr_ratio)
                        for r,o in inner_optimizers.items()}
    progress = dict(phase='inner',role_index=0,inner={r:_progress() for r in roles},outer=_progress())
    data_hash = _digest(data_path)
    if args.resume:
        payload = system.load_trainable_checkpoint(args.resume,strict_runtime=True)
        if (payload.get('training_schedule') != training_schedule(system)
                or payload.get('inner_training_objective') != INNER_TRAINING_OBJECTIVE):
            raise ValueError('Inner training objective or optimizer parameters changed; --resume requires a checkpoint '
                             'from the CE+alignment Inner trainer. Use --init-checkpoint CHECKPOINT for a fresh run')
        if payload.get('training_data_sha256') != data_hash:
            raise ValueError('Resume requires the same training dataset')
        if not resume_sources_compatible(payload.get('source_sha256'),source_hashes(),mas_system=args.mas_system):
            raise ValueError('Source changed since checkpoint; choose a new training run')
        ignored = {'resume','init_checkpoint','output_dir','log_every','save_every','experience_file','inner_parallel'}
        for key,value in payload['training_args'].items():
            if key not in ignored and getattr(args,key) != value and str(getattr(args,key)) != str(value):
                raise ValueError(f'Resume requires matching --{key.replace("_","-")}: saved={value!r}')
        progress = payload['progress']
        system.token_usage.load_state_dict(payload.get('token_usage'),progress=progress)
        if not system.token_usage.coverage['complete_from_run_start']:
            print('Processed token positions: earlier training was not measured; totals cover the recorded portion only.',flush=True)
        outer_optimizer.load_state_dict(payload['optimizer_state_dict'])
        outer_scheduler.load_state_dict(payload['lr_scheduler_state_dict'])
        for role in roles:
            inner_optimizers[role].load_state_dict(payload['inner_optimizer_state_dicts'][role])
            inner_schedulers[role].load_state_dict(payload['inner_lr_scheduler_state_dicts'][role])
        torch.set_rng_state(payload['rng_state']); random.setstate(payload['python_rng_state'])
        if payload['cuda_rng_state'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload['cuda_rng_state'])
        print(f"Resume: stage={progress['phase']} outer_updates={progress['outer']['update']}",flush=True)
    else:
        for role, name, inner_link in zip(roles, MAS_MODEL_ROLES[args.mas_system], system.inner_links):
            path = getattr(args,name+'_inner_path')
            if path:
                state = torch.load(Path(path).expanduser(),map_location='cpu',weights_only=False)
                inner_link.load_state_dict(state.get('adapter',state),strict=True)
                progress['inner'][role]['epoch'] = args.inner_epochs
            elif args.inner_epochs == 0 and not args.init_checkpoint:
                raise ValueError('--inner-epochs 0 requires a trained --*-inner-path adapter for each active agent')
    def log_tokens():
        write_json(args.output_dir/'token_usage.json',system.token_usage.snapshot())

    def save(name='csi_latest.pt'):
        save_checkpoint(args.output_dir/name,system,outer_optimizer,inner_optimizers=inner_optimizers,
                        outer_scheduler=outer_scheduler,inner_schedulers=inner_schedulers,
                        progress=progress,args=args,training_data_sha256=data_hash)
        log_tokens()
    write_json(args.output_dir/'training_args.json',
               {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    metrics_path = args.output_dir/'training_metrics.json'
    metrics = json.loads(metrics_path.read_text()) if args.resume and metrics_path.exists() else []
    for metric in metrics:
        if 'token_usage' in metric:
            metric['token_usage'] = system.token_usage.normalize_counter(metric['token_usage'])
    write_json(metrics_path,metrics)
    log_tokens()
    print(f'Training rows={len(dataset)} outer_batch_size={args.batch_size} '
          f'inner_batch_size={args.inner_batch_size} accumulation={args.gradient_accumulation_steps}',flush=True)
    if progress['phase'] == 'inner':
        if args.inner_parallel:
            run_inner_parallel(system,dataset.rows,inner_optimizers,inner_schedulers,progress,args,
                               on_save=save,on_log=log_tokens)
        for i in ([] if args.inner_parallel else range(progress['role_index'],len(roles))):
            role = roles[i]; progress['role_index'] = i
            state = progress['inner'][role]
            system.configure_training_stage('inner',i)
            print(f'Inner {role}: start local_CSI_CE_weight={args.inner_ce_weight} '
                  f'InnerLink_alignment_weight={args.inner_align_weight}',flush=True)
            while state['epoch'] < args.inner_epochs and (args.inner_max_updates < 0 or state['update'] < args.inner_max_updates):
                with inner_context(system,args.inner_max_length), system.token_usage.scope('inner'):
                    summary = optimize_epoch(system,dataset.rows,inner_optimizers[role],inner_schedulers[role],state,args,
                        epochs=args.inner_epochs,cap=args.inner_max_updates,salt=10000*i,label=f'Inner {role}',
                        loss_fn=lambda row:system.inner_loss(row,i,ce_weight=args.inner_ce_weight,
                            alignment_weight=args.inner_align_weight,cosine_weight=args.inner_cos_weight,
                            mse_weight=args.inner_mse_weight,return_components=True),
                        batch_size=args.inner_batch_size,csi_steps=1,
                        batch_loss_fn=lambda rows:system.inner_loss_batch(rows,i,ce_weight=args.inner_ce_weight,
                            alignment_weight=args.inner_align_weight,cosine_weight=args.inner_cos_weight,
                            mse_weight=args.inner_mse_weight,return_components=True),
                        on_save=save,on_log=log_tokens)
                # Save with the regular outer max_length restored in model_config.
                save()
            system.set_inner_trainable(False)
            save(f'csi_inner_{role}.pt')
            print(f"Inner {role}: complete updates={state['update']}",flush=True)
        progress['phase'] = 'outer'; save()
    system.configure_training_stage('outer')
    state = progress['outer']
    if state['epoch'] == 0 and state['cursor'] == 0 and state['update'] == 0:
        save('csi_outer_epoch_0.pt')
    while state['epoch'] < args.epochs and (args.max_updates < 0 or state['update'] < args.max_updates):
        with system.token_usage.scope('outer'):
            summary = optimize_epoch(system,dataset.rows,outer_optimizer,outer_scheduler,state,args,
                        epochs=args.epochs,cap=args.max_updates,salt=100000,label='Outer',
                        loss_fn=lambda row:system.training_loss(row)[0],batch_size=args.batch_size,
                        batch_loss_fn=lambda rows:system.training_loss_batch(rows)[0],on_save=save,on_log=log_tokens)
        if summary['complete']:
            if args.outer_validation:
                from evaluate_math500_csi import evaluate_examples
                validation = load_records(args.validation_jsonl)
                if args.validation_limit:
                    validation = validation[:args.validation_limit]
                try:
                    with system.token_usage.scope('validation'):
                        result = evaluate_examples(system,validation,args.output_dir/'validation'/f"outer_epoch_{state['epoch']}.jsonl",
                                    max_new_tokens=args.validation_max_new_tokens,grader_mode=args.validation_grader,
                                    log_every=args.log_every,batch_size=args.validation_batch_size)
                finally:
                    log_tokens()
                summary['validation'] = result
            metrics = [m for m in metrics if m['epoch'] != summary['epoch']]+[summary]
            write_json(metrics_path,metrics)
            save(f"csi_outer_epoch_{state['epoch']}.pt")
        save()
    save('csi_final.pt')
    print(f"Outer: complete updates={state['update']}",flush=True)
    tokens = system.token_usage.snapshot()
    print(f"Processed token positions: inner={tokens['phases']['inner']['totals']['processed_token_positions']} "
          f"outer={tokens['phases']['outer']['totals']['processed_token_positions']} "
          f"total={tokens['totals']['training_processed_token_positions']} "
          f"validation={tokens['totals']['validation_processed_token_positions']}",flush=True)
    print(f'Processed token position report: {args.output_dir/"token_usage.json"}',flush=True)
    print(f'Final checkpoint: {args.output_dir/"csi_final.pt"}',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
