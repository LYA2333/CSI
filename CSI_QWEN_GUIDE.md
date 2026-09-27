# CSI with three Qwen agents

This implementation connects three frozen Qwen-family causal LMs to the CSI
Conv1D+GCN core:

```text
Planner Qwen  ─┐
Critic Qwen   ─┼─> warmed Inner Adapters ─> four feature encodings ─> [B,3,4,S,D]
Executor Qwen ─┘                                                       │
                                                     evolving message ─┘
                                              v
                                      [B,3,5,S,D]
                                              │
                                              v
                   feature-axis Conv1D over task/role/experience/prompt/message
                                              │
                                              v
                       directed-cycle GCN for several cheap rounds
                                      │             │
                                      │      intermediate answer alignment
                                              │
                                              v
                           Executor message ─> soft prefix
                                              │
                                              v
                                  Executor Qwen ─> Python code
```

The Qwen models are frozen. Training first warms the three role-specific
residual Inner Adapters sequentially: Planner, then Refiner, then Solver. Each
role has its own optimizer and supervision target. The warmed adapters are then
frozen while outer CSI training updates the
heterogeneous projectors, Conv1D blocks, GCN, and Executor prefix mapper.

The intra-agent Conv1D treats the five representations as its sequence axis.
For each latent-token position it reshapes `[B,N,5,S,D]` to `[B*N*S,5,D]`,
applies the dilated convolution stack across the five feature types, then uses
gated pooling to produce one local state per agent. It does not convolve along
the latent-token axis `S`.

## Files

- `csi_qwen.py`: Qwen loading, feature encoding, latent resampling, CSI model,
  language-model loss, and cached autoregressive generation.
- `train_csi.py`: Parquet/JSONL training and checkpoint entry point.
- `ours_runner.py`: real CSI runner for `compare_codegen.py`.
- `csi_train.example.jsonl`: minimal training-record format.
- `test_csi_qwen.py`: fake-model integration tests; no Qwen download required.

## 1. Fill paths

The defaults are intentionally empty in `train_csi.py` and `ours_runner.py`:

```python
parser.add_argument("--planner-model-path", default="")
parser.add_argument("--critic-model-path", default="")
parser.add_argument("--executor-model-path", default="")
```

You may edit those defaults on the model machine or, preferably, pass the paths
on the command line. The latter keeps machine-specific paths out of source
control.

## 2. Dataset format

### RecursiveMAS Sequential-Math Parquet

Download `data/train-00000-of-00001.parquet` from
`RecursiveMAS/Sequential-Math`. The loader uses `pandas.read_parquet()` and
maps its fields as follows:

```text
question      -> prompt
answer        -> target_text
plan          -> Planner Inner Adapter supervision
refined_plan  -> Critic Inner Adapter supervision
answer_field  -> retained metadata
task_type     -> "math" (added by the loader)
```

Install the Parquet dependencies on the training machine:

```bash
pip install pandas pyarrow
```

`plan` and `refined_plan` supervise the role adapters during training, but are
deliberately not copied into the CSI `experience` input. They are unavailable
for a new question at inference time, so using them as ordinary features would
leak teacher information.

### Original JSONL format

Training requires one JSON object per line:

```json
{
  "task_id": "factorial",
  "prompt": "Write factorial(n)...",
  "entry_point": "factorial",
  "plan": "Validate n, then compute an iterative product.",
  "refined_plan": "Handle n=0 and n=1 through result=1; reject n<0 first.",
  "canonical_solution": "def factorial(n): ...",
  "experience": {
    "planner": "...",
    "critic": "...",
    "executor": "..."
  },
  "tests": ["assert factorial(5) == 120"]
}
```

`canonical_solution` (or the aliases `solution`/`code`) is required. By default,
`plan` and `refined_plan` are also required for the preliminary Inner stage.
Pass `--inner-warmup-epochs 0` to disable Inner training; the untrained
post-LayerNorm adapters will still remain in the forward path, so this option
is intended only for ablation rather than the main configuration. `tests` are
used by the separate evaluation harness, not by the differentiable loss.

Like the released RecursiveMAS training pipeline, `train_csi.py` consumes one
training split and does not run a validation pass or early stopping. Downstream
accuracy/pass-rate evaluation remains separate in `compare_codegen.py`.

## 3. Training objectives

Training follows two strictly separated stages:

1. **Inner warm-up:** before any Outer update, three separate sub-stages run in
   order. Planner aligns to `plan`, Refiner aligns to `refined_plan`, and Solver
   aligns to `target_text` using next-token embedding cosine loss (with optional
   MSE). Each sub-stage has its own backward pass and optimizer. Every role
   adapter applies post-residual LayerNorm to prevent its latent norm from
   growing without bound under cosine-only supervision.
2. **Outer CSI training:** after all requested Inner passes finish, the Inner
   Adapters are frozen for every Outer epoch. The final CSI round uses Solver
   token CE. Earlier-round cosine alignment is optional and disabled by default.

With the default `--inner-warmup-epochs 1`, the schedule is:

```text
Planner Inner: 1 complete dataset pass
Refiner Inner: 1 complete dataset pass
Solver Inner:  1 complete dataset pass
Outer stage: epoch 1 -> epoch 2 -> ... -> epoch N
```

Setting `--inner-warmup-epochs N` runs exactly `N` dataset passes for each of
the three Inner roles before the first Outer epoch. `--inner-max-updates` is
also applied separately to each role. Setting the epoch count to `0` disables
Inner updates. Logs are labeled `Planner:`, `Refiner:`, and `Solver:`. At the stage
boundary, training proceeds directly to the Outer stage without writing an
intermediate checkpoint. Only `csi_final.pt` is saved after all Outer epochs.
It contains the completed Inner weights, the three Inner optimizer states, and
the Outer optimizer state.

For both training and generation, Executor input order is `prompt embeddings ->
CSI soft-prefix -> target/generated tokens`. This lets the CSI tokens refine an
already encoded task prompt and keeps training and inference consistent.

The outer objective is:

```text
loss = final_ce + intermediate_loss_weight * mean(non-final round alignment)
```

`intermediate_loss_weight` defaults to `0`, so the baseline objective is only
`final_ce`. The optional intermediate objective does not run the full Solver
once per round, which avoids multiplying Solver activation memory on its GPU.

## 4. Two-update canary training

Start with one or two records before a full run:

```bash
python train_csi.py \
  --planner-model-path pre-trained/Qwen3-0.6B \
  --critic-model-path pre-trained/Qwen3.5-0.8B \
  --executor-model-path pre-trained/Qwen2.5-1.5B-Instruct \
  --train-parquet datasets/Sequential-Math/data/train-00000-of-00001.parquet \
  --planner-device cuda:4 \
  --critic-device cuda:5 \
  --executor-device cuda:6 \
  --csi-device cuda:7 \
  --model-dtype bfloat16 \
  --d-model 512 \
  --latent-tokens 32 \
  --prefix-tokens 8 \
  --dilations 1 2 \
  --batch-size 1 \
  --inner-warmup-epochs 1 \
  --inner-max-updates 2 \
  --intermediate-loss-weight 0 \
  --max-updates 2 \
  --log-every 1 \
  --output-dir csi_checkpoints_canary
```

If all models fit on one GPU, set every device to `cuda:0`. If they do not fit,
use smaller Qwen variants or separate GPUs. This first implementation keeps all
three models resident so that later latency measurements do not include model
loading/offloading.

The run completes Planner, Refiner, and Solver Inner work in that order, freezes
all three adapters, and then begins Outer work. Intermediate Inner and per-epoch
checkpoints are not written; the combined model plus all optimizer states are
saved once as `csi_final.pt` after training finishes.
Outer logs show `loss`, `final_ce`, and `intermediate` separately;
`intermediate` is zero by default.

## 5. Full training

After the canary succeeds, remove `--max-updates 2`, point to the full training
file, and run multiple epochs:

```bash
python train_csi.py \
  --planner-model-path pre-trained/Qwen3-0.6B \
  --critic-model-path pre-trained/Qwen3.5-0.8B \
  --executor-model-path pre-trained/Qwen2.5-1.5B-Instruct \
  --train-parquet datasets/Sequential-Math/data/train-00000-of-00001.parquet \
  --planner-device cuda:4 \
  --critic-device cuda:5 \
  --executor-device cuda:6 \
  --csi-device cuda:7 \
  --epochs 3 \
  --inner-warmup-epochs 1 \
  --intermediate-loss-weight 0 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --output-dir csi_checkpoints
```

The final small-module checkpoint is `csi_checkpoints/csi_final.pt`; frozen Qwen
weights are not copied into it.

## 6. Generate one program

Architecture options must match the checkpoint used during training:

```bash
python ours_runner.py \
  --planner-model-path /path/to/planner-qwen \
  --critic-model-path /path/to/critic-qwen \
  --executor-model-path /path/to/executor-qwen \
  --checkpoint-path csi_checkpoints/csi_final.pt \
  --planner-device cuda:4 \
  --critic-device cuda:5 \
  --executor-device cuda:6 \
  --csi-device cuda:7 \
  --kernel-size 5 \
  --entry-point add \
  --prompt "Write a Python function add(a, b) that returns a + b."
```

## 7. Compare with RecursiveMAS

```bash
python compare_codegen.py \
  --tasks codegen_tasks.jsonl \
  --recursive-runner recursive_mas_runner_example.py:build_runner \
  --recursive-kwargs '{"repo_root":"/path/to/RecursiveMAS","device":"cuda:0","dataset":"mbppplus"}' \
  --ours-runner ours_runner.py:build_runner \
  --ours-kwargs '{"planner_model_path":"/path/to/planner-qwen","critic_model_path":"/path/to/critic-qwen","executor_model_path":"/path/to/executor-qwen","checkpoint_path":"/path/to/csi_final.pt","planner_device":"cuda:0","critic_device":"cuda:1","executor_device":"cuda:2","csi_device":"cuda:2"}' \
  --executor docker \
  --output codegen_report.json
```

Run both systems with the same base models, decoding options, warm-up policy,
hardware allocation, and test set before interpreting pass@1 or latency.

## 8. Local structure tests

```bash
python -m unittest -v test_csi.py test_csi_qwen.py
```

These tests verify shapes, directed-cycle topology, GCN normalization, gradients
from the language-model loss into CSI, and the KV-cache generation loop. They do
not establish real code-generation quality without your Qwen weights and data.
