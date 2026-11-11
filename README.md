<div align="center">
  <img src="./264091041.jfif" alt="openJiuwen Logo" width="180">
  <h1>OpenJiuwen</h1>
  <p><strong>Convolutional co-evolution for multi-agent language models</strong></p>
  <p>
    <a href="#model"><img src="https://img.shields.io/badge/ConvRSI-v3-5C3C97?style=flat-square" alt="ConvRSI v3"></a>
    <a href="requirements.txt"><img src="https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch 2.9.0"></a>
    <a href="requirements.txt"><img src="https://img.shields.io/badge/Transformers-5.3.0-AD861F?style=flat-square" alt="Transformers 5.3.0"></a>
    <a href="#evaluation"><img src="https://img.shields.io/badge/Evaluation-MATH--500-146C5E?style=flat-square" alt="MATH-500 evaluation"></a>
  </p>
</div>

## Demo

<!-- DEMO START. Replace this blank panel with the demo image, GIF, or video link. -->
<table width="100%">
  <tr>
    <td align="center" width="100%" height="144">
      &nbsp;<br><br><br><br><br>&nbsp;
    </td>
  </tr>
</table>
<!-- DEMO END -->

## Overview

**ConvRSI** gives each frozen language-model agent a continuous representation that evolves over multiple rounds of collaboration. Intra-agent convolution combines the features available to each agent. A graph convolution updates their messages jointly, and a learned projection turns the messages into soft prefixes for the language models.

This v3 checkout includes two-stage training, single-question inference, and batched MATH-500 evaluation. Its reference configuration runs Planner, Critic, and Solver agents in a sequential math workflow.

## Table of contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Third-party licenses](#third-party-licenses)

## Installation

Run the following commands from the v3 project directory.

```bash
conda create -n convrsi python=3.10 -y
conda activate convrsi
python -m pip install -r requirements.txt
```

The example uses a CUDA device with BF16 support. The trainer also supports `--model-dtype float32`.

### Backbone weights

Place the pretrained model files in the directories below. The evaluator and single-question runner can restore these paths from a training checkpoint.

| Agent | Model | Default local directory |
| --- | --- | --- |
| Planner | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `pre-trained/Qwen3-1.7B` |
| Critic | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | `pre-trained/Llama-3.2-1B-Instruct` |
| Solver | [Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) | `pre-trained/Qwen2.5-Math-1.5B-Instruct` |

To point at another location, pass `--planner-model-path`, `--critic-model-path`, or `--executor-model-path`. The CLI uses `executor` for the Solver.

## Datasets

| Split | Records | Local file |
| --- | ---: | --- |
| Sequential-Math training | 1,904 | [`data/sequential_math.jsonl`](data/sequential_math.jsonl) |
| MATH-500 test | 500 | [`data/math500.jsonl`](data/math500.jsonl) |

Training rows contain `question`, `plan`, `refined_plan`, and `answer`. The training split removes six questions whose text exactly matches a MATH-500 question after case folding and whitespace removal. Evaluation supplies only the question to the model and uses the reference answer for grading. [Dataset provenance](data/README.md)

## Model

Each frozen backbone encodes task, role, system, and experience once per example in a system forward pass. A learned reader compresses each feature into **8 slots** by default. A fifth feature carries the recurrent message, which starts as a masked zero.

- **Intra-agent Conv** projects the feature slots into a shared width and applies depthwise Conv1D along the feature axis, independently at each slot. Gated pooling produces the local messages.
- **Inter-agent GCN** aggregates local messages across agents at each slot using a graph with normalized self-loops. Its output supplies both the current round’s soft prefix and the next round’s message.
- **Latent collaboration** transfers agent execution states through a separate link path. The v3 trainer interleaves one Conv–GCN update with one Planner → Critic → Solver pass per round. Each intermediate Solver sends latent feedback to the next round's Planner.

Fresh training records `conv_pointwise=False`, `cross_feature_fusion=False`, and `gcn_residual=False`. The convolution block has its own residual switch.

```text
.
├── csi_qwen.py              # Frozen backbones, slot encoding, and multi-round execution
├── conv_graph_mas_demo.py   # Intra-agent convolution and graph convolution
├── train_csi.py             # Inner and Outer training
├── evaluate_math500_csi.py  # Batched evaluation and answer grading
├── ours_runner.py           # Single-question inference
├── data/                    # Prepared training and evaluation splits
└── vendor/recursivemas/      # Vendored prompts and latent-link components
```

## Training

With the model directories in place, train on Sequential-Math. The defaults run the three backbones and CSI modules on `cuda:0`.

```bash
python train_csi.py \
  --train-jsonl data/sequential_math.jsonl \
  --inner-epochs 1 --epochs 3 \
  --inner-batch-size 2 --batch-size 2 \
  --max-steps 3 \
  --output-dir runs/convrsi_v3
```

| Stage | Objective | Trainable components |
| --- | --- | --- |
| **Inner** | Per-agent token CE and cosine alignment on detached hidden states | Local CSI modules via CE and InnerLinks via alignment |
| **Outer** | Teacher-forced collaboration with token CE on the final Solver answer | Local CSI modules, GCN, and OuterLinks |

Use `--inner-parallel` to run the independent Inner objectives concurrently. The device flags `--planner-device`, `--critic-device`, `--executor-device`, and `--csi-device` control placement.

<details>
<summary><strong>Run a short training smoke test</strong></summary>

```bash
python train_csi.py \
  --train-jsonl data/sequential_math_canary.jsonl \
  --inner-batch-size 1 --batch-size 1 \
  --inner-max-updates 2 --max-updates 2 \
  --output-dir runs/convrsi_canary
```

</details>

### Configuration

| Flag | Default | Controls |
| --- | ---: | --- |
| `--max-steps` | 3 | Complete Conv–GCN and collaboration rounds |
| `--csi-prefix-tokens` | 8 | Continuous CSI slots per feature and soft prefix |
| `--d-model` | 512 | Shared convolution and graph width |
| `--max-prefix-tokens` | 80 | Maximum number of transferred latent positions in Outer training |
| `--latent-length` | 32 | Latent rollout steps per non-final agent invocation during inference |

### Checkpoints

The run directory contains `csi_latest.pt`, completed Outer epoch checkpoints, `csi_final.pt`, and `token_usage.json`.

| Flag | Use |
| --- | --- |
| `--resume` | Continue a matching run with its optimizer, scheduler, and progress state. |
| `--init-checkpoint` | Start a new run from compatible weights with fresh optimizer and scheduler state. |
| `--checkpoint-path` | Load a trained run for evaluation or inference. |

Detailed options and checkpoint compatibility are covered in the [technical guide](CSI_QWEN_GUIDE.md).

## Evaluation

Run MATH-500 with a trained checkpoint. Intermediate agents generate continuous latent states, and the final Solver decodes the answer.

```bash
python evaluate_math500_csi.py \
  --dataset-path data/math500.jsonl \
  --checkpoint-path runs/convrsi_v3/csi_final.pt \
  --output-dir runs/convrsi_v3_math500 \
  --batch-size 4 --grader math_verify
```

The default `after` evaluation writes per-question records to `after_predictions.jsonl` and metrics to `after_predictions_summary.json` and `summary.json`. Use `--limit` to select a subset and `--batch-size` to set the number of questions processed together.

### Single-question inference

```bash
python ours_runner.py \
  --checkpoint-path runs/convrsi_v3/csi_final.pt \
  --question "Find the sum of the first 20 positive integers."
```

## Acknowledgements

The v3 implementation uses prompts and latent-link components from RecursiveMAS. The vendored revision and local changes are recorded in the project provenance. [Upstream code](https://github.com/RecursiveMAS/RecursiveMAS) · [Vendor provenance](UPSTREAM.json)

The training and test splits come from [Sequential-Math](https://huggingface.co/datasets/RecursiveMAS/Sequential-Math) and [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

## Third-party licenses

The vendored RecursiveMAS code and Sequential-Math data carry their respective MIT license notices. [RecursiveMAS license](vendor/recursivemas/LICENSE) · [Sequential-Math license](data/Sequential-Math-LICENSE)
