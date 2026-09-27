# CSI testing guide

`conv_graph_mas_demo.py` contains the module-level CSI core: shared latent
projection, a feature-axis intra-agent dilated Conv1D stack (`kernel_size=3`,
dilations `1, 2`, effective receptive field `5`), and the inter-agent normalized
GCN update. The Qwen integration and benchmark runner live in `csi_qwen.py` and
`ours_runner.py`.

## Level 1: module smoke test

```bash
python conv_graph_mas_demo.py
```

This runs one training update and one adaptive inference pass with synthetic
features. A successful run prints a finite training loss, a 3-by-3 learned
adjacency matrix, and the number of inference steps.

## Level 2: automated module tests

```bash
python -m unittest -v test_csi.py
```

The tests cover:

- projection of heterogeneous agent states into a shared latent space;
- Conv1D input reshaping from `[B,N,5,S,D]` to `[B*N*S,5,D]` so convolution
  runs across the five feature representations;
- two dilation scales and full effective coverage of the five-feature axis;
- output tensor shapes;
- symmetric GCN adjacency normalization with self-loops;
- preservation of missing communication edges;
- finite gradients through Conv1D and graph updates;
- adaptive stopping during evaluation.

These checks establish that the CSI neural blocks run correctly, but they do
not establish code-generation quality.

## Level 3: end-to-end code-generation test

Before measuring pass@1 or comparing with RecursiveMAS, provide:

1. identical Planner, Critic, and Executor base models to the baseline;
2. a trained CSI checkpoint for the Conv1D, graph, and projection modules;
3. the same task records and execution evaluator for both systems.

Once those pieces exist, run the same task set through `compare_codegen.py`
with Docker execution. Model loading belongs in the runner constructor; only
the `generate` call is timed.
