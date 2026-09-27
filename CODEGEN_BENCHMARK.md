# Code-generation comparison

This benchmark measures two separate outcomes on exactly the same tasks:

- `pass@1`: the first generated program passes every test.
- inference latency: time spent in `runner.generate`, excluding model loading and
  program execution.

Both models are loaded once before timing. Evaluation uses batch size 1 and
synchronizes CUDA/MPS around every generation request.

## Smoke test

The included runners return hand-written code and are only for checking the
benchmark pipeline:

```bash
python3 compare_codegen.py \
  --tasks codegen_tasks.example.jsonl \
  --recursive-runner codegen_runner_example.py:build_recursive_runner \
  --ours-runner codegen_runner_example.py:build_ours_runner \
  --executor local \
  --allow-unsafe-local-execution \
  --output codegen_report.json
```

Use Docker for untrusted model output:

```bash
python3 compare_codegen.py \
  --tasks codegen_tasks.jsonl \
  --recursive-runner recursive_runner.py:build_runner \
  --ours-runner ours_runner.py:build_runner \
  --executor docker
```

The Docker executor disables networking, drops Linux capabilities, limits CPU,
memory and processes, and mounts only the generated test script as read-only.

## Real runner interface

Each runner file needs a factory. Model and checkpoint loading belongs in the
constructor so it remains outside the timed section:

```python
class RecursiveRunner:
    def __init__(self):
        self.system = load_recursive_mas_once(...)

    def generate(self, prompt: str, task_id: str, task: dict) -> str:
        return self.system.generate_code(
            prompt,
            num_recursive_rounds=3,
            latent_length=48,
            temperature=0.2,
        )

def build_runner():
    return RecursiveRunner()
```

Implement the same interface for CSI-GCN. Match the following settings:

- identical task IDs, prompts and tests;
- identical Planner/Critic/Executor base models;
- batch size, dtype, maximum new tokens and decoding parameters;
- recursion rounds and number of rollouts;
- hardware and process placement.

For the real three-Qwen CSI implementation, use
`ours_runner.py:build_runner`. See `CSI_QWEN_GUIDE.md` for training, checkpoint,
and runner arguments.

For MBPP+ or LiveCodeBench, convert each benchmark record into the JSONL fields
`task_id`, `prompt`, optional `setup`, and `tests`. For publication results,
prefer the official EvalPlus/LiveCodeBench evaluator rather than the lightweight
assert executor in this prototype.

`recursive_mas_runner_example.py` provides a more concrete adapter over the
current official sequential RecursiveMAS stage functions. Its stages load and
release models internally, matching the memory-conscious official pipeline but
also including that lifecycle in `generate()` time. Use it first for functional
validation; refactor both compared runners to the same model-lifecycle policy
before reporting latency.

Example with the official repository and released checkpoints:

```bash
python3 compare_codegen.py \
  --tasks codegen_tasks.jsonl \
  --recursive-runner recursive_mas_runner_example.py:build_runner \
  --recursive-kwargs '{
    "repo_root": "/absolute/path/to/RecursiveMAS",
    "style": "sequential_light",
    "dataset": "mbppplus",
    "device": "cuda:0",
    "recursive_rounds": 3,
    "latent_steps": 48,
    "max_new_tokens": 2048
  }' \
  --ours-runner ours_runner.py:build_runner \
  --ours-kwargs '{"device": "cuda:0"}' \
  --executor docker \
  --output codegen_report.json
```

The first construction may download released Hugging Face checkpoints through
the official RecursiveMAS resolver. Each task record may provide `task_type`
(`function` or `complete`) and `entry_point`; the adapter forwards these fields
to the official code Planner/Critic/Solver prompts.
