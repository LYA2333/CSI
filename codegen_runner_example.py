"""Runner interface example; replace these smoke-test bodies with real models."""

from __future__ import annotations

import time
from typing import Any, Mapping


class ExampleRunner:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        # Load tokenizer/model/checkpoints here. Factory construction is not timed.
        self.delay_seconds = delay_seconds

    def generate(
        self,
        prompt: str,
        task_id: str,
        task: Mapping[str, Any],
    ) -> str:
        # Replace with one complete RecursiveMAS or CSI-GCN inference request.
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if task_id == "add_two_numbers":
            return "def add(a, b):\n    return a + b"
        if task_id == "factorial":
            return (
                "def factorial(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    result = 1\n"
                "    for value in range(2, n + 1):\n"
                "        result *= value\n"
                "    return result"
            )
        return "raise NotImplementedError('example runner does not know this task')"


def build_recursive_runner(**kwargs):
    return ExampleRunner(delay_seconds=0.01, **kwargs)


def build_ours_runner(**kwargs):
    return ExampleRunner(delay_seconds=0.005, **kwargs)
