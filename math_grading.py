"""Math answer grading reused from the original CSI evaluator."""
from __future__ import annotations
import math
import re
from typing import Any

def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _balanced_command_argument(text: str, command_start: int) -> tuple[str, int] | None:
    brace_start = text.find("{", command_start)
    if brace_start < 0:
        return None
    depth = 0
    for index in range(brace_start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index], index + 1
    return None


def extract_last_boxed(text: str) -> str:
    candidates: list[tuple[int, str]] = []
    for command in (r"\boxed", r"\fbox"):
        start = 0
        while True:
            index = text.find(command, start)
            if index < 0:
                break
            argument = _balanced_command_argument(text, index + len(command))
            if argument is None:
                break
            content, end = argument
            candidates.append((index, content.strip()))
            start = end
    return max(candidates, key=lambda item: item[0])[1] if candidates else ""


def boxed_output_status(text: str) -> dict:
    """Format diagnostics only; leave the existing extraction/grading rules alone.

    boxed_status describes the LAST attempted boxed/fbox expression. An earlier
    complete answer can coexist with a final incomplete box.
    """
    def escaped(position):
        slashes = 0
        while position > 0 and text[position-1] == '\\':
            slashes += 1; position -= 1
        return slashes % 2 == 1

    statuses = []
    for match in re.finditer(r'\\(?:boxed|fbox)\b', text):
        if escaped(match.start()):
            continue
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text) or text[start] != '{':
            statuses.append('incomplete'); continue
        depth, status = 1, 'incomplete'
        for position in range(start+1, len(text)):
            if text[position] not in '{}' or escaped(position):
                continue
            depth += 1 if text[position] == '{' else -1
            if depth == 0:
                status = 'complete' if text[start+1:position].strip() else 'empty'
                break
        statuses.append(status)
    count = statuses.count('complete')
    return dict(boxed_status=statuses[-1] if statuses else 'missing',
                has_boxed_answer=count > 0, complete_boxed_count=count)


def extract_final_answer(text: str) -> str:
    text = _text(text)
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed

    matches = list(
        re.finditer(
            r"(?is)(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)",
            text,
        )
    )
    if matches:
        candidate = matches[-1].group(1).strip()
        return candidate.splitlines()[0].strip().strip("$ ")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].strip("$ ") if lines else ""


def normalize_answer(text: str) -> str:
    value = extract_final_answer(text)
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\!", "").replace(r"\,", "")
    value = value.replace(r"\;", "").replace(r"\:", "")
    value = value.replace(r"\displaystyle", "")
    value = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", value)
    value = value.replace("$", "").replace(" ", "")
    value = value.rstrip(".,;:")
    while len(value) >= 2 and value[0] == "{" and value[-1] == "}":
        depth = 0
        closes_at_end = False
        for index, character in enumerate(value):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    closes_at_end = index == len(value) - 1
                    break
        if not closes_at_end:
            break
        value = value[1:-1]
    return value.lower()


def _replace_latex_binary_command(text: str, command: str, operator: str) -> str:
    start = 0
    while True:
        index = text.find(command, start)
        if index < 0:
            return text
        first = _balanced_command_argument(text, index + len(command))
        if first is None:
            return text
        numerator, first_end = first
        second = _balanced_command_argument(text, first_end)
        if second is None:
            return text
        denominator, second_end = second
        replacement = f"(({numerator}){operator}({denominator}))"
        text = text[:index] + replacement + text[second_end:]
        start = 0


def _replace_latex_sqrt(text: str) -> str:
    start = 0
    command = r"\sqrt"
    while True:
        index = text.find(command, start)
        if index < 0:
            return text
        argument = _balanced_command_argument(text, index + len(command))
        if argument is None:
            return text
        content, end = argument
        replacement = f"sqrt({content})"
        text = text[:index] + replacement + text[end:]
        start = 0


def _simple_sympy_expression(text: str) -> Any | None:
    value = normalize_answer(text)
    if not value or any(token in value for token in ("\\begin", "\\end", "\\cup")):
        return None
    value = _replace_latex_binary_command(value, r"\frac", "/")
    value = _replace_latex_sqrt(value)
    value = value.replace(r"\cdot", "*").replace(r"\times", "*")
    value = value.replace(r"\pi", "pi").replace(r"\infty", "oo")
    value = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", value)
    value = value.replace("^", "**")
    value = value.replace("{", "(").replace("}", ")")
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    if re.search(r"\\[a-zA-Z]+", value):
        return None
    # Generated answers are data. Permit only a small numeric-expression grammar.
    if re.search(r'[^0-9A-Za-z+*/().,\-\s]', value):
        return None
    if any(name not in {'pi','oo','sqrt'} for name in re.findall(r'[A-Za-z]+',value)):
        return None
    try:
        import sympy

        return sympy.sympify(value, locals={"pi": sympy.pi, "oo": sympy.oo})
    except Exception:
        return None


def normalized_equivalent(reference: str, response: str) -> bool:
    reference_normalized = normalize_answer(reference)
    response_normalized = normalize_answer(response)
    if not reference_normalized or not response_normalized:
        return False
    if reference_normalized == response_normalized:
        return True

    reference_expression = _simple_sympy_expression(reference)
    response_expression = _simple_sympy_expression(response)
    if reference_expression is None or response_expression is None:
        return False
    try:
        import sympy

        difference = sympy.simplify(reference_expression - response_expression)
        if difference == 0:
            return True
        numeric = complex(sympy.N(difference))
        return math.isfinite(numeric.real) and math.isfinite(numeric.imag) and abs(numeric) < 1e-8
    except Exception:
        return False


class MathGrader:
    def __init__(self, mode: str) -> None:
        self.requested_mode = mode
        self._parse = None
        self._verify = None
        if mode in {"auto", "math_verify"}:
            try:
                from math_verify import parse, verify

                self._parse = parse
                self._verify = verify
            except ImportError:
                if mode == "math_verify":
                    raise ImportError(
                        "--grader math_verify requires: pip install math-verify"
                    )
        self.name = (
            "math_verify_with_normalized_fallback"
            if self._parse is not None
            else "normalized_fallback"
        )
        if mode == "auto" and self._parse is None:
            print(
                "WARNING: math-verify is not installed; accuracy uses the less "
                "complete normalized fallback. Install it with: pip install math-verify",
                flush=True,
            )

    def _math_verify_equivalent(self, reference: str, response: str) -> bool | None:
        if self._parse is None or self._verify is None:
            return None
        reference_answer = extract_final_answer(reference)
        response_answer = extract_final_answer(response)
        try:
            gold = self._parse(f"$\\boxed{{{reference_answer}}}$")
            prediction = self._parse(response)
            if not prediction:
                prediction = self._parse(f"$\\boxed{{{response_answer}}}$")
            if not gold or not prediction:
                return None
            return bool(self._verify(gold, prediction))
        except Exception:
            return None

    def equivalent(self, reference: str, response: str) -> bool:
        verified = self._math_verify_equivalent(reference, response)
        if verified is not None:
            return verified
        return normalized_equivalent(reference, response)

