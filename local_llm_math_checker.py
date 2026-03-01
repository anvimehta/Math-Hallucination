"""
Local LLM math answer checker (no Wolfram Alpha).

This tool does two things:
1. Calls an LLM (via OPENAI_API_KEY) with a math word problem.
2. Tries to extract and safely evaluate the expression in the LLM's answer,
   then compares the numeric result to a target you supply to decide whether
   the LLM is hallucinating.

You supply:
    - the question text
    - the correct numeric target

The script:
    - sends the question to the LLM
    - gets the answer text
    - extracts an expression from the answer
    - evaluates it locally
    - reports hallucinating / not hallucinating
"""

from __future__ import annotations

import json
import ast
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any


NUMERIC_TOLERANCE = 1e-6


def _load_dotenv() -> None:
    """
    Load simple KEY=VALUE lines from a local .env file into the environment
    (only if a key is not already set).
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()


@dataclass
class CheckResult:
    is_correct: bool
    expression: str | None
    value: float | None
    target: float
    difference: float | None
    error: str | None = None


def _extract_latex_expression(answer: str) -> str | None:
    """If the answer contains a LaTeX inline math segment \\( ... \\), return the last such segment."""
    matches = list(re.finditer(r"\\\((.+?)\\\)", answer, flags=re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _extract_expression_from_text(answer: str) -> str | None:
    """
    Try to extract a math expression from an LLM's natural-language answer.
    Heuristics:
      - Prefer LaTeX inline math \\( ... \\)
      - Otherwise, look for the last line that contains digits and math operators
      - If that line contains '=', take the left-hand side as the expression
    """
    # 1) LaTeX inline math
    latex_expr = _extract_latex_expression(answer)
    if latex_expr:
        # If there's an '=', take the left-hand side as the expression
        if "=" in latex_expr:
            return latex_expr.split("=", 1)[0].strip()
        return latex_expr

    # 2) Non-LaTeX: scan lines
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    candidate = None
    for line in reversed(lines):
        if re.search(r"[0-9]", line) and re.search(r"[+\-*/÷×()]", line):
            candidate = line
            break
    if not candidate:
        return None
    if "=" in candidate:
        return candidate.split("=", 1)[0].strip()
    return candidate


def _normalize_expression(expr: str) -> str:
    """
    Normalize an arithmetic expression string into something Python's AST can parse.
    Handles:
      - LaTeX / Unicode division (\\div, ÷) and multiplication (×)
      - Implicit whitespace
    """
    s = expr.strip()
    # Replace common division / multiplication markers with Python operators
    s = s.replace("\\div", "/").replace("÷", "/").replace("×", "*")
    # Remove repeated spaces
    s = re.sub(r"\s+", " ", s)
    return s


class SafeEvaluator(ast.NodeVisitor):
    """
    Very small safe evaluator for arithmetic expressions.
    Supports: +, -, *, /, **, parentheses, and numeric literals.
    """

    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    allowed_unaryops = (ast.UAdd, ast.USub)

    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, self.allowed_binops):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self.allowed_unaryops):
            operand = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
        # Python 3.8+ uses Constant; older versions used Num. Support both without
        # depending on ast.Num existing on newer interpreters.
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # pragma: no cover
        raise ValueError(f"Disallowed or unsupported expression node: {ast.dump(node)}")


def safe_eval(expr: str) -> float:
    """Safely evaluate a simple arithmetic expression and return a float."""
    normalized = _normalize_expression(expr)
    try:
        # Use eval mode so the top-level is a single expression
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Cannot parse expression: {normalized!r} ({e})") from e
    evaluator = SafeEvaluator()
    value = evaluator.visit(tree)
    return float(value)


def check_llm_answer(answer_text: str, target: float, tolerance: float = NUMERIC_TOLERANCE) -> CheckResult:
    """
    Check whether an LLM answer is hallucinating with respect to a known numeric target.

    - answer_text: full text returned by the LLM (answer + reasoning)
    - target: the correct numeric value (e.g. 6.0)
    - tolerance: allowed numeric difference
    """
    expr = _extract_expression_from_text(answer_text)
    if not expr:
        return CheckResult(
            is_correct=False,
            expression=None,
            value=None,
            target=target,
            difference=None,
            error="Could not extract a numeric expression from the LLM answer.",
        )
    try:
        value = safe_eval(expr)
    except Exception as e:
        return CheckResult(
            is_correct=False,
            expression=expr,
            value=None,
            target=target,
            difference=None,
            error=f"Failed to evaluate expression {expr!r}: {e}",
        )
    diff = value - target
    is_ok = math.isfinite(value) and abs(diff) <= tolerance
    return CheckResult(
        is_correct=is_ok,
        expression=expr,
        value=value,
        target=target,
        difference=diff,
        error=None,
    )


def _parse_literal_constraint(question_text: str) -> tuple[int, int] | None:
    """
    Try to infer a constraint of the form "use N Ks" from the question text,
    e.g. "use four 10s", "use 3 7s", "with three 5s", etc.

    Returns (required_count, required_literal_value) or None if no pattern found.
    """
    q = question_text.lower()
    word_to_int = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    # Pattern 1: "use four 10s", "with 3 7s", etc.
    m = re.search(r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(\d+)\s*s\b", q)
    if m:
        count_word, literal_str = m.groups()
        if count_word.isdigit():
            count = int(count_word)
        else:
            count = word_to_int.get(count_word, 0)
        if count > 0:
            literal_val = int(literal_str)
            return count, literal_val
    # Pattern 2: "use four tens" (implicitly 10s)
    m = re.search(r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+tens?\b", q)
    if m:
        count_word = m.group(1)
        if count_word.isdigit():
            count = int(count_word)
        else:
            count = word_to_int.get(count_word, 0)
        if count > 0:
            return count, 10
    return None


def call_llm(question: str) -> dict[str, str]:
    """
    Call an OpenAI-compatible chat model to solve the math question.

    Returns:
        {"answer": short_line, "reasoning": full_text}
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (in environment or .env).")

    system_msg = "You are a math tutor. Give a short final answer and brief step-by-step reasoning."
    user_msg = f"Solve step by step, then state the final answer clearly:\n{question}"

    try:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip()
        except ImportError:
            import urllib.request

            body = json.dumps(
                {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.1,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:  # network / API errors
        raise RuntimeError(f"Error calling LLM: {e}") from e

    if not text:
        return {"answer": "[LLM empty response]", "reasoning": ""}

    # Use the same heuristic as in the validator: last line, preferring short lines with '='.
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    answer = lines[-1] if lines else text
    for line in reversed(lines):
        if "=" in line and len(line) < 120:
            answer = line
            break
    return {"answer": answer, "reasoning": text}


def _cli(argv: list[str]) -> int:
    """
    Simple command-line interface.

    Usage:
        # Non-interactive (target and question on CLI)
        python local_llm_math_checker.py <target> "<question_text>"

        # Interactive (target only, then you'll be prompted for the question)
        python local_llm_math_checker.py <target>

    Examples:
        python local_llm_math_checker.py 6 "Use four 10s and basic operations to make 6."
        python local_llm_math_checker.py 6
    """
    if len(argv) < 2:
        print("Usage: python local_llm_math_checker.py <target> \"<question_text>\"")
        print("   or: python local_llm_math_checker.py <target>   # then enter question interactively")
        return 1
    try:
        target = float(argv[1])
    except ValueError:
        print(f"Target must be numeric, got: {argv[1]!r}")
        return 1
    if len(argv) >= 3:
        question_text = " ".join(argv[2:])
    else:
        # Interactive prompt for the question text
        try:
            question_text = input("Enter your math question for the LLM: ").strip()
        except EOFError:
            print("No question provided.")
            return 1
        if not question_text:
            print("No question provided.")
            return 1

    try:
        llm = call_llm(question_text)
    except RuntimeError as e:
        print(e)
        return 1

    combined_answer = f"{llm['answer']}\n{llm['reasoning']}"
    result = check_llm_answer(combined_answer, target=target)

    print(f"Question: {question_text}")
    print(f"LLM short answer line: {llm['answer']}")
    print("\n--- Full LLM reasoning ---")
    print(llm["reasoning"])
    print("\n--- Local numeric check ---")
    print(f"Target: {result.target}")
    print(f"Extracted expression: {result.expression!r}")
    print(f"Evaluated value: {result.value}")
    if result.error:
        print(f"Error: {result.error}")
        print("Verdict: hallucinating (cannot verify expression against target).")
        return 0

    # Optional constraint: "use N Ks" style problems (e.g. "use four 10s", "use 3 sevens")
    constraint_ok = True
    constraint_reason = ""
    parsed = _parse_literal_constraint(question_text)
    if parsed is not None:
        required_count, required_literal = parsed
        if not result.expression:
            constraint_ok = False
            constraint_reason = (
                f"No expression extracted; cannot verify 'use {required_count} {required_literal}s' constraint."
            )
        else:
            nums = [int(m.group()) for m in re.finditer(r"\d+", result.expression)]
            count_required = sum(1 for n in nums if n == required_literal)
            other_nums = [n for n in nums if n != required_literal]
            if count_required != required_count or other_nums:
                constraint_ok = False
                constraint_reason = (
                    f"'use {required_count} {required_literal}s' constraint violated: "
                    f"found {count_required} occurrences of {required_literal} "
                    f"and other literals {other_nums} in expression {result.expression!r}"
                )

    print(f"Difference (value - target): {result.difference}")
    if not constraint_ok:
        print(f"Constraint check failed: {constraint_reason}")
        print("Verdict: hallucinating (constraint violated, even though numeric value matches).")
    else:
        print(f"Verdict: {'not hallucinating' if result.is_correct else 'hallucinating'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))

