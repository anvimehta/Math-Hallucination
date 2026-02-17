"""
Math LLM Hallucination Validator (early college level).
Uses Wolfram Alpha for ground truth; tabular + FOPC-style reasoning; rule-based + deduction.
Set OPENAI_API_KEY in the environment or in a .env file to use real ChatGPT output.
"""
import ast
import json
import math
import os
import re
import sys
from typing import Any

# Load .env from script directory if present (OPENAI_API_KEY=... one per line, no extra deps)
def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

_load_dotenv()

# Reuse existing Wolfram integration (no extra dependency)
from wolfram_alpha import wolfram_query, extract_wolfram_steps

# -------------------------
# Configuration
# -------------------------
RESULTS_DIR = "wolfram_data"
RESULTS_TABLE = "validation_results.csv"
REASONING_LOG = "reasoning_log.json"
NUMERIC_TOLERANCE = 1e-6

# Early-college math question bank (minimal set)
MATH_QUESTIONS = [
    {"id": "q1", "question": "sqrt((34*52 + 73) - 144/4) + ln(e^5) + cos(0)^2", "category": "arithmetic_trig"},
    {"id": "q2", "question": "integrate x^4 sin(x)", "category": "calculus"},
    {"id": "q3", "question": "solve x^2 - 5x + 6 = 0", "category": "algebra"},
    {"id": "q4", "question": "derivative of x^3 * e^x", "category": "calculus"},
    {"id": "q5", "question": "limit of (sin(x)/x) as x goes to 0", "category": "calculus"},
]


def get_decimal_value_from_ground(ground: dict) -> float | None:
    """Extract single numeric value from Wolfram result (e.g. from 'Decimal approximation' pod)."""
    for pod in ground.get("pods", []):
        if "decimal" in (pod.get("title") or "").lower():
            for sub in pod.get("subpods", []):
                pt = sub.get("plaintext", "") or ""
                nums = extract_numbers(pt)
                if nums:
                    return nums[0]
    # Fallback: use first number from primary/plaintext
    pt = ground.get("raw_plaintext", "") or ""
    nums = extract_numbers(pt)
    return nums[0] if nums else None


def get_ground_truth(question: str) -> dict[str, Any]:
    """Get Wolfram Alpha result as ground truth. Returns structured answer + primary plaintext (+ decimal_value when available)."""
    raw = wolfram_query(question, include_step_by_step=False)
    structured = extract_wolfram_steps(question, raw)
    primary_text = ""
    for pod in structured.get("pods", []):
        if pod.get("primary"):
            for sub in pod.get("subpods", []):
                primary_text = sub.get("plaintext", "") or primary_text
                if primary_text:
                    break
        if primary_text:
            break
    if not primary_text and structured.get("pods"):
        for sub in structured["pods"][0].get("subpods", []):
            primary_text = sub.get("plaintext", "") or primary_text
            if primary_text:
                break
    out = {
        "raw_plaintext": primary_text,
        "definite_result": structured.get("definite_result"),
        "pods": structured.get("pods", []),
    }
    dec = get_decimal_value_from_ground(out)
    if dec is not None:
        out["decimal_value"] = dec
    return out


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values from a string (for comparison)."""
    if not text:
        return []
    # Match integers, decimals, and numbers in scientific notation
    pattern = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
    out = []
    for m in re.finditer(pattern, text):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def normalize_expression(s: str) -> str:
    """Simple normalization for symbolic comparison (e.g. integrals, equations)."""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("**", "^").replace(" ", "")
    return s


def numeric_match(wolfram_text: str, llm_text: str, tolerance: float = NUMERIC_TOLERANCE) -> tuple[bool, str]:
    """
    Rule: numeric_match(a, b) iff the set of numbers in a and b agree within tolerance.
    Returns (match, reason).
    """
    wa = extract_numbers(wolfram_text)
    la = extract_numbers(llm_text)
    if not wa and not la:
        return False, "no_numbers_in_both"
    if not wa:
        return False, "no_numbers_in_ground_truth"
    if not la:
        return False, "no_numbers_in_llm"
    # Compare last/final numeric results often; or all if same count
    w_last, l_last = wa[-1], la[-1]
    if abs(w_last - l_last) <= tolerance:
        return True, f"numeric_match(last: {w_last} ~ {l_last})"
    if len(wa) == len(la) and all(abs(a - b) <= tolerance for a, b in zip(wa, la)):
        return True, "numeric_match(all)"
    return False, f"numeric_mismatch(w={wa}, l={la})"


def symbolic_match(wolfram_text: str, llm_text: str) -> tuple[bool, str]:
    """Rule: symbolic_match(a,b) iff normalized forms are close (substring or equality)."""
    nw = normalize_expression(wolfram_text)
    nl = normalize_expression(llm_text)
    if not nw or not nl:
        return False, "empty_after_normalize"
    if nw == nl:
        return True, "symbolic_exact_match"
    if nw in nl or nl in nw:
        return True, "symbolic_substring_match"
    return False, "symbolic_mismatch"


# -------------------------
# Step-by-step breakdown (order of operations) + FOPC
# -------------------------
def _normalize_expr_for_ast(expr: str) -> str:
    """Prepare expression for ast.parse: ^ -> **, ln( -> log(."""
    s = expr.strip()
    # Replace ^ with ** (careful: don't break **)
    s = re.sub(r"\^", "**", s)
    s = re.sub(r"\bln\s*\(", "log(", s, flags=re.IGNORECASE)
    return s


def _safe_math_eval_with_steps(expr: str) -> tuple[list[dict], float | None]:
    """
    Parse expression and evaluate in order-of-operations order, recording each step.
    Returns (list of {step_index, subexpr, value}, final_value or None on error).
    Only allows numbers, +, -, *, /, **, sqrt, log, cos, sin, tan, e, pi.
    """
    steps: list[dict] = []
    source = _normalize_expr_for_ast(expr)

    def get_src(node: ast.AST) -> str:
        if hasattr(ast, "get_source_segment") and hasattr(node, "end_col_offset"):
            seg = ast.get_source_segment(source, node)
            if seg is not None:
                return seg.strip()
        return ""

    class StepRecorder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[tuple[str, float]] = []
            self.step_list: list[dict] = []

        def visit_Constant(self, node: ast.Constant) -> None:
            v = float(node.value) if isinstance(node.value, (int, float)) else float(node.value)
            self.stack.append((get_src(node) or str(v), v))

        def visit_Name(self, node: ast.Name) -> None:
            name = node.id
            if name == "e":
                self.stack.append(("e", math.e))
            elif name == "pi":
                self.stack.append(("pi", math.pi))
            else:
                raise ValueError(f"Unknown name: {name}")

        def _binop(self, left: tuple[str, float], right: tuple[str, float], op: str, sym: str) -> tuple[str, float]:
            ls, lv = left
            rs, rv = right
            subexpr = f"({ls} {sym} {rs})"
            if op == "add":
                v = lv + rv
            elif op == "sub":
                v = lv - rv
            elif op == "mult":
                v = lv * rv
            elif op == "div":
                v = lv / rv if rv != 0 else float("nan")
            elif op == "pow":
                v = lv**rv
            else:
                raise ValueError(op)
            self.step_list.append({"step_index": len(self.step_list) + 1, "subexpr": subexpr, "value": v})
            return (subexpr, v)

        def visit_BinOp(self, node: ast.BinOp) -> None:
            self.generic_visit(node)
            if len(self.stack) < 2:
                return
            right = self.stack.pop()
            left = self.stack.pop()
            op_map = {
                ast.Add: ("add", "+"),
                ast.Sub: ("sub", "-"),
                ast.Mult: ("mult", "*"),
                ast.Div: ("div", "/"),
                ast.Pow: ("pow", "**"),
            }
            op_type, sym = op_map.get(type(node.op), (None, "?"))
            if op_type is None:
                self.stack.append(left)
                self.stack.append(right)
                return
            res = self._binop(left, right, op_type, sym)
            self.stack.append(res)

        def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
            self.generic_visit(node)
            if not self.stack or not isinstance(node.op, ast.USub):
                return
            sub, v = self.stack.pop()
            res = (-v, f"(-{sub})")
            self.step_list.append({"step_index": len(self.step_list) + 1, "subexpr": res[1], "value": res[0]})
            self.stack.append((res[1], res[0]))

        def visit_Call(self, node: ast.Call) -> None:
            if not isinstance(node.func, ast.Name):
                self.generic_visit(node)
                return
            name = node.func.id
            # Visit only args (do not visit node.func, else Name('sqrt') etc. raises)
            for arg in node.args:
                self.visit(arg)
            args = []
            for _ in node.args:
                if self.stack:
                    args.append(self.stack.pop())
            args.reverse()
            if name == "sqrt" and len(args) == 1:
                subexpr = f"sqrt({args[0][0]})"
                v = math.sqrt(args[0][1])
            elif name in ("log", "ln") and len(args) == 1:
                subexpr = f"log({args[0][0]})"
                v = math.log(args[0][1]) if args[0][1] > 0 else float("nan")
            elif name == "cos" and len(args) == 1:
                subexpr = f"cos({args[0][0]})"
                v = math.cos(args[0][1])
            elif name == "sin" and len(args) == 1:
                subexpr = f"sin({args[0][0]})"
                v = math.sin(args[0][1])
            elif name == "tan" and len(args) == 1:
                subexpr = f"tan({args[0][0]})"
                v = math.tan(args[0][1])
            else:
                self.stack.extend(args)
                return
            self.step_list.append({"step_index": len(self.step_list) + 1, "subexpr": subexpr, "value": v})
            self.stack.append((subexpr, v))

    try:
        tree = ast.parse(source, mode="eval")
        rec = StepRecorder()
        rec.visit(tree)
        if len(rec.stack) != 1:
            return [], None
        final_subexpr, final_value = rec.stack[0]
        # Append final step if we have more than one component (e.g. a+b+c)
        if rec.step_list and rec.step_list[-1].get("subexpr") != final_subexpr:
            rec.step_list.append({"step_index": len(rec.step_list) + 1, "subexpr": final_subexpr, "value": final_value})
        return rec.step_list, final_value
    except Exception:
        return [], None


def step_by_step_breakdown_and_fopc(
    llm_equation: str,
    claimed_target: float,
    actual_value: float,
    tolerance: float = NUMERIC_TOLERANCE,
) -> tuple[list[dict], list[dict], str]:
    """
    Produce step-by-step breakdown and FOPC representation showing why the claim is incorrect.
    Returns (steps_list, fopc_facts, language_explanation).
    """
    steps, computed = _safe_math_eval_with_steps(llm_equation)
    # Use Wolfram actual_value as ground truth when our eval fails or differs
    actual = actual_value if actual_value is not None else computed
    equals_claimed = actual is not None and claimed_target is not None and abs(actual - claimed_target) <= tolerance

    # FOPC: step(i, subexpr, value) for each step; result(actual); claimed(claimed_target);
    # equals(claimed, actual) or ¬equals(claimed, actual); incorrect(claimed) ↔ ¬equals(claimed, actual)
    fopc: list[dict] = []
    for s in steps:
        fopc.append({
            "predicate": "step",
            "args": [s["step_index"], s["subexpr"]],
            "value": s["value"],
            "form": f"step({s['step_index']}, {repr(s['subexpr'])}, {s['value']})",
        })
    fopc.append({"predicate": "result", "args": ["actual"], "value": actual, "form": f"result(actual) = {actual}"})
    fopc.append({"predicate": "claimed", "args": ["target"], "value": claimed_target, "form": f"claimed(target) = {claimed_target}"})
    fopc.append({
        "predicate": "equals",
        "args": ["claimed", "result"],
        "value": equals_claimed,
        "form": "equals(claimed, result)" if equals_claimed else "¬equals(claimed, result)",
    })
    if actual is not None and not equals_claimed:
        fopc.append({
            "predicate": "incorrect",
            "args": ["claimed_value"],
            "value": True,
            "form": f"incorrect({claimed_target}) ← ¬equals(claimed, result) ∧ result = {actual}",
        })

    # Language: step-by-step breakdown then why it is incorrect
    lines = ["Step-by-step (order of operations):"]
    for s in steps:
        lines.append(f"  Step {s['step_index']}: {s['subexpr']} = {s['value']}")
    if actual is not None:
        lines.append(f"  → Final value = {actual}")
    lines.append("")
    if not equals_claimed and actual is not None:
        lines.append(f"Why the claim is incorrect: claimed value = {claimed_target}, but correct value = {actual}. So ¬equals(claimed, result); hence the equation does not equal {claimed_target} (hallucination).")
    else:
        lines.append("The claimed value matches the computed result (no hallucination).")
    explanation = "\n".join(lines)
    return steps, fopc, explanation


# -------------------------
# FOPC-style representation and deduction
# -------------------------
def facts_and_rules(question_id: str, question: str, ground: dict, llm_answer: str, llm_reasoning: str) -> list[dict]:
    """Build a minimal set of logical facts (for trace)."""
    gt_text = ground.get("raw_plaintext", "") or ""
    return [
        {"predicate": "question", "args": [question_id], "value": question},
        {"predicate": "ground_truth", "args": [question_id], "value": gt_text},
        {"predicate": "llm_answer", "args": [question_id], "value": llm_answer},
        {"predicate": "llm_reasoning", "args": [question_id], "value": llm_reasoning[:200] + "..." if len(llm_reasoning) > 200 else llm_reasoning},
    ]


def apply_rules_and_deduce(question_id: str, ground: dict, llm_answer: str) -> tuple[str, list[dict], str]:
    """
    Rule-based reasoning + deduction.
    Returns (verdict, reasoning_trace, language_explanation).
    """
    gt_text = ground.get("raw_plaintext", "") or ""
    trace = []
    language_parts = []

    num_ok, num_reason = numeric_match(gt_text, llm_answer)
    trace.append({"rule": "numeric_match", "result": num_ok, "reason": num_reason})
    language_parts.append(f"Numeric comparison: {num_reason}.")

    sym_ok, sym_reason = symbolic_match(gt_text, llm_answer)
    trace.append({"rule": "symbolic_match", "result": sym_ok, "reason": sym_reason})
    language_parts.append(f"Symbolic comparison: {sym_reason}.")

    # Deduction: not_hallucinating iff (numeric_match OR symbolic_match)
    not_hallucinating = num_ok or sym_ok
    verdict = "not_hallucinating" if not_hallucinating else "hallucinating"
    trace.append({"inference": "verdict", "predicate": verdict, "premise": "numeric_match ∨ symbolic_match"})
    language_parts.append(f"Verdict: {verdict} (by rule: correct if numeric or symbolic match).")

    return verdict, trace, " ".join(language_parts)


# -------------------------
# LLM interface (mock or OpenAI / ChatGPT API)
# -------------------------
def call_llm(question: str) -> dict[str, str]:
    """
    Return {"answer": "...", "reasoning": "..."}.
    Uses OpenAI ChatGPT if OPENAI_API_KEY is set (env or .env); otherwise mock.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            # Prefer official openai package if installed
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a math tutor. Give a short final answer and brief step-by-step reasoning."},
                        {"role": "user", "content": f"Solve step by step, then state the final answer clearly:\n{question}"},
                    ],
                    temperature=0.1,
                )
                text = (resp.choices[0].message.content or "").strip()
            except ImportError:
                import urllib.request
                body = json.dumps({
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a math tutor. Give a short final answer and brief step-by-step reasoning."},
                        {"role": "user", "content": f"Solve step by step, then state the final answer clearly:\n{question}"},
                    ],
                    "temperature": 0.1,
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode())
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if not text:
                return {"answer": "[LLM empty response]", "reasoning": ""}
            # Heuristic: last line or line with "=" as final answer
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            answer = lines[-1] if lines else text
            for line in reversed(lines):
                if "=" in line and len(line) < 120:
                    answer = line
                    break
            return {"answer": answer, "reasoning": text}
        except Exception as e:
            return {"answer": f"[LLM error: {e}]", "reasoning": str(e)}
    # Mock: correct for most, wrong for q1 to demonstrate hallucination detection
    mock_answers = {
        "sqrt((34*52 + 73) - 144/4) + ln(e^5) + cos(0)^2": ("150.0", "Step 1: 34*52=1768, 73, 144/4=36. Step 2: 1768+73-36=1805. Step 3: sqrt(1805)+5+1. So 150.0."),  # wrong
        "integrate x^4 sin(x)": ("integral x^4 sin(x) dx = (4 x (x^2 - 6) sin(x) - (x^4 - 12 x^2 + 24) cos(x)) + C", "By parts repeatedly."),
        "solve x^2 - 5x + 6 = 0": ("x = 2 or x = 3", "Factor (x-2)(x-3)=0."),
        "derivative of x^3 * e^x": ("e^x (x^3 + 3 x^2)", "Product rule: (x^3)' e^x + x^3 (e^x)'."),
        "limit of (sin(x)/x) as x goes to 0": ("1", "Standard limit sin(x)/x -> 1 as x->0."),
    }
    a, r = mock_answers.get(question, ("[no mock answer]", "No reasoning."))
    return {"answer": a, "reasoning": r}


def ask_llm_for_equation(target: float) -> dict[str, Any]:
    """
    Ask ChatGPT for a quite hard equation that equals `target`.
    Returns {"equation": "...", "raw_response": "...", "error": None or str}.
    Requires OPENAI_API_KEY.
    """
    prompt = (
        f"Give a single, quite hard mathematical expression that equals exactly {target}. "
        "Use only one line. Use * for multiplication, sqrt(), ln(), cos(), sin(), ^ for power, and numbers. "
        "Reply with only the expression, no explanation or equals sign."
    )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"equation": None, "raw_response": "", "error": "OPENAI_API_KEY not set"}
    try:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You output only a single mathematical expression, nothing else."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            text = (resp.choices[0].message.content or "").strip()
        except ImportError:
            import urllib.request
            body = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You output only a single mathematical expression, nothing else."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not text:
            return {"equation": None, "raw_response": "", "error": "Empty response"}
        # Extract equation: remove markdown code blocks, take best line
        raw = text
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        equation = None
        for line in lines:
            line = line.strip()
            # Remove trailing = 67 or = target
            line = re.sub(r"\s*=\s*" + re.escape(str(int(target))) + r"\s*$", "", line)
            if len(line) > 5 and re.search(r"[\d+\-*/\^sqrtlncose\s()]+", line):
                equation = line
                break
        if not equation and lines:
            equation = lines[0]
        if not equation:
            equation = raw.strip()[:200]
        return {"equation": equation, "raw_response": raw, "error": None}
    except Exception as e:
        return {"equation": None, "raw_response": "", "error": str(e)}


# -------------------------
# Equation-claim validation (e.g. "LLM said this equation equals 67")
# -------------------------
def validate_equation_claim(
    target: float,
    llm_equation: str,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """
    Validate: LLM claimed an equation equals `target`. Check with Wolfram.
    Returns row with verdict, actual_value, reasoning trace (FOPC-style), and language.
    """
    try:
        ground = get_ground_truth(llm_equation)
    except Exception as e:
        return {
            "question_id": "equation_claim",
            "question": f"equation equals {target}?",
            "llm_equation": llm_equation,
            "claimed_target": target,
            "actual_value": None,
            "wolfram_answer": f"[Wolfram error: {e}]",
            "verdict": "error",
            "reasoning_explanation": str(e),
            "_reasoning_trace": [{"rule": "wolfram_query", "result": False, "reason": str(e)}],
            "_facts": [],
        }
    actual = ground.get("decimal_value")
    gt_text = ground.get("raw_plaintext", "") or ""
    if actual is None:
        actual = extract_numbers(gt_text)[0] if extract_numbers(gt_text) else None
    equals_target = actual is not None and abs(actual - target) <= tolerance
    verdict = "not_hallucinating" if equals_target else "hallucinating"
    reason = (
        f"equation_value({actual}) equals claimed_target({target})"
        if equals_target
        else f"equation_value({actual}) != claimed_target({target})"
    )
    trace = [
        {"rule": "equals_target", "result": equals_target, "reason": reason},
        {"inference": "verdict", "predicate": verdict, "premise": "equals_target ↔ (|actual - target| ≤ tolerance)"},
    ]
    facts = [
        {"predicate": "claimed_target", "args": [], "value": target},
        {"predicate": "llm_equation", "args": [], "value": llm_equation},
        {"predicate": "ground_truth_expression", "args": [], "value": gt_text},
        {"predicate": "actual_value", "args": [], "value": actual},
    ]
    # Step-by-step breakdown and FOPC (why incorrect)
    steps_list, fopc_steps, step_explanation = step_by_step_breakdown_and_fopc(llm_equation, target, actual, tolerance)
    facts.extend([{"predicate": f["predicate"], "args": f["args"], "value": f["value"], "form": f.get("form", "")} for f in fopc_steps])
    explanation = (
        step_explanation + "\n\n"
        f"Wolfram (exact): {gt_text[:150]}. "
        f"Verdict: {verdict} (equation {'equals' if equals_target else 'does not equal'} the claimed value)."
    )
    return {
        "question_id": "equation_claim",
        "question": f"equation equals {target}?",
        "llm_equation": llm_equation,
        "claimed_target": target,
        "actual_value": actual,
        "wolfram_answer": gt_text[:300],
        "verdict": verdict,
        "reasoning_explanation": explanation,
        "step_by_step": steps_list,
        "fopc_step_and_incorrect": fopc_steps,
        "_reasoning_trace": trace,
        "_facts": facts,
    }


# -------------------------
# Word problem validation (Wolfram = ground truth, ChatGPT = LLM answer)
# -------------------------
def validate_word_problem(word_problem: str) -> dict[str, Any]:
    """
    Send the same word problem to Wolfram (ground truth) and to the LLM; compare answers.
    Returns row with verdict, reasoning trace, FOPC-style facts.
    """
    try:
        ground = get_ground_truth(word_problem)
    except Exception as e:
        return {
            "question_id": "word_problem",
            "question": word_problem,
            "wolfram_answer": f"[Wolfram error: {e}]",
            "llm_answer": "",
            "numeric_match": False,
            "symbolic_match": False,
            "verdict": "error",
            "reasoning_explanation": str(e),
            "_reasoning_trace": [{"rule": "wolfram_query", "result": False, "reason": str(e)}],
            "_facts": [],
        }
    llm_out = call_llm(word_problem)
    verdict, trace, explanation = apply_rules_and_deduce("word_problem", ground, llm_out["answer"])
    gt_text = ground.get("raw_plaintext", "") or ""
    facts = [
        {"predicate": "question", "args": ["word_problem"], "value": word_problem},
        {"predicate": "ground_truth", "args": [], "value": gt_text},
        {"predicate": "llm_answer", "args": [], "value": llm_out["answer"]},
        {"predicate": "llm_reasoning", "args": [], "value": (llm_out["reasoning"] or "")[:300]},
    ]
    for t in trace:
        facts.append({
            "predicate": t.get("rule") or t.get("inference"),
            "value": t.get("result"),
            "reason": t.get("reason") or t.get("premise"),
        })
    return {
        "question_id": "word_problem",
        "question": word_problem,
        "wolfram_answer": gt_text[:300],
        "llm_answer": (llm_out["answer"] or "")[:300],
        "llm_reasoning": (llm_out["reasoning"] or "")[:500],
        "numeric_match": trace[0]["result"] if trace else False,
        "symbolic_match": trace[1]["result"] if len(trace) > 1 else False,
        "verdict": verdict,
        "reasoning_explanation": explanation,
        "_reasoning_trace": trace,
        "_facts": facts,
    }


# -------------------------
# Tabular output and run
# -------------------------
def run_validation(questions: list[dict] = None, use_cache: bool = True) -> list[dict]:
    """Run full pipeline: ground truth (Wolfram), LLM, rules, deduction; return table rows."""
    questions = questions or MATH_QUESTIONS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache_file = os.path.join(RESULTS_DIR, "ground_truth_cache.json")
    if use_cache and os.path.isfile(cache_file):
        with open(cache_file) as f:
            gt_cache = json.load(f)
    else:
        gt_cache = {}

    rows = []
    for item in questions:
        qid = item["id"]
        q = item["question"]
        if q not in gt_cache:
            try:
                gt_cache[q] = get_ground_truth(q)
            except Exception as e:
                gt_cache[q] = {"raw_plaintext": f"[Wolfram error: {e}]", "definite_result": None, "pods": []}
        ground = gt_cache[q]
        llm_out = call_llm(q)
        verdict, trace, explanation = apply_rules_and_deduce(qid, ground, llm_out["answer"])

        row = {
            "question_id": qid,
            "question": q,
            "category": item.get("category", ""),
            "wolfram_answer": (ground.get("raw_plaintext") or "")[:300],
            "llm_answer": (llm_out["answer"] or "")[:300],
            "numeric_match": trace[0]["result"] if trace else False,
            "symbolic_match": trace[1]["result"] if len(trace) > 1 else False,
            "verdict": verdict,
            "reasoning_explanation": explanation,
        }
        row["_reasoning_trace"] = trace
        row["_facts"] = facts_and_rules(qid, q, ground, llm_out["answer"], llm_out["reasoning"])
        rows.append(row)

    if use_cache:
        # Save cache without non-JSON-serializable content; keep only raw_plaintext for size
        save_cache = {q: {"raw_plaintext": g.get("raw_plaintext"), "definite_result": g.get("definite_result")} for q, g in gt_cache.items()}
        with open(cache_file, "w") as f:
            json.dump(save_cache, f, indent=2)
    return rows


def write_table_and_log(rows: list[dict]) -> None:
    """Write CSV table and reasoning log (FOPC-style + trace)."""
    import csv
    table_path = os.path.join(RESULTS_DIR, RESULTS_TABLE)
    log_path = os.path.join(RESULTS_DIR, REASONING_LOG)

    # CSV: drop internal keys
    table_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=table_rows[0].keys() if table_rows else [])
        w.writeheader()
        w.writerows(table_rows)
    print(f"Table: {table_path}")

    log = {
        "representation": "FOPC-style facts + rule-based deduction",
        "questions": [
            {
                "id": r["question_id"],
                "facts": r.get("_facts", []),
                "reasoning_trace": r.get("_reasoning_trace", []),
                "verdict": r["verdict"],
            }
            for r in rows
        ],
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Reasoning log: {log_path}")


def main():
    argv = sys.argv[1:]
    # Word problem: send to Wolfram + ChatGPT, compare answers
    if argv and argv[0] == "--word-problem":
        if len(argv) < 2:
            print("Usage: python math_hallucination_validator.py --word-problem \"YOUR WORD PROBLEM\"")
            print('Example: python math_hallucination_validator.py --word-problem "A train leaves at 2pm at 60 mph. Another leaves at 3pm from the same station at 80 mph. When does the second catch the first?"')
            return 1
        word_problem = " ".join(argv[1:]).strip()
        if not word_problem:
            print("Provide a non-empty word problem in quotes.")
            return 1
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is required for --word-problem (to get ChatGPT's answer). Set it in .env or export it.")
            return 1
        print("Math LLM Hallucination Validator — word problem")
        print(f"Problem: {word_problem[:200]}{'...' if len(word_problem) > 200 else ''}\n")
        print("Getting ground truth from Wolfram...")
        print("Getting answer from ChatGPT...\n")
        row = validate_word_problem(word_problem)
        print(f"Wolfram: {row.get('wolfram_answer', '')[:300]}")
        print(f"ChatGPT: {row.get('llm_answer', '')[:300]}")
        if row.get("llm_reasoning"):
            print(f"ChatGPT reasoning (excerpt): {row['llm_reasoning'][:200]}...")
        print(f"\nVerdict: {row['verdict']}  (numeric_match={row.get('numeric_match')}, symbolic_match={row.get('symbolic_match')})")
        print(f"Reasoning: {row['reasoning_explanation']}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "word_problem_log.json")
        with open(log_path, "w") as f:
            json.dump(
                {
                    "question": row["question"],
                    "wolfram_answer": row["wolfram_answer"],
                    "llm_answer": row["llm_answer"],
                    "llm_reasoning": row.get("llm_reasoning"),
                    "verdict": row["verdict"],
                    "numeric_match": row.get("numeric_match"),
                    "symbolic_match": row.get("symbolic_match"),
                    "reasoning_explanation": row["reasoning_explanation"],
                    "facts": row.get("_facts", []),
                    "reasoning_trace": row.get("_reasoning_trace", []),
                },
                f,
                indent=2,
            )
        print(f"\nLog: {log_path}")
        return 0

    # Ask ChatGPT for an equation that equals TARGET, then validate it
    if argv and argv[0] == "--ask-equation":
        if len(argv) < 2:
            print("Usage: python math_hallucination_validator.py --ask-equation TARGET")
            print("Example: python math_hallucination_validator.py --ask-equation 67")
            return 1
        try:
            target = float(argv[1])
        except ValueError:
            print("TARGET must be a number (e.g. 67)")
            return 1
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is required for --ask-equation. Set it in .env or export it.")
            return 1
        print(f"Asking ChatGPT for a quite hard equation that equals {target}...\n")
        out = ask_llm_for_equation(target)
        if out.get("error"):
            print("Error:", out["error"])
            if out.get("raw_response"):
                print("Raw response:", out["raw_response"][:300])
            return 1
        llm_equation = out["equation"]
        print(f"ChatGPT gave: {llm_equation}\n")
        print("Validating with Wolfram (step-by-step + FOPC)...\n")
        row = validate_equation_claim(target, llm_equation)
        print(f"Wolfram (exact): {row.get('wolfram_answer', '')[:200]}...")
        print(f"Wolfram (decimal): {row.get('actual_value')}")
        print(f"Verdict: {row['verdict']}\n")
        print("--- Step-by-step (order of operations) ---")
        for s in row.get("step_by_step", []):
            print(f"  Step {s['step_index']}: {s['subexpr']} = {s['value']}")
        print("\n--- FOPC (why incorrect/correct) ---")
        for f in row.get("fopc_step_and_incorrect", []):
            print(f"  {f.get('form', f)}")
        print(f"\n{row['reasoning_explanation']}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "equation_claim_log.json")
        with open(log_path, "w") as f:
            json.dump(
                {
                    "prompt_target": target,
                    "llm_equation": row["llm_equation"],
                    "claimed_target": row["claimed_target"],
                    "actual_value": row["actual_value"],
                    "verdict": row["verdict"],
                    "step_by_step": row.get("step_by_step", []),
                    "fopc": row.get("fopc_step_and_incorrect", []),
                    "facts": row.get("_facts", []),
                    "reasoning_trace": row.get("_reasoning_trace", []),
                },
                f,
                indent=2,
            )
        print(f"\nReasoning log: {log_path}")
        return 0

    # Equation-claim mode: --equation-claim TARGET "EXPRESSION"
    if argv and argv[0] == "--equation-claim":
        if len(argv) < 3:
            print("Usage: python math_hallucination_validator.py --equation-claim TARGET \"EXPRESSION\"")
            print('Example: python math_hallucination_validator.py --equation-claim 67 "sqrt((34*52 + 73) - 144/4) + ln(e^5) + cos(0)^2"')
            return 1
        try:
            target = float(argv[1])
        except ValueError:
            print("TARGET must be a number (e.g. 67)")
            return 1
        llm_equation = argv[2]
        print("Math LLM Hallucination Validator — equation claim")
        print(f"Claim: the following equation equals {target}")
        print(f"LLM equation: {llm_equation}\n")
        row = validate_equation_claim(target, llm_equation)
        print(f"Wolfram (exact): {row.get('wolfram_answer', '')[:200]}...")
        print(f"Wolfram (decimal): {row.get('actual_value')}")
        print(f"Verdict: {row['verdict']}\n")
        print("--- Step-by-step (order of operations) ---")
        for s in row.get("step_by_step", []):
            print(f"  Step {s['step_index']}: {s['subexpr']} = {s['value']}")
        print("\n--- FOPC (why incorrect) ---")
        for f in row.get("fopc_step_and_incorrect", []):
            print(f"  {f.get('form', f)}")
        print(f"\n{row['reasoning_explanation']}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "equation_claim_log.json")
        with open(log_path, "w") as f:
            json.dump(
                {
                    "claimed_target": row["claimed_target"],
                    "llm_equation": row["llm_equation"],
                    "actual_value": row["actual_value"],
                    "verdict": row["verdict"],
                    "step_by_step": row.get("step_by_step", []),
                    "fopc": row.get("fopc_step_and_incorrect", []),
                    "facts": row.get("_facts", []),
                    "reasoning_trace": row.get("_reasoning_trace", []),
                },
                f,
                indent=2,
            )
        print(f"\nReasoning log: {log_path}")
        return 0

    print("Math LLM Hallucination Validator")
    print("Ground truth: Wolfram Alpha | Reasoning: rule-based + deduction")
    if os.environ.get("OPENAI_API_KEY"):
        print("LLM: ChatGPT (OpenAI API)\n")
    else:
        print("LLM: mock (set OPENAI_API_KEY or add to .env for real ChatGPT)\n")
    rows = run_validation(use_cache=True)
    write_table_and_log(rows)
    print("\n--- Summary ---")
    for r in rows:
        print(f"  {r['question_id']}: {r['verdict']}  (numeric={r['numeric_match']}, symbolic={r['symbolic_match']})")
    print("\nReasoning representation: tabular (CSV) + FOPC-style facts and deduction trace in", REASONING_LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
