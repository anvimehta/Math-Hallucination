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
from wolfram_alpha import wolfram_query, wolfram_short_answer, extract_wolfram_steps

# Real FOPC engine (unification + resolution)
from fopc import (
    fopc_deduce_verdict,
    fopc_deduce_equation_verdict,
    fopc_abduce_causes,
    fopc_deduce_evidence_strength,
    fopc_build_inference_chain,
    fopc_aggregate_result,
    fopc_deduce_diverges_at,
    fopc_deduce_priority,
    fopc_deduce_error_verdict,
    fopc_expect_symbolic,
)

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
    # Fallback: try to interpret simple fractions like "1/20" from primary/plaintext
    pt = ground.get("raw_plaintext", "") or ""
    if pt:
        frac = re.search(r"(-?\\d+)\\s*/\\s*(\\d+)", pt)
        if frac:
            try:
                num = float(frac.group(1))
                den = float(frac.group(2))
                if den != 0:
                    return num / den
            except ValueError:
                pass
    # Final fallback: use first number from primary/plaintext
    nums = extract_numbers(pt)
    return nums[0] if nums else None


def _plaintext_from_raw_pods(raw_query_result: dict) -> str:
    """Extract the best plaintext answer from raw Wolfram queryresult pods.
    Prefers pods with answer-like titles; else shortest plaintext that contains a number.
    """
    pods = raw_query_result.get("queryresult", {}).get("pods", [])
    answer_titles = ("result", "solution", "answer", "exact result", "decimal approximation", "value", "root", "decimal form")
    preferred = ""
    all_plaintexts: list[tuple[str, str]] = []  # (title_lower, plaintext)
    for pod in pods:
        title = (pod.get("title") or "").lower()
        for sub in pod.get("subpods", []):
            pt = (sub.get("plaintext") or "").strip()
            if not pt:
                continue
            all_plaintexts.append((title, pt))
            if any(t in title for t in answer_titles):
                if not preferred or len(pt) < len(preferred):
                    preferred = pt
    if preferred:
        return preferred
    if not all_plaintexts:
        return ""
    # Prefer shortest plaintext that contains a digit (likely the direct answer)
    with_num = [(t, p) for t, p in all_plaintexts if re.search(r"\d", p)]
    if with_num:
        return min(with_num, key=lambda x: len(x[1]))[1]
    return all_plaintexts[0][1]


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
    # Fallback: scan all structured pods for any plaintext
    if not primary_text:
        for pod in structured.get("pods", []):
            for sub in pod.get("subpods", []):
                pt = (sub.get("plaintext") or "").strip()
                if pt:
                    primary_text = pt
                    break
            if primary_text:
                break
    # Fallback: read directly from raw API (handles different pod structure / word problems)
    if not primary_text and raw:
        primary_text = _plaintext_from_raw_pods(raw)
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


def relative_error_and_confidence(wolfram_text: str, llm_text: str) -> tuple[float | None, str, str]:
    """
    Aggregation/statistical: compute relative error between ground truth and LLM answer (when both are numeric).
    Returns (relative_error or None, confidence_level, explanation).
    """
    wa = extract_numbers(wolfram_text)
    la = extract_numbers(llm_text)
    if not wa or not la:
        return None, "unknown", "Cannot compute relative error (missing numbers)."
    w_last, l_last = wa[-1], la[-1]
    if abs(w_last) < 1e-12:
        return None, "unknown", "Ground truth near zero; relative error undefined."
    rel_err = abs(w_last - l_last) / abs(w_last)
    if rel_err <= NUMERIC_TOLERANCE:
        return rel_err, "high", f"Relative error ≈ {rel_err:.2e} (match)."
    if rel_err < 0.01:
        confidence = "medium"
        expl = f"Relative error ≈ {rel_err:.2%}; small numeric deviation."
    elif rel_err < 0.5:
        confidence = "high"
        expl = f"Relative error ≈ {rel_err:.2%}; moderate evidence of hallucination."
    else:
        confidence = "very_high"
        expl = f"Relative error ≈ {rel_err:.2%}; strong evidence of hallucination."
    return rel_err, confidence, expl


def abduce_causes(numeric_ok: bool, symbolic_ok: bool, gt_text: str, llm_text: str) -> list[dict]:
    """
    Abduction: infer possible causes for hallucination (why LLM answer might disagree with ground truth).
    Returns list of FOPC-style hypotheses.
    """
    causes = []
    if not numeric_ok and extract_numbers(gt_text) and extract_numbers(llm_text):
        causes.append({
            "predicate": "abduced_cause",
            "hypothesis": "numeric_error",
            "form": "LLM may have made a numerical mistake (wrong constant, wrong operation, or rounding).",
        })
    if not symbolic_ok:
        causes.append({
            "predicate": "abduced_cause",
            "hypothesis": "symbolic_error",
            "form": "LLM may have used a different (incorrect) symbolic form or expression.",
        })
    if not numeric_ok and not symbolic_ok:
        causes.append({
            "predicate": "abduced_cause",
            "hypothesis": "both_numeric_and_symbolic",
            "form": "Both numeric and symbolic mismatch; hallucination likely substantive.",
        })
    if numeric_ok and not symbolic_ok:
        causes.append({
            "predicate": "abduced_cause",
            "hypothesis": "form_difference_only",
            "form": "Numbers agree but form differs (e.g. equivalent expression in different form).",
        })
    return causes


def build_inference_chain(gt_text: str, llm_answer: str, num_ok: bool, sym_ok: bool, verdict: str) -> list[dict]:
    """
    Explicit logical inference chain (FOPC): premises → rule application → conclusion.
    """
    chain = [
        {"step": 1, "type": "premise", "form": "P1: ground_truth(G)", "value": gt_text[:80]},
        {"step": 2, "type": "premise", "form": "P2: llm_answer(L)", "value": llm_answer[:80]},
        {"step": 3, "type": "observation", "form": f"P3: numeric_match(G,L) = {num_ok}", "value": num_ok},
        {"step": 4, "type": "observation", "form": f"P4: symbolic_match(G,L) = {sym_ok}", "value": sym_ok},
        {"step": 5, "type": "rule", "form": "R: not_hallucinating ↔ (P3 ∨ P4)", "value": None},
        {"step": 6, "type": "conclusion", "form": f"C: verdict = {verdict}", "value": verdict},
    ]
    return chain


# -------------------------
# Step-by-step breakdown (order of operations) + FOPC
# -------------------------
def _normalize_expr_for_ast(expr: str) -> str:
    """Prepare expression for ast.parse: ^ -> **, ln( -> log(, π -> pi."""
    s = expr.strip()
    # Replace ^ with ** (careful: don't break **)
    s = re.sub(r"\^", "**", s)
    s = re.sub(r"\bln\s*\(", "log(", s, flags=re.IGNORECASE)
    # Greek π (U+03C0) / Π (U+03A0) as identifier → pi so visit_Name recognizes it
    s = s.replace("\u03c0", "pi").replace("\u03a0", "pi")
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
            self.step_list.append({
                "step_index": len(self.step_list) + 1,
                "subexpr": subexpr,
                "value": v,
                "op": op,
                "children": [ls, rs],
            })
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
                subexpr = f"ln({args[0][0]})"  # natural log; display as ln
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
            self.step_list.append({
                "step_index": len(self.step_list) + 1,
                "subexpr": subexpr,
                "value": v,
                "op": name,
                "children": [args[0][0]],
            })
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


def _back_solve_required_values(
    steps: list[dict],
    claimed_target: float,
    tolerance: float = NUMERIC_TOLERANCE,
) -> list[dict]:
    """
    Back-solve from claimed_target: at each sub-expression, what value would it need
    for the whole expression to equal claimed_target? Compare to actual (derived ourselves).
    Returns list of {subexpr, actual, required, discrepancy, step_index} for localization.
    """
    if not steps:
        return []
    # Map subexpr -> {value, op, children}; prefer keeping op/children when a duplicate final step (no op) overwrites
    by_subexpr: dict[str, dict] = {}
    for s in steps:
        subexpr = s.get("subexpr")
        if not subexpr:
            continue
        existing = by_subexpr.get(subexpr)
        op = s.get("op") or (existing.get("op") if existing else None)
        children = s.get("children") or (existing.get("children") if existing else None) or []
        by_subexpr[subexpr] = {"value": s["value"], "op": op, "children": children}
    # Root is the last step that has op (full expression; sometimes last step is a duplicate without op)
    root_subexpr = None
    for s in reversed(steps):
        subexpr = s.get("subexpr")
        if subexpr and by_subexpr.get(subexpr, {}).get("op"):
            root_subexpr = subexpr
            break
    if not root_subexpr:
        return []
    def _child_value(sub: str):
        v = by_subexpr.get(sub, {}).get("value")
        if v is not None:
            return v
        try:
            return float(sub)
        except (TypeError, ValueError):
            return None

    required: dict[str, float] = {root_subexpr: claimed_target}
    # Process steps in reverse order so parent required is set before we compute children
    for s in reversed(steps):
        subexpr = s.get("subexpr")
        if subexpr not in required or subexpr not in by_subexpr:
            continue
        R = required[subexpr]
        info = by_subexpr[subexpr]
        op, children = info.get("op"), info.get("children")
        if not op or not children:
            continue
        val = info["value"]
        if op == "add" and len(children) >= 2:
            left_sub, right_sub = children[0], children[1]
            left_val = _child_value(left_sub)
            right_val = _child_value(right_sub)
            if left_val is not None and right_val is not None:
                required[left_sub] = R - right_val
                required[right_sub] = R - left_val
        elif op == "sub" and len(children) >= 2:
            left_sub, right_sub = children[0], children[1]
            left_val = _child_value(left_sub)
            right_val = _child_value(right_sub)
            if left_val is not None and right_val is not None:
                required[left_sub] = R + right_val
                required[right_sub] = left_val - R
        elif op == "mult" and len(children) >= 2:
            left_sub, right_sub = children[0], children[1]
            left_val = _child_value(left_sub)
            right_val = _child_value(right_sub)
            if left_val is not None and right_val is not None and abs(right_val) > 1e-12 and abs(left_val) > 1e-12:
                required[left_sub] = R / right_val
                required[right_sub] = R / left_val
        elif op == "div" and len(children) >= 2:
            left_sub, right_sub = children[0], children[1]
            left_val = _child_value(left_sub)
            right_val = _child_value(right_sub)
            if right_val is not None and abs(right_val) > 1e-12 and left_val is not None:
                required[left_sub] = R * right_val
                required[right_sub] = left_val / R
        elif op == "pow" and len(children) >= 2:
            left_sub, right_sub = children[0], children[1]
            left_val = _child_value(left_sub)
            right_val = _child_value(right_sub)
            if left_val is not None and right_val is not None and left_val > 0 and R > 0:
                try:
                    required[left_sub] = R ** (1.0 / right_val)
                    required[right_sub] = math.log(R) / math.log(left_val) if left_val != 1 else 0
                except (ZeroDivisionError, ValueError):
                    pass
        elif op == "sqrt" and len(children) >= 1 and R >= 0:
            required[children[0]] = R * R
        elif op in ("log", "ln") and len(children) >= 1:
            try:
                required[children[0]] = math.exp(R)
            except OverflowError:
                pass
        # cos, sin, tan: skip inverse (required for arg not uniquely defined for localization)
    # Build localization list: subexpr, actual, required, discrepancy
    out = []
    for s in steps:
        subexpr = s.get("subexpr")
        if not subexpr or subexpr not in by_subexpr:
            continue
        actual = by_subexpr[subexpr]["value"]
        req = required.get(subexpr)
        if req is None:
            continue
        disc = req - actual
        if abs(disc) < tolerance and abs(actual - claimed_target) < tolerance:
            continue
        out.append({
            "step_index": s.get("step_index"),
            "subexpr": subexpr,
            "actual_value": actual,
            "required_value_to_reach_target": req,
            "discrepancy": disc,
            "abs_discrepancy": abs(disc),
        })
    # Combo that sums to the gap = root's direct children; their (actual - required) sum to total_gap
    root_children = by_subexpr.get(root_subexpr, {}).get("children") or []
    actual_final = by_subexpr.get(root_subexpr, {}).get("value")
    total_gap = (actual_final - claimed_target) if actual_final is not None else None
    # Ensure every root child is in out (add synthetic entries for constants so combo sum is exact)
    out_subexprs = {loc["subexpr"] for loc in out}
    for sub in root_children:
        if sub in out_subexprs:
            continue
        req = required.get(sub)
        if req is None:
            continue
        actual_child = _child_value(sub)
        if actual_child is None:
            continue
        disc = req - actual_child
        out.append({
            "step_index": 9999,  # synthetic (constant) so combo list is complete
            "subexpr": sub,
            "actual_value": actual_child,
            "required_value_to_reach_target": req,
            "discrepancy": disc,
            "abs_discrepancy": abs(disc),
        })
        out_subexprs.add(sub)
    for loc in out:
        loc["in_gap_combo"] = loc["subexpr"] in root_children
    # If no child is in out (e.g. constants), treat root as the single "combo" member so sum still equals gap
    if not any(loc.get("in_gap_combo") for loc in out) and len(out) >= 1 and total_gap is not None:
        for loc in out:
            if loc["subexpr"] == root_subexpr:
                loc["in_gap_combo"] = True
                break
    root_op = by_subexpr.get(root_subexpr, {}).get("op")
    # For subtraction: (actual_L - required_L) + (actual_R - required_R) = 0 (always). So include only the child whose contribution equals total_gap.
    if root_op == "sub" and total_gap is not None and root_children:
        for loc in out:
            if not loc.get("in_gap_combo"):
                continue
            contrib = loc["actual_value"] - loc["required_value_to_reach_target"]
            # Keep in combo only if this part's contribution equals the total gap (the one that "accounts for" the error)
            if abs(contrib - total_gap) > tolerance:
                loc["in_gap_combo"] = False
    # For add, back-solve required values are from different counterfactuals; adjust so combo (actual−required) sums exactly to total_gap
    elif root_op == "add" and total_gap is not None and root_children:
        # Required choices: need sum(actual_i - required_i) = total_gap with required_i so that interpretation is consistent.
        # Set one child's required = actual (contribution 0), other(s) take the rest. Pick child with smallest |discrepancy| to get 0.
        combo_locs = [loc for loc in out if loc.get("in_gap_combo")]
        if combo_locs:
            by_sub = {loc["subexpr"]: loc for loc in combo_locs}
            actual_vals = {sub: _child_value(sub) for sub in root_children}
            # Who gets contribution 0? The one with smallest abs(discrepancy) so we "blame" the part that's most off
            combo_locs_sorted = sorted(combo_locs, key=lambda x: x["abs_discrepancy"])
            zero_sub = combo_locs_sorted[0]["subexpr"]
            # required for zero_sub = actual; for others required = claimed_target - sum(actual of zero_sub and others we set to 0)
            # So: required[zero_sub] = actual_vals[zero_sub]. Others: we need sum(actual - required) = total_gap. So sum(required) = sum(actual) - total_gap = claimed_target.
            # required[zero_sub] = actual_vals[zero_sub]. So sum(required others) = claimed_target - actual_vals[zero_sub].
            # For one other: required[other] = claimed_target - actual_vals[zero_sub]. For two others: split?
            # General: required[zero_sub]=actual_vals[zero_sub]. For each other child, required[other] = actual_vals[other] - (share of total_gap). So we assign full total_gap to the non-zero children. With two children total, one has contribution 0, one has contribution total_gap. So required[non_zero] = actual_vals[non_zero] - total_gap.
            for sub in root_children:
                loc = by_sub.get(sub)
                if not loc:
                    continue
                if sub == zero_sub:
                    req = actual_vals[sub]
                else:
                    req = actual_vals[sub] - total_gap  # so (actual - required) = total_gap for this one
                loc["required_value_to_reach_target"] = req
                loc["discrepancy"] = req - loc["actual_value"]
                loc["abs_discrepancy"] = abs(loc["discrepancy"])
    # Exclude from combo any part whose contribution (actual − required) is effectively zero
    for loc in out:
        if loc.get("in_gap_combo"):
            contrib = loc["actual_value"] - loc["required_value_to_reach_target"]
            if abs(contrib) < tolerance:
                loc["in_gap_combo"] = False
    # Prefer the first step (in evaluation order) that accounts for the total error, so we point to "first occurrence"
    if total_gap is not None:
        candidates = [
            loc for loc in out
            if abs((loc["actual_value"] - loc["required_value_to_reach_target"]) - total_gap) < tolerance
        ]
        if candidates:
            first_in_order = min(candidates, key=lambda x: x["step_index"] if x.get("step_index") is not None else 9999)
            for loc in out:
                loc["in_gap_combo"] = loc is first_in_order
    combo_sum = sum(loc["discrepancy"] for loc in out if loc.get("in_gap_combo"))
    for loc in out:
        loc["gap_combo_sum"] = combo_sum if loc.get("in_gap_combo") else None
    # First "wrong" = earliest step (eval order) with significant discrepancy (kept for ordering)
    by_step = sorted(out, key=lambda x: (x["step_index"] if x["step_index"] is not None else 9999))
    first_wrong_step_index = None
    for loc in by_step:
        if loc["abs_discrepancy"] >= tolerance:
            first_wrong_step_index = loc["step_index"]
            break
    for loc in out:
        loc["first_wrong"] = loc["step_index"] == first_wrong_step_index
    out.sort(key=lambda x: -x["abs_discrepancy"])
    return out


def _hallucination_location_explanation(
    localization: list[dict],
    claimed_target: float,
    actual_final: float,
    tolerance: float = NUMERIC_TOLERANCE,
) -> str:
    """Explain where the equation misses the target. Highlight the combo of (top-level) parts whose gaps sum exactly to the total error."""
    if not localization:
        return f"Equation evaluates to {actual_final}; claimed {claimed_target}. Could not trace sub-expressions."
    total_gap = actual_final - claimed_target
    gap_combo = [loc for loc in localization if loc.get("in_gap_combo")]
    combo_sum = sum(loc["discrepancy"] for loc in gap_combo)
    by_step = sorted(localization, key=lambda x: x["step_index"])[:8]
    lines = [
        f"Claimed target: {claimed_target}  |  Actual result (full expression): {actual_final}  |  Total gap (actual − target) = {total_gap:+.4g}",
        "(For each part below: 'value needed so total = target' = what that part would need to be for the whole expression to equal the target.)",
        "",
    ]
    for i, loc in enumerate(by_step):
        sub = loc["subexpr"]
        if len(sub) > 70:
            sub = sub[:67] + "..."
        a, r, d = loc["actual_value"], loc["required_value_to_reach_target"], loc["discrepancy"]
        step_num = loc["step_index"]
        step_head = f"Step {step_num}" if step_num != 9999 else "—"
        in_combo = loc.get("in_gap_combo")
        if in_combo:
            lines.append(f"  {step_head}  {sub}   ← this part accounts for the total error")
        else:
            lines.append(f"  {step_head}  {sub}")
        lines.append(f"        value when evaluated = {a:.4g}   |   value needed so total = target = {r:.4g}   |   discrepancy (required − actual) = {d:+.4g}")
        lines.append("")
    if gap_combo:
        # discrepancy = required - actual; so (actual - required) = -discrepancy. Sum over combo = total_gap.
        contributions = [loc["actual_value"] - loc["required_value_to_reach_target"] for loc in gap_combo]
        contrib_sum = sum(contributions)
        contrib_str = " + ".join(f"({c:+.4g})" for c in contributions)
        lines.append(f"→ Parts that account for the total error (their contributions sum to the gap above):")
        for loc in sorted(gap_combo, key=lambda x: (x["step_index"] or 0)):
            sub = (loc["subexpr"][:55] + "..") if len(loc["subexpr"]) > 57 else loc["subexpr"]
            contrib = loc["actual_value"] - loc["required_value_to_reach_target"]
            step_label = sub if loc.get("step_index") == 9999 else f"Step {loc['step_index']}: {sub}"
            lines.append(f"     {step_label}  →  (actual − required) = {contrib:+.4g}")
        lines.append(f"     Sum = {contrib_str} = {contrib_sum:+.4g}  (= total gap {total_gap:+.4g})")
    else:
        lines.append(f"→ The equation's total does not equal {claimed_target}; total gap = {total_gap:+.4g}.")
    return "\n".join(lines)


def step_by_step_breakdown_and_fopc(
    llm_equation: str,
    claimed_target: float,
    actual_value: float,
    tolerance: float = NUMERIC_TOLERANCE,
) -> tuple[list[dict], list[dict], str, list[dict], str]:
    """
    Produce step-by-step breakdown and FOPC; derive where equation diverges (back-solve, no LLM steps).
    Returns (steps_list, fopc_facts, language_explanation, hallucination_location, location_explanation).
    """
    steps, computed = _safe_math_eval_with_steps(llm_equation)
    # Use Wolfram actual_value as ground truth when our eval fails or differs
    actual = actual_value if actual_value is not None else computed
    equals_claimed = actual is not None and claimed_target is not None and abs(actual - claimed_target) <= tolerance

    # Back-solve from claimed_target to find where equation diverges (our own derivation, not ChatGPT)
    hallucination_location: list[dict] = []
    location_explanation = ""
    if not equals_claimed and actual is not None and steps:
        hallucination_location = _back_solve_required_values(steps, claimed_target, tolerance)
        location_explanation = _hallucination_location_explanation(
            hallucination_location, claimed_target, actual, tolerance
        )

    # FOPC: step(i, subexpr, value) for each step; result(actual); claimed(claimed_target);
    # equals(claimed, actual) or ¬equals(claimed, actual); incorrect(claimed); required(subexpr) for localization
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
        for loc in hallucination_location[:3]:
            fopc.append({
                "predicate": "required_to_reach_target",
                "args": [loc["subexpr"][:80]],
                "value": loc["required_value_to_reach_target"],
                "form": f"required({loc['subexpr'][:50]}...) = {loc['required_value_to_reach_target']:.4g} (actual {loc['actual_value']:.4g}, discrepancy {loc['discrepancy']:+.4g})",
            })

    # Language: step-by-step breakdown, then where it diverges (our derivation), then why incorrect
    lines = ["Step-by-step (order of operations, derived ourselves):"]
    for s in steps:
        lines.append(f"  Step {s['step_index']}: {s['subexpr']} = {s['value']}")
    if actual is not None:
        lines.append(f"  → Final value = {actual}")
    lines.append("")
    if location_explanation:
        lines.append("Where it diverges from the claimed value (back-solve from target, no ChatGPT steps):")
        lines.append(location_explanation)
        lines.append("")
    if not equals_claimed and actual is not None:
        lines.append(f"Why the claim is incorrect: claimed value = {claimed_target}, but correct value = {actual}. So ¬equals(claimed, result); hence the equation does not equal {claimed_target} (hallucination).")
    else:
        lines.append("The claimed value matches the computed result (no hallucination).")
    explanation = "\n".join(lines)
    return steps, fopc, explanation, hallucination_location, location_explanation


# -------------------------
# FOPC-style representation and deduction
# -------------------------
def facts_and_rules(
    question_id: str,
    question: str,
    ground: dict,
    llm_answer: str,
    llm_reasoning: str,
    category: str = "",
) -> list[dict]:
    """Build a minimal set of logical facts (for trace). Includes expect_symbolic_answer when category is calculus/algebra."""
    gt_text = ground.get("raw_plaintext", "") or ""
    facts = [
        {"predicate": "question", "args": [question_id], "value": question},
        {"predicate": "ground_truth", "args": [question_id], "value": gt_text},
        {"predicate": "llm_answer", "args": [question_id], "value": llm_answer},
        {"predicate": "llm_reasoning", "args": [question_id], "value": llm_reasoning[:200] + "..." if len(llm_reasoning) > 200 else llm_reasoning},
    ]
    if category:
        facts.append({"predicate": "category", "args": [question_id], "value": category})
        expect_sym, expect_proof = fopc_expect_symbolic(category)
        if expect_sym:
            facts.append({
                "predicate": "expect_symbolic_answer",
                "args": [question_id],
                "value": True,
                "form": f"expect_symbolic_answer({question_id}) ← category({question_id}, {category})",
                "fopc_proof": expect_proof,
            })
    return facts


def apply_rules_and_deduce(question_id: str, ground: dict, llm_answer: str) -> tuple[str, list[dict], str, dict]:
    """
    Rule-based reasoning + deduction + abduction + confidence + inference chain.
    Returns (verdict, reasoning_trace, language_explanation, extras).
    extras: {relative_error, confidence, abduced_causes, inference_chain}
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

    # Aggregation/statistical: relative error (oracle) + confidence (FOPC-derived)
    rel_err, _, _ = relative_error_and_confidence(gt_text, llm_answer)
    confidence, conf_expl, conf_proof = fopc_deduce_evidence_strength(rel_err, num_ok, NUMERIC_TOLERANCE)
    trace.append({
        "rule": "confidence",
        "result": confidence,
        "reason": conf_expl,
        "relative_error": rel_err,
        "fopc_engine": "real",
        "fopc_proof": conf_proof,
    })
    language_parts.append(f"Evidence strength: {conf_expl}")

    # Deduction via real FOPC: unification + resolution over Horn clauses
    verdict, fopc_proof = fopc_deduce_verdict(num_ok, sym_ok)
    trace.append({
        "inference": "verdict",
        "predicate": verdict,
        "premise": "numeric_match ∨ symbolic_match",
        "fopc_engine": "real",
        "fopc_proof": fopc_proof,
    })
    language_parts.append(f"Verdict: {verdict} (by rule: correct if numeric or symbolic match).")

    # Abduction via real FOPC: derive possible causes when hallucinating
    has_numbers = bool(extract_numbers(gt_text)) and bool(extract_numbers(llm_answer))
    abduced, _ = fopc_abduce_causes(
        hallucinating=(verdict == "hallucinating"),
        numeric_ok=num_ok,
        symbolic_ok=sym_ok,
        has_numbers_in_both=has_numbers,
    )
    trace.append({"inference": "abduction", "abduced_causes": abduced})
    if abduced:
        language_parts.append("Possible causes: " + "; ".join(c.get("form", "") for c in abduced[:2]))

    # Inference chain from FOPC proof trace (replaces fixed template)
    chain = fopc_build_inference_chain(gt_text, llm_answer, num_ok, sym_ok, verdict, fopc_proof)
    trace.append({"inference": "chain", "inference_chain": chain})

    # Meta-rules: prioritization (high_priority_hallucination, needs_human_review)
    priority_flags, priority_proof = fopc_deduce_priority(
        hallucinating=(verdict == "hallucinating"),
        evidence_strength=confidence,
        abduced_causes=abduced,
    )

    extras = {
        "relative_error": rel_err,
        "confidence": confidence,
        "abduced_causes": abduced,
        "inference_chain": chain,
        "fopc_proof": fopc_proof,
        "priority_flags": priority_flags,
        "priority_fopc_proof": priority_proof,
    }
    return verdict, trace, " ".join(language_parts), extras


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
            # Heuristic: prefer a line that looks like the requested final answer
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            answer = lines[-1] if lines else text
            q_lower = question.lower()
            # If question asks for a percent(age), prefer a line containing "%" and a number
            if "percent" in q_lower or "%" in question:
                for line in reversed(lines):
                    if "%" in line and re.search(r"\d", line) and len(line) < 150:
                        answer = line
                        break
            # Else: last short line containing "=" (equation-style answer)
            if not ("percent" in q_lower or "%" in question):
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
        verdict, error_proof = fopc_deduce_error_verdict(wolfram_failed=True)
        return {
            "question_id": "equation_claim",
            "question": f"equation equals {target}?",
            "llm_equation": llm_equation,
            "claimed_target": target,
            "actual_value": None,
            "wolfram_answer": f"[Wolfram error: {e}]",
            "verdict": verdict,
            "reasoning_explanation": str(e),
            "_reasoning_trace": [
                {"rule": "wolfram_query", "result": False, "reason": str(e)},
                {"inference": "verdict", "predicate": verdict, "fopc_engine": "real", "fopc_proof": error_proof},
            ],
            "_facts": [{"predicate": "wolfram_query_failed", "args": ["question"], "value": True}],
        }
    actual = ground.get("decimal_value")
    gt_text = ground.get("raw_plaintext", "") or ""
    if actual is None:
        actual = extract_numbers(gt_text)[0] if extract_numbers(gt_text) else None
    equals_target = actual is not None and abs(actual - target) <= tolerance
    verdict, fopc_eq_proof = fopc_deduce_equation_verdict(equals_target)
    reason = (
        f"equation_value({actual}) equals claimed_target({target})"
        if equals_target
        else f"equation_value({actual}) != claimed_target({target})"
    )
    trace = [
        {"rule": "equals_target", "result": equals_target, "reason": reason},
        {
            "inference": "verdict",
            "predicate": verdict,
            "premise": "equals_target ↔ (|actual - target| ≤ tolerance)",
            "fopc_engine": "real",
            "fopc_proof": fopc_eq_proof,
        },
    ]
    facts = [
        {"predicate": "claimed_target", "args": [], "value": target},
        {"predicate": "llm_equation", "args": [], "value": llm_equation},
        {"predicate": "ground_truth_expression", "args": [], "value": gt_text},
        {"predicate": "actual_value", "args": [], "value": actual},
    ]
    # Step-by-step breakdown and FOPC; derive where it diverges (back-solve, no ChatGPT steps)
    steps_list, fopc_steps, step_explanation, hallucination_location, location_explanation = step_by_step_breakdown_and_fopc(
        llm_equation, target, actual, tolerance
    )
    facts.extend([{"predicate": f["predicate"], "args": f["args"], "value": f["value"], "form": f.get("form", "")} for f in fopc_steps])

    # FOPC: diverges_at(Subexpr) ← step_with_discrepancy ∧ first_in_eval_order
    diverges_subexprs, diverges_proof = fopc_deduce_diverges_at(hallucination_location)
    for subexpr in diverges_subexprs:
        facts.append({"predicate": "diverges_at", "args": [subexpr], "value": True, "form": f"diverges_at('{subexpr}')"})

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
        "hallucination_location": hallucination_location,
        "where_diverges_explanation": location_explanation,
        "diverges_at": diverges_subexprs,
        "_diverges_fopc_proof": diverges_proof,
        "_reasoning_trace": trace,
        "_facts": facts,
        "_fopc_proof": fopc_eq_proof,
    }


# -------------------------
# Wolfram-friendly rephrasings for word problems (when natural language returns no result)
# -------------------------
def _wolfram_friendly_query(word_problem: str) -> str | None:
    """If the problem matches a known type, return a query Wolfram is more likely to answer."""
    q = word_problem.lower().strip()
    # Bat and ball: x + (x+1) = 1.10 → 2x + 1 = 1.10 → x = 0.05
    if "bat" in q and "ball" in q and ("1.10" in q or "1.10" in word_problem):
        return "solve 2x+1=1.10 for x"
    if "ball" in q and "1.10" in word_problem and "1 " in word_problem and "more" in q:
        return "solve 2x+1=1.10 for x"
    # Average speed: total dist / total time = 240/5 = 48
    if "average speed" in q and "60" in word_problem and "40" in word_problem:
        return "(60*2 + 40*3) / 5"
    if "60 mph" in q and "2 hours" in q and "40 mph" in q and "3 hours" in q:
        return "average speed 240 miles in 5 hours"
    # Double discount: 20% then 20% → 1 - 0.8*0.8 = 0.36 (36%)
    if "20%" in word_problem and "20%" in word_problem and ("equivalent" in q or "single" in q) and "discount" in q:
        return "100 * (1 - 0.8*0.8)"
    # Stock drops 50%: what % rise to get back? (1 - 0.5)/0.5 = 1 = 100% (not 50%)
    if "50%" in word_problem and ("drop" in q or "drops" in q) and ("rise" in q or "gain" in q) and ("back" in q or "original" in q):
        return "(1 - 0.5) / 0.5"
    # Lily pad: doubles every day, 48 days to cover pond → half covered the day before = 47 (not 24)
    if ("lily" in q or "doubles" in q or "double" in q) and "48" in word_problem and ("half" in q or "halfway" in q):
        return "48 - 1"
    # Snail: 10 ft wall, up 3 per day down 2 per night → 8 days (many say 10)
    if "snail" in q and "10" in word_problem and "3" in word_problem and "2" in word_problem:
        return "(10 - 3) / (3 - 2) + 1"
    return None


# -------------------------
# Word problem validation (Wolfram = ground truth, ChatGPT = LLM answer)
# -------------------------
def validate_word_problem(word_problem: str, expected_answer: str | None = None) -> dict[str, Any]:
    """
    Send the same word problem to Wolfram (ground truth) and to the LLM; compare answers.
    If expected_answer is provided, it is used as the ground truth (no Wolfram call); otherwise Wolfram is the ground truth.
    Returns row with verdict, reasoning trace, FOPC-style facts, and explicit ground_truth / ground_truth_value.
    """
    if expected_answer is not None:
        ground = {
            "raw_plaintext": expected_answer.strip(),
            "decimal_value": None,
            "pods": [],
        }
        nums = extract_numbers(expected_answer)
        if nums:
            ground["decimal_value"] = nums[0]
    else:
        try:
            # For word problems, first try Wolfram Alpha's Short Answers API
            short = wolfram_short_answer(word_problem)
            if short:
                ground = {
                    "raw_plaintext": short,
                    "decimal_value": None,
                    "pods": [],
                }
                nums = extract_numbers(short)
                if nums:
                    ground["decimal_value"] = nums[-1]
            else:
                ground = get_ground_truth(word_problem)
                # If still no plaintext, try a Wolfram-friendly rephrase for known problem types
                if not (ground.get("raw_plaintext") or "").strip():
                    alt = _wolfram_friendly_query(word_problem)
                    if alt:
                        ground_alt = get_ground_truth(alt)
                        if (ground_alt.get("raw_plaintext") or "").strip():
                            ground = ground_alt
        except Exception as e:
            verdict, error_proof = fopc_deduce_error_verdict(wolfram_failed=True)
            return {
                "question_id": "word_problem",
                "question": word_problem,
                "ground_truth": None,
                "ground_truth_value": None,
                "wolfram_answer": f"[Wolfram error: {e}]",
                "llm_answer": "",
                "numeric_match": False,
                "symbolic_match": False,
                "verdict": verdict,
                "reasoning_explanation": str(e),
                "_reasoning_trace": [
                    {"rule": "wolfram_short_answer", "result": False, "reason": str(e)},
                    {"inference": "verdict", "predicate": verdict, "fopc_engine": "real", "fopc_proof": error_proof},
                ],
                "_facts": [{"predicate": "wolfram_query_failed", "args": ["question"], "value": True}],
            }
    llm_out = call_llm(word_problem)
    verdict, trace, explanation, extras = apply_rules_and_deduce("word_problem", ground, llm_out["answer"])
    gt_text = ground.get("raw_plaintext", "") or ""
    ground_value = ground.get("decimal_value")  # numeric when Wolfram gives a clear number
    facts = [
        {"predicate": "question", "args": ["word_problem"], "value": word_problem},
        {"predicate": "ground_truth", "args": [], "value": gt_text},
        {"predicate": "ground_truth_value", "args": [], "value": ground_value},
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
        "ground_truth": gt_text[:500],
        "ground_truth_value": ground_value,
        "wolfram_answer": gt_text[:300],
        "llm_answer": (llm_out["answer"] or "")[:300],
        "llm_reasoning": (llm_out["reasoning"] or "")[:500],
        "numeric_match": trace[0]["result"] if trace else False,
        "symbolic_match": trace[1]["result"] if len(trace) > 1 else False,
        "verdict": verdict,
        "reasoning_explanation": explanation,
        "_reasoning_trace": trace,
        "_facts": facts,
        "_extras": extras,
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
        verdict, trace, explanation, extras = apply_rules_and_deduce(qid, ground, llm_out["answer"])

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
        row["_facts"] = facts_and_rules(qid, q, ground, llm_out["answer"], llm_out["reasoning"], item.get("category", ""))
        row["_extras"] = extras
        rows.append(row)

    if use_cache:
        # Save cache without non-JSON-serializable content; keep only raw_plaintext for size
        save_cache = {q: {"raw_plaintext": g.get("raw_plaintext"), "definite_result": g.get("definite_result")} for q, g in gt_cache.items()}
        with open(cache_file, "w") as f:
            json.dump(save_cache, f, indent=2)
    return rows


def write_table_and_log(rows: list[dict]) -> None:
    """Write CSV table and reasoning log (FOPC-style + trace + aggregation + abduction + inference chain)."""
    import csv
    table_path = os.path.join(RESULTS_DIR, RESULTS_TABLE)
    log_path = os.path.join(RESULTS_DIR, REASONING_LOG)

    # Aggregation via FOPC: aggregate_result(N, K, R) ← all_processed ∧ count_hallucinating ∧ rate
    n = len(rows)
    k = sum(1 for r in rows if r.get("verdict") == "hallucinating")
    aggregation, agg_proof = fopc_aggregate_result(n, k)
    aggregation["fopc_proof"] = agg_proof
    aggregation["fopc"] = aggregation.get("fopc_aggregation", [])  # backward compatibility

    # CSV: drop internal keys; add confidence and relative_error if present
    table_rows = []
    for r in rows:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        ex = r.get("_extras", {})
        if ex:
            row["confidence"] = ex.get("confidence")
            row["relative_error"] = ex.get("relative_error")
        table_rows.append(row)
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        all_keys = list(table_rows[0].keys()) if table_rows else []
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(table_rows)
    print(f"Table: {table_path}")

    log = {
        "representation": "Real FOPC (unification + resolution) + FOPC-style facts + abduction + aggregation + inference chain",
        "aggregation": aggregation,
        "questions": [
            {
                "id": r["question_id"],
                "facts": r.get("_facts", []),
                "reasoning_trace": r.get("_reasoning_trace", []),
                "verdict": r["verdict"],
                "abduced_causes": r.get("_extras", {}).get("abduced_causes", []),
                "inference_chain": r.get("_extras", {}).get("inference_chain", []),
                "fopc_proof": r.get("_extras", {}).get("fopc_proof", []),
                "confidence": r.get("_extras", {}).get("confidence"),
                "relative_error": r.get("_extras", {}).get("relative_error"),
                "priority_flags": r.get("_extras", {}).get("priority_flags", []),
            }
            for r in rows
        ],
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Reasoning log: {log_path}")


def main():
    # Ensure output is visible (e.g. when run from some IDEs or scripts)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    argv = sys.argv[1:]
    print("Math LLM Hallucination Validator — starting ...", flush=True)
    # Word problem: send to Wolfram + ChatGPT, compare answers
    if argv and argv[0] == "--word-problem":
        if len(argv) < 2:
            print("Usage: python math_hallucination_validator.py --word-problem \"YOUR WORD PROBLEM\" [--expected \"ANSWER\"]")
            print('Example: python math_hallucination_validator.py --word-problem "A train leaves at 2pm at 60 mph..."')
            print('Example with known answer: python math_hallucination_validator.py --word-problem "What is 2+2?" --expected "4"')
            return 1
        try:
            ex_idx = argv.index("--expected")
        except ValueError:
            ex_idx = -1
        if ex_idx >= 0:
            word_problem = " ".join(argv[1:ex_idx]).strip()
            expected_answer = " ".join(argv[ex_idx + 1:]).strip() if ex_idx + 1 < len(argv) else ""
        else:
            word_problem = " ".join(argv[1:]).strip()
            expected_answer = None
        if not word_problem:
            print("Provide a non-empty word problem in quotes.")
            return 1
        if expected_answer is not None and not expected_answer:
            expected_answer = None
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is required for --word-problem (to get ChatGPT's answer). Set it in .env or export it.")
            return 1
        print("Math LLM Hallucination Validator — word problem")
        print(f"Problem: {word_problem[:200]}{'...' if len(word_problem) > 200 else ''}\n")
        if expected_answer is not None:
            print("Using provided ground truth (--expected).")
        else:
            print("Getting ground truth from Wolfram...")
        print("Getting answer from ChatGPT...\n")
        row = validate_word_problem(word_problem, expected_answer=expected_answer)
        gt_label = "Ground truth (provided)" if expected_answer is not None else "Ground truth (Wolfram Alpha)"
        gt_text = (row.get("ground_truth") or row.get("wolfram_answer") or "")[:400]
        print(f"--- {gt_label} ---")
        print(f"  Answer: {gt_text}")
        if row.get("ground_truth_value") is not None:
            print(f"  Numeric value: {row['ground_truth_value']}")
        print("\n--- LLM answer (ChatGPT) ---")
        print(f"  Answer: {row.get('llm_answer', '')[:300]}")
        if row.get("llm_reasoning"):
            print(f"  Reasoning (excerpt): {row['llm_reasoning'][:200]}...")
        print(f"\nVerdict: {row['verdict']}  (numeric_match={row.get('numeric_match')}, symbolic_match={row.get('symbolic_match')})")
        extras = row.get("_extras", {})
        if extras.get("confidence"):
            print(f"Evidence strength: {extras['confidence']}")
        if extras.get("abduced_causes"):
            print("Abduced causes:", [c.get("hypothesis") for c in extras["abduced_causes"]])
        print(f"Reasoning: {row['reasoning_explanation']}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "word_problem_log.json")
        with open(log_path, "w") as f:
            json.dump(
                {
                    "question": row["question"],
                    "ground_truth": row.get("ground_truth", row.get("wolfram_answer")),
                    "ground_truth_value": row.get("ground_truth_value"),
                    "wolfram_answer": row["wolfram_answer"],
                    "llm_answer": row["llm_answer"],
                    "llm_reasoning": row.get("llm_reasoning"),
                    "verdict": row["verdict"],
                    "numeric_match": row.get("numeric_match"),
                    "symbolic_match": row.get("symbolic_match"),
                    "reasoning_explanation": row["reasoning_explanation"],
                    "facts": row.get("_facts", []),
                    "reasoning_trace": row.get("_reasoning_trace", []),
                    "abduced_causes": extras.get("abduced_causes", []),
                    "inference_chain": extras.get("inference_chain", []),
                    "fopc_proof": extras.get("fopc_proof", []),
                    "confidence": extras.get("confidence"),
                    "relative_error": extras.get("relative_error"),
                    "priority_flags": extras.get("priority_flags", []),
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
        print("Validating with Wolfram (step-by-step + where it goes wrong)...\n")
        row = validate_equation_claim(target, llm_equation)
        print("Result from Wolfram Alpha:")
        print(f"  Exact: {row.get('wolfram_answer', '')[:200]}...")
        print(f"  Decimal: {row.get('actual_value')}")
        print(f"Verdict: {row['verdict']}\n")
        print("--- Step-by-step (how we evaluated the expression) ---")
        for s in row.get("step_by_step", []):
            print(f"  Step {s['step_index']}: {s['subexpr']} = {s['value']}")
        if row.get("where_diverges_explanation"):
            print("\n--- Where the equation misses the target ---")
            print(row["where_diverges_explanation"])
            locs = row.get("hallucination_location", [])[:5]
            if locs:
                locs_by_step = sorted(locs, key=lambda x: x["step_index"])
                print("\n  Summary (evaluation order):")
                print("  " + "-" * 72)
                for loc in locs_by_step[:8]:
                    sub = (loc["subexpr"][:52] + "..") if len(loc["subexpr"]) > 54 else loc["subexpr"]
                    wrong_marker = "   ← this part accounts for the total error" if loc.get("in_gap_combo") else ""
                    step_lbl = f"Step {loc['step_index']}." if loc.get("step_index") != 9999 else "—"
                    print(f"  {step_lbl} {sub}{wrong_marker}")
                    print(f"     value when evaluated: {loc['actual_value']:>8.4g}   needed so total = target: {loc['required_value_to_reach_target']:>8.4g}   discrepancy (required − actual): {loc['discrepancy']:>+8.4g}")
                print("  " + "-" * 72)
        print("\n--- Logic-style facts (FOPC) ---")
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
                    "hallucination_location": row.get("hallucination_location", []),
                    "where_diverges_explanation": row.get("where_diverges_explanation", ""),
                    "diverges_at": row.get("diverges_at", []),
                    "fopc": row.get("fopc_step_and_incorrect", []),
                    "facts": row.get("_facts", []),
                    "reasoning_trace": row.get("_reasoning_trace", []),
                    "fopc_proof": row.get("_fopc_proof", []),
                    "diverges_fopc_proof": row.get("_diverges_fopc_proof", []),
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
        print(f"Expression: {llm_equation}\n")
        row = validate_equation_claim(target, llm_equation)
        print("Result from Wolfram Alpha (ground truth):")
        print(f"  Exact: {row.get('wolfram_answer', '')[:200]}...")
        print(f"  Decimal: {row.get('actual_value')}")
        print(f"Verdict: {row['verdict']}  (equation does {'not ' if row['verdict'] == 'hallucinating' else ''}equal the claimed value)\n")
        print("--- Step-by-step (how we evaluated the expression) ---")
        print("  Each line is one sub-expression and the value we got for it.\n")
        for s in row.get("step_by_step", []):
            print(f"  Step {s['step_index']}: {s['subexpr']} = {s['value']}")
        if row.get("where_diverges_explanation"):
            print("\n--- Where the equation misses the target ---")
            print("  For each part: what it actually is vs what it would need to be so the whole expression equals the target.\n")
            print(row["where_diverges_explanation"])
            locs = row.get("hallucination_location", [])[:5]
            if locs:
                locs_by_step = sorted(locs, key=lambda x: x["step_index"])
                print("\n  Summary (evaluation order):")
                print("  " + "-" * 72)
                for loc in locs_by_step[:8]:
                    sub = (loc["subexpr"][:52] + "..") if len(loc["subexpr"]) > 54 else loc["subexpr"]
                    wrong_marker = "   ← this part accounts for the total error" if loc.get("in_gap_combo") else ""
                    step_lbl = f"Step {loc['step_index']}." if loc.get("step_index") != 9999 else "—"
                    print(f"  {step_lbl} {sub}{wrong_marker}")
                    print(f"     value when evaluated: {loc['actual_value']:>8.4g}   needed so total = target: {loc['required_value_to_reach_target']:>8.4g}   discrepancy (required − actual): {loc['discrepancy']:>+8.4g}")
                print("  " + "-" * 72)
        print("\n--- Logic-style facts (FOPC) ---")
        print("  Formal statements used for reasoning (step values, result, claimed, equals/not).\n")
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
                    "hallucination_location": row.get("hallucination_location", []),
                    "where_diverges_explanation": row.get("where_diverges_explanation", ""),
                    "diverges_at": row.get("diverges_at", []),
                    "fopc": row.get("fopc_step_and_incorrect", []),
                    "facts": row.get("_facts", []),
                    "reasoning_trace": row.get("_reasoning_trace", []),
                    "fopc_proof": row.get("_fopc_proof", []),
                    "diverges_fopc_proof": row.get("_diverges_fopc_proof", []),
                },
                f,
                indent=2,
            )
        print(f"\nReasoning log: {log_path}")
        return 0

    print("Math LLM Hallucination Validator — check if a math equation equals a claimed value (using Wolfram as ground truth).")
    print("")
    print("Usage:")
    print('  Check one equation:  --equation-claim TARGET "EXPRESSION"')
    print("  Example:            --equation-claim 88 \"sqrt(64) + ln(e^5)\"")
    print("")
    print("  Ask ChatGPT for an equation, then check:  --ask-equation TARGET   (needs OPENAI_API_KEY)")
    print("  Compare Wolfram vs ChatGPT on a word problem:  --word-problem \"PROBLEM\"   (needs OPENAI_API_KEY)")
    print("  Run question bank:  (no args)")
    print("")
    if os.environ.get("OPENAI_API_KEY"):
        print("LLM: ChatGPT (OpenAI API)")
    else:
        print("LLM: mock (set OPENAI_API_KEY or add to .env for real ChatGPT)")
    print("")
    rows = run_validation(use_cache=True)
    write_table_and_log(rows)
    n = len(rows)
    k = sum(1 for r in rows if r.get("verdict") == "hallucinating")
    print("\n--- Summary ---")
    for r in rows:
        ex = r.get("_extras", {})
        conf = ex.get("confidence", "")
        print(f"  {r['question_id']}: {r['verdict']}  (numeric={r['numeric_match']}, symbolic={r['symbolic_match']}, confidence={conf})")
    print(f"\n--- Aggregation ---  total={n}, hallucinating={k}, rate={k/n:.0%}" if n else "")
    print("\nReasoning: Real FOPC (unification+resolution) + abduction + inference chain + aggregation. Log:", REASONING_LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
