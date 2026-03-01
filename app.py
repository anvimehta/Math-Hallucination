"""
Math LLM Hallucination Validator — Web API & Chat Frontend
Run: python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory

# Import after path is set
from math_hallucination_validator import (
    validate_word_problem,
    validate_equation_claim,
    ask_llm_for_equation,
)
from local_llm_math_checker import call_llm as local_llm_call, check_llm_answer, check_constraint

app = Flask(__name__, static_folder="frontend", static_url_path="")


def _json_safe_float(x):
    """Return a float that is JSON-serializable (replace nan/inf with None)."""
    if x is None:
        return None
    try:
        f = float(x)
        if f != f or f == float("inf") or f == float("-inf"):  # nan or inf
            return None
        return f
    except (TypeError, ValueError):
        return None


# Allow frontend to call API when served from same origin or file
@app.after_request
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _format_word_problem_response(row):
    """Format word-problem result for chat display. reasoning_trace is rule-based (no LLM)."""
    extras = row.get("_extras", {})
    trace = row.get("_reasoning_trace") or []
    # Build a UI-friendly reasoning trace: rule/inference -> result, reason (all rule-based, no LLM)
    reasoning_trace = []
    for t in trace:
        step = t.get("rule") or t.get("inference") or ""
        result = t.get("result")
        reason = t.get("reason") or t.get("premise") or ""
        if t.get("abduced_causes"):
            parts = []
            for c in t["abduced_causes"][:2]:
                v = c.get("form") or c.get("hypothesis") or ""
                parts.append(v if isinstance(v, str) else str(v))
            reason = "; ".join(parts)
        if t.get("inference_chain"):
            chain = t["inference_chain"][:6]
            reason = " → ".join(
                item.get("form", str(item)) if isinstance(item, dict) else str(item)
                for item in chain
            )
        reasoning_trace.append({"step": step, "result": result, "reason": reason})
    return {
        "verdict": row.get("verdict"),
        "ground_truth": (row.get("ground_truth") or row.get("wolfram_answer") or "")[:500],
        "ground_truth_value": row.get("ground_truth_value"),
        "answer_value": row.get("answer_value"),
        "llm_answer": (row.get("llm_answer") or "")[:500],
        "llm_reasoning": (row.get("llm_reasoning") or "")[:400],
        "answer_was_provided": row.get("answer_was_provided", False),
        "numeric_match": row.get("numeric_match"),
        "symbolic_match": row.get("symbolic_match"),
        "reasoning_explanation": row.get("reasoning_explanation"),
        "reasoning_narrative": row.get("reasoning_narrative", ""),
        "reasoning_trace": reasoning_trace,
        "confidence": extras.get("confidence"),
        "abduced_causes": [c.get("hypothesis") for c in extras.get("abduced_causes", [])],
        "relative_error": extras.get("relative_error"),
    }


def _format_equation_claim_response(row):
    """Format equation-claim result for chat display. Same step_by_step/hallucination_location for both Check equation and Ask equation."""
    return {
        "verdict": row.get("verdict"),
        "llm_equation": row.get("llm_equation"),  # expression (user's or ChatGPT's)
        "wolfram_answer": (row.get("wolfram_answer") or "")[:300],
        "actual_value": row.get("actual_value"),
        "claimed_target": row.get("claimed_target"),
        "step_by_step": row.get("step_by_step", []),
        "where_diverges_explanation": row.get("where_diverges_explanation"),
        "hallucination_location": row.get("hallucination_location", [])[:8],
        "fopc_step_and_incorrect": row.get("fopc_step_and_incorrect", []),
        "reasoning_explanation": row.get("reasoning_explanation"),
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    """Single endpoint: run a validator mode. Body: { mode, ...params }."""
    try:
        body = request.get_json() or {}
        mode = (body.get("mode") or "").strip().lower()
        if not mode:
            return jsonify({"ok": False, "error": "Missing 'mode'. Use: word_problem, equation."}), 400

        # --- Word problem: two optional fields — "Correct answer" and "Your answer" ---
        if mode == "word_problem":
            word_problem = (body.get("word_problem") or "").strip()
            if not word_problem:
                return jsonify({"ok": False, "error": "Please provide a word problem or math question."}), 400
            correct_raw = body.get("correct_answer") or body.get("expected_answer")
            correct_raw = (str(correct_raw).strip() or None) if correct_raw is not None else None
            your_answer = body.get("your_answer") or body.get("answer_to_check")
            your_answer = (str(your_answer).strip() or None) if your_answer is not None else None
            try:
                target_num = float(correct_raw) if correct_raw else None
            except (TypeError, ValueError):
                target_num = None
            use_llm_cleanup = bool(body.get("use_llm_cleanup"))

            # Local LLM check: Correct answer is a number and Your answer is blank → ask LLM, extract expression, compare (no Wolfram)
            if target_num is not None and not your_answer:
                if not os.environ.get("OPENAI_API_KEY"):
                    return jsonify({"ok": False, "error": "OPENAI_API_KEY is required for local check. Add it to .env or export it."}), 400
                try:
                    llm_out = local_llm_call(word_problem)
                    combined = f"{llm_out.get('answer', '')}\n{llm_out.get('reasoning', '')}"
                    result = check_llm_answer(combined, target=target_num)
                    constraint_ok, constraint_reason = check_constraint(word_problem, result)
                    if not constraint_ok:
                        verdict = "hallucinating"
                        verdict_note = "Constraint violated, even though numeric value matches."
                    else:
                        verdict = "not_hallucinating" if result.is_correct else "hallucinating"
                        verdict_note = None
                    return jsonify({
                        "ok": True,
                        "mode": "word_problem",
                        "result": {
                            "question": word_problem,
                            "verdict": verdict,
                            "verdict_note": verdict_note,
                            "target": _json_safe_float(result.target),
                            "expression": result.expression,
                            "value": _json_safe_float(result.value),
                            "difference": _json_safe_float(result.difference),
                            "error": result.error,
                            "constraint_ok": constraint_ok,
                            "constraint_reason": constraint_reason or None,
                            "llm_answer": llm_out.get("answer", "")[:400],
                            "llm_reasoning": llm_out.get("reasoning", "")[:500],
                            "is_local_check": True,
                        },
                    })
                except Exception as e:
                    return jsonify({"ok": False, "error": f"Local check failed: {e!s}"}), 200

            # Word problem vs Wolfram (or vs Correct answer as ground truth)
            if your_answer is None and not os.environ.get("OPENAI_API_KEY"):
                return jsonify({"ok": False, "error": "OPENAI_API_KEY is required when you don't provide 'Your answer'. Add it to .env or enter an answer to check."}), 400
            if use_llm_cleanup and not os.environ.get("OPENAI_API_KEY"):
                use_llm_cleanup = False
            row = validate_word_problem(word_problem, expected_answer=correct_raw, answer_to_check=your_answer, use_llm_cleanup=use_llm_cleanup)
            out = _format_word_problem_response(row)
            out["is_local_check"] = False
            return jsonify({"ok": True, "mode": "word_problem", "result": out})

        # --- Equation: check expression or ask ChatGPT for one (same validation) ---
        if mode == "equation":
            try:
                target = float(body.get("target"))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "Please provide a numeric target (e.g. 67)."}), 400
            expression = (body.get("expression") or "").strip()

            def run_equation_validation(t: float, expr: str):
                row = validate_equation_claim(t, expr)
                return _format_equation_claim_response(row)

            if expression:
                result = run_equation_validation(target, expression)
                result["asked_equation"] = False
                return jsonify({"ok": True, "mode": "equation", "result": result})
            if not os.environ.get("OPENAI_API_KEY"):
                return jsonify({"ok": False, "error": "OPENAI_API_KEY is required to ask for an equation. Add it to .env or provide an expression to check."}), 400
            out = ask_llm_for_equation(target)
            if out.get("error"):
                return jsonify({"ok": False, "error": out["error"], "raw": out.get("raw_response", "")[:500]}), 400
            result = run_equation_validation(target, out["equation"])
            result["asked_equation"] = True
            return jsonify({"ok": True, "mode": "equation", "result": result})

        return jsonify({"ok": False, "error": f"Unknown mode: {mode}. Use: word_problem, equation."}), 400

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("Math LLM Hallucination Validator — Web UI")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=True)
