# Math LLM Hallucination Validator

**Check whether a math equation or answer is correct** by comparing it to Wolfram Alpha. Useful when an LLM (e.g. ChatGPT) claims an equation equals a certain value—this tool tells you if that’s true and, if not, **where** the equation goes wrong.

---

## What it does

- **Equation-claim mode**: You give a target number (e.g. 88) and an equation (e.g. `sqrt(64) + ln(e^5)`). The tool:
  1. Asks Wolfram what the equation actually equals.
  2. Breaks the expression into steps (order of operations).
  3. If the result ≠ target, it shows **which part(s) of the expression** account for the error and by how much.

- **Other modes**: Compare an LLM’s answer to Wolfram on word problems, or run a small question bank and get a verdict per question.

---

## Quick start

**Requirements**

- Python 3
- `requests` (for Wolfram): `pip install requests`
- A **Wolfram Alpha App ID** (free at [developer.wolfram.com](https://developer.wolfram.com)) — set it in `wolfram_alpha.py` as `APP_ID`.
- Optional: `OPENAI_API_KEY` in the environment or in a `.env` file if you use `--word-problem` or `--ask-equation`.

**Check one equation**

```bash
python math_hallucination_validator.py --equation-claim 88 "sqrt(6400) + 2*(sin(pi/2)*24) + ln(e^5)"
```

You’ll see:

- **Verdict**: Does the equation actually equal 88? (e.g. “hallucinating” if not.)
- **Step-by-step**: How we evaluated the expression (each sub-expression and its value).
- **Where it diverges**: Which sub-parts would need to change so the whole expression would hit the target, and by how much they’re off.
- **Which part explains the error**: One or more parts whose “contribution” to the error adds up exactly to the total gap.

---

## Understanding the output (equation-claim)

| Section | Meaning |
|--------|--------|
| **Claim** | “The following equation equals X.” |
| **Wolfram (exact / decimal)** | What Wolfram says the expression equals (ground truth). |
| **Verdict** | `hallucinating` = equation does *not* equal the claimed value; otherwise it matches. |
| **Step-by-step** | Our own evaluation: each step shows a sub-expression and its numeric value (e.g. `(5*13) = 65`). |
| **Where it diverges** | For each step we show: **value when evaluated** (what we got), **value needed so total = target** (what that part would need to be for the full expression to hit the target), and **discrepancy** (required − actual). |
| **PART OF COMBO** | Those sub-parts whose “contribution” (actual − required) **sums exactly to the total error**. So “the equation goes wrong” in those parts. |
| **Logic-style facts (FOPC)** | First-order logic style facts used internally (e.g. `result(actual) = 75`, `¬equals(claimed, result)`). You can ignore this unless you care about the formal reasoning. |

Supported in expressions: `+`, `-`, `*`, `/`, `**` (or `^`), `sqrt`, `ln` / `log` (natural log), `cos`, `sin`, `tan`, `e`, `pi` (or π).

---

## What question types does this work for?

| Mode | Question type | What you get |
|------|----------------|--------------|
| **Equation-claim** | **Numeric expressions** (e.g. `sqrt(64) + ln(e^5)`, `2*sin(pi/2)`) | Full pipeline: Wolfram result, **step-by-step**, **where it diverges**, and which part accounts for the error. Only expressions we can parse (the operators above) get the step-by-step breakdown. |
| **Equation-claim** | Other expressions (e.g. `integrate x^2 dx`) | Wolfram result and verdict (equals target or not). No step-by-step or “where it diverges,” because we don’t parse those. |
| **Ask-equation** | N/A (you only give a target number) | ChatGPT suggests an equation; we then run equation-claim on it. So same as equation-claim: step-by-step only if the suggested expression is in our supported set. |
| **Word-problem** | **Any** (word problems, applied math, etc.) | Wolfram’s answer is the **ground truth** (or use `--expected "answer"` to supply your own). We compare the LLM’s answer to that with **numeric match** and **symbolic match** and give a verdict. No step-by-step; good for “did the LLM get the same answer as the ground truth?” |
| **Question bank** (no args) | **Any** (built-in set: arithmetic/trig, algebra, calculus) | Same as word-problem: each question is sent to Wolfram and the LLM; we compare answers and report verdict. |

**Summary**

- For **“does this equation equal X?”** and **“where does it go wrong?”** → use **equation-claim** with a **single numeric expression** (the one we can parse).
- For **“did the LLM get the right answer?”** on **any** question (word problems, integrals, equations, etc.) → use **word-problem** or the **question bank**; you get match/mismatch and reasoning, but no expression-level step-by-step.

---

## All modes

| Command | What it does |
|---------|----------------|
| `python math_hallucination_validator.py --equation-claim TARGET "EXPRESSION"` | Check if the expression equals TARGET (no LLM call). |
| `python math_hallucination_validator.py --ask-equation TARGET` | Ask ChatGPT for an equation that equals TARGET, then check it (needs `OPENAI_API_KEY`). |
| `python math_hallucination_validator.py --word-problem "PROBLEM"` | Send the problem to Wolfram and ChatGPT; compare answers. **Ground truth** = Wolfram’s answer (shown explicitly). Needs `OPENAI_API_KEY`. |
| `python math_hallucination_validator.py --word-problem "PROBLEM" --expected "ANSWER"` | Same, but use your provided answer as **ground truth** (no Wolfram call). Useful for labeled datasets. |
| `python math_hallucination_validator.py` | Run the built-in question bank; compare each to Wolfram (and optionally ChatGPT). |

Results are written under `wolfram_data/` (e.g. `equation_claim_log.json` for equation-claim).

---

## More detail

- **README_VALIDATOR.md** — Modes, reasoning (rule-based, FOPC, abduction, aggregation), and assignment-style outline.
- **wolfram_alpha.py** — Wolfram API and App ID configuration.
