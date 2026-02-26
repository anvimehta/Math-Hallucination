# Math LLM Hallucination Validator

## How to use

**Prerequisites**
- Python 3 with `requests` (for Wolfram). Optional: `openai` for ChatGPT (`pip install openai`).
- Wolfram Alpha App ID set in `wolfram_alpha.py` (variable `APP_ID`).
- For real LLM output: set `OPENAI_API_KEY` in the environment or in a `.env` file (copy `.env.example` to `.env`).

**Four modes:**

1. **Word problem**  
   Send one word problem to Wolfram (ground truth) and to ChatGPT; compare answers and get a verdict. Requires `OPENAI_API_KEY`.
   ```bash
   python math_hallucination_validator.py --word-problem "A train leaves at 2pm at 60 mph. Another leaves at 3pm from the same station at 80 mph. When does the second catch the first?"
   ```
   Output: Wolfram’s answer, ChatGPT’s answer and reasoning, verdict (hallucinating / not), and `wolfram_data/word_problem_log.json`.

2. **Full validation (question bank)**  
   Runs the built-in math questions, gets ground truth from Wolfram, gets answers from ChatGPT (or mock if no API key), and compares.
   ```bash
   cd /path/to/wolfram
   python math_hallucination_validator.py
   ```
   Output: `wolfram_data/validation_results.csv`, `wolfram_data/reasoning_log.json`, and a summary in the terminal.

3. **Ask ChatGPT for an equation, then validate it**  
   Ask ChatGPT for a quite hard equation that equals a target (e.g. 67); then check with Wolfram whether it really does (step-by-step + FOPC). Requires `OPENAI_API_KEY`.
   ```bash
   python math_hallucination_validator.py --ask-equation 67
   ```
   Output: the equation ChatGPT gave, Wolfram’s evaluation, verdict (hallucinating / not), step-by-step, FOPC, and `wolfram_data/equation_claim_log.json`.

4. **Equation-claim validation**  
   You already have a target value and an equation an LLM claimed equals that value. Check if the equation really equals the target (step-by-step + FOPC).
   ```bash
   python math_hallucination_validator.py --equation-claim 67 "sqrt((34*52 + 73) - 144/4) + ln(e^5) + cos(0)^2"
   ```
   Output: verdict (hallucinating / not), step-by-step breakdown, FOPC facts, and `wolfram_data/equation_claim_log.json`.

---

## Reasoning included

- **Rule-based**: numeric match (tolerance), symbolic match (normalized form).
- **Logical inference (deduction)**: verdict = not_hallucinating ↔ (numeric_match ∨ symbolic_match).
- **Abduction**: when hallucinating, possible causes (numeric_error, symbolic_error, both) as FOPC-style hypotheses.
- **Aggregation / statistical**: relative error (ground truth vs LLM), confidence (evidence strength); over the question bank: total_questions, hallucination_count, rate_hallucinating(k, n).
- **Explicit inference chain (FOPC)**: premises P1–P4 (ground_truth, llm_answer, numeric_match, symbolic_match), rule R, conclusion C (verdict).
- **Step-by-step (equation-claim)**: order-of-operations breakdown and FOPC step(·, ·, ·), result(·), incorrect(·).

## Steps (assignment outline)

1. **Question bank** — Define early-college math questions (algebra, calculus, trig, etc.) in code or a single data file.
2. **Ground truth** — For each question, call Wolfram Alpha API (reuse `wolfram_alpha.py`), parse primary pod for the canonical answer.
3. **LLM interface** — Send each question to an LLM; capture both final answer and reasoning trace (language).
4. **Representation** — Store and reason over:
   - **Tabular**: rows = question, Wolfram answer, LLM answer, match flags, verdict.
   - **FOPC / logic**: predicates such as `ground_truth(q,a)`, `llm_answer(q,b)`, `numeric_match(a,b)`; deduction to `hallucinating(q)` or `¬hallucinating(q)`.
   - **Optional minimal graph**: nodes = question, answers, steps; edges = derives_from, contradicts.
5. **Reasoning** — Use **rule-based reasoning** and **logical inference (deduction)**:
   - Rules: e.g. if numeric match within tolerance then not hallucinating; if symbolic form equivalent then not hallucinating.
   - Deduction: combine rule outcomes into a single verdict with explicit reasoning trace.
6. **Comparison** — Numeric (tolerance), symbolic (normalized form), and step consistency where applicable.
7. **Output** — Emit table (e.g. CSV), reasoning trace in representation + natural language, and final verdict per question.
