# Proposal Sections 3–7: Math LLM Hallucination Validator

---

## 3. Type(s) of Knowledge Needed

The system uses the following types of knowledge:

- **Question bank**: Early-college math items (algebra, calculus, trigonometry, etc.), each with an identifier, question text, and optional category. This knowledge is defined in code (e.g., `MATH_QUESTIONS`) or can be loaded from a data file.
- **Ground-truth answers**: For each question, the canonical answer from an authoritative source. This comes from the **Wolfram Alpha API**: we send the question, parse the primary pod (and optionally “Decimal approximation”), and extract the definitive result and plaintext. For equation-claim validation we also get step-by-step evaluation and a single numeric result.
- **LLM answers and reasoning traces**: The model’s final answer and its natural-language reasoning. Obtained by sending each question to an LLM (e.g., via the **OpenAI API**) and capturing both the stated answer and the reasoning trace.
- **Step-level facts (for equation claims)**: When validating an equation that an LLM claims equals a target value, we use Wolfram’s step-by-step evaluation to get intermediate subexpressions and values. These are used for FOPC-style predicates (e.g., `step(i, subexpr, value)`) and for explaining why a claim is correct or incorrect.
- **Comparison outcomes**: Numeric match (within tolerance), symbolic match (normalized form or substring), and the derived verdict (hallucinating / not hallucinating). These are the key relationships the system reasons over.

**Key entities and relationships**: questions, ground-truth answers, LLM answers, LLM reasoning text, numeric/symbolic match flags, verdicts, and (for equation mode) steps and claimed vs. actual values.

**Sources**: Wolfram Alpha API (ground truth and step-by-step), OpenAI API (LLM output), and the project’s question bank (code or data file).

---

## 4. Knowledge Representation Method(s) and Justification

The project uses three representation methods:

- **Tabular (e.g., Pandas/CSV)**  
  Each row corresponds to one question and holds: question id, question text, category, Wolfram answer, LLM answer, numeric_match, symbolic_match, verdict, and reasoning_explanation. This supports batch analysis, filtering, and reporting. **Justification**: Tabular representation is appropriate for running many questions through the same pipeline and producing a single table (e.g., `validation_results.csv`) that is easy to inspect and share.

- **First-order predicate logic (FOPC-style)**  
  Facts are represented as predicates such as `question(q_id, text)`, `ground_truth(q_id, answer)`, `llm_answer(q_id, answer)`, `llm_reasoning(q_id, text)`, and derived predicates such as `numeric_match(a, b)` and `symbolic_match(a, b)`. Deduction yields `hallucinating(q)` or `¬hallucinating(q)` from the rule: *not hallucinating* iff *numeric_match* ∨ *symbolic_match*. For equation-claim mode we add predicates like `step(i, subexpr, value)`, `result(actual)`, `claimed(target)`, and `equals` / `incorrect`. **Justification**: FOPC-style representation makes the reasoning trace explicit and auditable (e.g., in `reasoning_log.json` and equation logs), and allows clear, deterministic deduction from ground facts and rules to a verdict.

- **Knowledge graph (outside this folder)**  
  A separate knowledge graph is used elsewhere in the project. It can model nodes such as questions, answers, and steps, and edges such as *derives_from* and *contradicts*. **Justification**: A graph supports relational and structural reasoning (e.g., which steps derive from which, or which answers contradict ground truth), complementing the tabular and logical representations used in this validator codebase.

---

## 5. Type(s) of Reasoning and Justification

The system uses **rule-based reasoning** and **logical inference (deduction)**:

- **Rule-based reasoning**  
  Two main rules are applied to each (ground_truth, llm_answer) pair:  
  (1) **Numeric match**: extract numbers from both answers and compare (e.g., final value or full list) within a small tolerance (e.g., 1e-6).  
  (2) **Symbolic match**: normalize expressions (whitespace, case, notation) and check for equality or substring containment.  
  The rules are deterministic and produce explicit outcomes (e.g., “numeric_match(last: 3.0 ~ 3.0)” or “symbolic_mismatch”) that are recorded in the trace.

- **Logical inference (deduction)**  
  The verdict is deduced from the rule outcomes: *not_hallucinating* iff *numeric_match* ∨ *symbolic_match*; otherwise *hallucinating*. The premise (e.g., “numeric_match ∨ symbolic_match”) is stored with the inferred predicate so the reasoning is fully traceable.

**Justification**: Rule-based reasoning plus deduction is suitable because (1) the task is well-defined (compare two answers and classify), (2) we need consistent, repeatable verdicts, and (3) we need explainability—every verdict is backed by a concrete comparison and a stated logical premise. The implementation emits both a machine-readable trace (FOPC-style facts and inference steps) and a natural-language explanation.

---

## 6. Technologies and Tools

- **Python 3**: Main implementation language for the validator script, API calls, parsing, and file I/O.
- **`requests`**: HTTP calls to the Wolfram Alpha API to obtain ground-truth and step-by-step results.
- **Wolfram Alpha API**: Source of ground-truth answers and step-by-step evaluation for math queries and equation-claim validation (App ID configured in `wolfram_alpha.py`).
- **OpenAI API** (optional): Sending questions to an LLM (e.g., ChatGPT) and retrieving final answer and reasoning; used when `OPENAI_API_KEY` is set (environment or `.env`).
- **CSV / tabular output**: Storing validation results (question, Wolfram answer, LLM answer, match flags, verdict, reasoning) in `validation_results.csv` for analysis and reporting.
- **JSON**: Storing FOPC-style reasoning traces and step-by-step data (e.g., `reasoning_log.json`, `equation_claim_log.json`, `word_problem_log.json`) and caches (e.g., ground truth cache).
- **Knowledge graph** (outside this folder): Used elsewhere in the project for graph-based representation (nodes/edges for questions, answers, steps, derives_from, contradicts).
- **Development / environment**: `.env` for API keys (see `.env.example`); no extra framework beyond standard library plus `requests` and optional `openai`.

---

## 7. Agentic AI Capabilities (Optional)

This validator is implemented as a **fixed pipeline**: fetch ground truth → get LLM answer → apply comparison rules → deduce verdict → write outputs. It does **not** currently implement agentic AI (e.g., dynamic choice of what to do next, goal-directed replanning, or trade-off evaluation between multiple actions).

**Optional future agentic directions** (for extra credit, if implemented later):

- **Deciding when to use numeric vs. symbolic comparison**: The agent could inspect the form of the ground-truth and LLM answers and decide whether to prioritize numeric comparison, symbolic comparison, or both, and in what order.
- **Deciding whether to retry or expand**: If the LLM output is ambiguous or unparseable, the agent could decide to rephrase the question, request step-by-step from the LLM again, or fall back to a different pod from Wolfram.
- **Deciding how to handle borderline cases**: When numeric match is near the tolerance boundary or symbolic match is partial, the agent could reason about whether to ask Wolfram for an alternate form or to flag the item for human review.

These would require the system to **reason about** the current state and **choose** among actions (e.g., which comparison to run, whether to retry, how to classify edge cases), rather than always executing the same sequence. This section is optional and is included only to describe how agentic capabilities could be added; the current codebase does not claim the optional agentic extra credit.
