"""
Real FOPC (First-Order Predicate Calculus) engine for math hallucination validation.
Implements: terms, predicates, clauses, unification (Robinson), and resolution/forward chaining.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# -------------------------
# Terms
# -------------------------
@dataclass(frozen=True)
class Term:
    """Base class for FOPC terms (constants and variables)."""

    def is_variable(self) -> bool:
        return isinstance(self, Var)

    def is_constant(self) -> bool:
        return isinstance(self, Const)


@dataclass(frozen=True)
class Const(Term):
    """Constant term, e.g. Const('q1'), Const(48.5)."""
    value: str | float | bool

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Var(Term):
    """Variable term, e.g. Var('G'), Var('L')."""
    name: str

    def __str__(self) -> str:
        return self.name


# -------------------------
# Predicates (atoms)
# -------------------------
@dataclass
class Atom:
    """Predicate atom: P(t1, t2, ...)."""
    pred: str
    args: tuple[Term, ...]

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.pred}({args_str})"

    def copy(self) -> Atom:
        return Atom(self.pred, tuple(a for a in self.args))

    def substitute(self, subst: dict[str, Term]) -> Atom:
        """Apply substitution to all terms in this atom."""
        new_args = []
        for a in self.args:
            if isinstance(a, Var) and a.name in subst:
                new_args.append(subst[a.name])
            else:
                new_args.append(a)
        return Atom(self.pred, tuple(new_args))


# -------------------------
# Literals and clauses
# -------------------------
@dataclass
class Literal:
    """Literal: atom or negated atom."""
    atom: Atom
    negated: bool = False

    def __str__(self) -> str:
        return ("¬" if self.negated else "") + str(self.atom)

    def substitute(self, subst: dict[str, Term]) -> Literal:
        return Literal(self.atom.substitute(subst), self.negated)


@dataclass
class Clause:
    """Horn clause: head ← body (head implied by conjunction of body literals)."""
    head: Literal | None  # None for goal/query clause
    body: list[Literal]

    def __str__(self) -> str:
        if self.head is None:
            return "?-" + ", ".join(str(b) for b in self.body)
        if not self.body:
            return str(self.head)
        return str(self.head) + " ← " + ", ".join(str(b) for b in self.body)


# -------------------------
# Unification (Robinson's algorithm)
# -------------------------
def occurs_check(var: Var, term: Term) -> bool:
    """True if var occurs in term (prevents infinite unification)."""
    if var == term:
        return True
    if isinstance(term, Const):
        return False
    if isinstance(term, Var):
        return var.name == term.name
    return False


def unify(term1: Term, term2: Term, subst: dict[str, Term] | None = None) -> dict[str, Term] | None:
    """
    Robinson's unification algorithm.
    Returns substitution that makes term1 = term2, or None if not unifiable.
    """
    subst = subst.copy() if subst else {}
    return _unify(term1, term2, subst)


def _unify(term1: Term, term2: Term, subst: dict[str, Term]) -> dict[str, Term] | None:
    # Apply current substitution
    t1 = subst.get(term1.name, term1) if isinstance(term1, Var) else term1
    t2 = subst.get(term2.name, term2) if isinstance(term2, Var) else term2

    if t1 == t2:
        return subst
    if isinstance(t1, Var):
        if occurs_check(t1, t2):
            return None
        subst[t1.name] = t2
        return subst
    if isinstance(t2, Var):
        if occurs_check(t2, t1):
            return None
        subst[t2.name] = t1
        return subst
    return None


def unify_atoms(a1: Atom, a2: Atom) -> dict[str, Term] | None:
    """Unify two atoms. Returns substitution if they unify, else None."""
    if a1.pred != a2.pred or len(a1.args) != len(a2.args):
        return None
    subst: dict[str, Term] = {}
    for t1, t2 in zip(a1.args, a2.args):
        result = unify(t1, t2, subst)
        if result is None:
            return None
        subst = result
    return subst


# -------------------------
# Knowledge base and inference
# -------------------------
@dataclass
class KnowledgeBase:
    """FOPC knowledge base: clauses + facts. Supports forward chaining and resolution."""
    clauses: list[Clause] = field(default_factory=list)
    facts: list[Atom] = field(default_factory=list)

    def add_fact(self, atom: Atom) -> None:
        """Add a fact as a unit clause (head with empty body)."""
        self.facts.append(atom)

    def add_clause(self, clause: Clause) -> None:
        self.clauses.append(clause)

    def add_rule(self, head: Atom, body: list[Literal]) -> None:
        """Add rule: head ← body."""
        self.clauses.append(Clause(Literal(head, False), body))

    def prove(self, goal: Literal) -> tuple[bool, list[dict[str, Any]]]:
        """
        Prove goal using resolution/backward chaining.
        Returns (success, proof_trace).
        """
        trace: list[dict[str, Any]] = []
        result = self._resolve([goal], trace, depth=0, max_depth=20)
        return (result, trace)

    def _resolve(self, goals: list[Literal], trace: list[dict], depth: int, max_depth: int) -> bool:
        if depth > max_depth:
            return False
        if not goals:
            return True

        goal = goals[0]
        rest_goals = goals[1:]

        # Try facts
        for fact in self.facts:
            if goal.negated:
                continue
            subst = unify_atoms(goal.atom, fact)
            if subst is not None:
                trace.append({
                    "step": "fact_match",
                    "goal": str(goal),
                    "fact": str(fact),
                    "subst": {k: str(v) for k, v in subst.items()},
                })
                new_goals = [g.substitute(subst) for g in rest_goals]
                if self._resolve(new_goals, trace, depth + 1, max_depth):
                    return True

        # Try clauses
        for clause in self.clauses:
            if clause.head is None:
                continue
            if goal.negated and clause.head.negated:
                continue
            if goal.negated != clause.head.negated:
                continue
            subst = unify_atoms(goal.atom, clause.head.atom)
            if subst is not None:
                trace.append({
                    "step": "rule_match",
                    "goal": str(goal),
                    "clause": str(clause),
                    "subst": {k: str(v) for k, v in subst.items()},
                })
                new_body = [lit.substitute(subst) for lit in clause.body]
                new_goals = new_body + [g.substitute(subst) for g in rest_goals]
                if self._resolve(new_goals, trace, depth + 1, max_depth):
                    return True

        return False

# -------------------------
# Simplified FOPC engine for our domain
# -------------------------
def fopc_deduce_verdict(
    numeric_match: bool,
    symbolic_match: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Real FOPC deduction: derive verdict from facts using logical inference.
    Rule: ¬hallucinating(q) ↔ (numeric_match(gt,llm) ∨ symbolic_match(gt,llm))
    Implemented via resolution over Horn clauses.
    """
    kb = KnowledgeBase()

    # Facts (from comparison oracle)
    if numeric_match:
        kb.add_fact(Atom("numeric_match", (Const("gt"), Const("llm"))))
    if symbolic_match:
        kb.add_fact(Atom("symbolic_match", (Const("gt"), Const("llm"))))

    # Rules (Horn clauses):
    # not_hallucinating(q) ← numeric_match(gt, llm)
    kb.add_clause(Clause(
        Literal(Atom("not_hallucinating", (Var("q"),)), False),
        [Literal(Atom("numeric_match", (Var("G"), Var("L"))), False)],
    ))
    # not_hallucinating(q) ← symbolic_match(gt, llm)
    kb.add_clause(Clause(
        Literal(Atom("not_hallucinating", (Var("q"),)), False),
        [Literal(Atom("symbolic_match", (Var("G"), Var("L"))), False)],
    ))

    # Query: not_hallucinating(q)
    goal = Literal(Atom("not_hallucinating", (Const("q"),)), False)
    success, trace = kb.prove(goal)

    verdict = "not_hallucinating" if success else "hallucinating"
    return (verdict, trace)


def fopc_deduce_equation_verdict(equals_claimed: bool) -> tuple[str, list[dict[str, Any]]]:
    """
    Real FOPC deduction for equation-claim verdict.
    Rule: not_hallucinating_eq ← equals(claimed, result)
          hallucinating_eq ← not_equals(claimed, result)
    Python oracle asserts equals/not_equals; FOPC derives verdict via resolution.
    """
    kb = KnowledgeBase()
    if equals_claimed:
        kb.add_fact(Atom("equals", (Const("claimed"), Const("result"))))
    else:
        kb.add_fact(Atom("not_equals", (Const("claimed"), Const("result"))))

    # not_hallucinating_eq ← equals(claimed, result)
    kb.add_clause(Clause(
        Literal(Atom("not_hallucinating_eq", (Var("q"),)), False),
        [Literal(Atom("equals", (Var("C"), Var("R"))), False)],
    ))
    # hallucinating_eq ← not_equals(claimed, result)
    kb.add_clause(Clause(
        Literal(Atom("hallucinating_eq", (Var("q"),)), False),
        [Literal(Atom("not_equals", (Var("C"), Var("R"))), False)],
    ))

    if equals_claimed:
        goal = Literal(Atom("not_hallucinating_eq", (Const("q"),)), False)
        success, trace = kb.prove(goal)
        verdict = "not_hallucinating" if success else "hallucinating"
    else:
        goal = Literal(Atom("hallucinating_eq", (Const("q"),)), False)
        success, trace = kb.prove(goal)
        verdict = "hallucinating" if success else "not_hallucinating"
    return (verdict, trace)


def fopc_abduce_causes(
    hallucinating: bool,
    numeric_ok: bool,
    symbolic_ok: bool,
    has_numbers_in_both: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Real FOPC abduction: derive possible causes when hallucinating.
    Rules:
      abduced_cause(numeric_error) ← hallucinating ∧ numeric_mismatch ∧ has_numbers
      abduced_cause(symbolic_error) ← hallucinating ∧ symbolic_mismatch
      abduced_cause(both) ← hallucinating ∧ numeric_mismatch ∧ symbolic_mismatch
      abduced_cause(form_difference_only) ← hallucinating ∧ numeric_match ∧ symbolic_mismatch
    Returns (causes with proof traces, all_proofs).
    """
    kb = KnowledgeBase()
    if hallucinating:
        kb.add_fact(Atom("hallucinating", (Const("q"),)))
    if numeric_ok:
        kb.add_fact(Atom("numeric_match", (Const("gt"), Const("llm"))))
    else:
        kb.add_fact(Atom("numeric_mismatch", (Const("gt"), Const("llm"))))
    if symbolic_ok:
        kb.add_fact(Atom("symbolic_match", (Const("gt"), Const("llm"))))
    else:
        kb.add_fact(Atom("symbolic_mismatch", (Const("gt"), Const("llm"))))
    if has_numbers_in_both:
        kb.add_fact(Atom("has_numbers", (Const("gt"), Const("llm"))))

    # abduced_cause(numeric_error) ← hallucinating ∧ numeric_mismatch ∧ has_numbers
    kb.add_clause(Clause(
        Literal(Atom("abduced_cause", (Const("numeric_error"),)), False),
        [
            Literal(Atom("hallucinating", (Var("q"),)), False),
            Literal(Atom("numeric_mismatch", (Var("G"), Var("L"))), False),
            Literal(Atom("has_numbers", (Var("G2"), Var("L2"))), False),
        ],
    ))
    # abduced_cause(symbolic_error) ← hallucinating ∧ symbolic_mismatch
    kb.add_clause(Clause(
        Literal(Atom("abduced_cause", (Const("symbolic_error"),)), False),
        [
            Literal(Atom("hallucinating", (Var("q"),)), False),
            Literal(Atom("symbolic_mismatch", (Var("G"), Var("L"))), False),
        ],
    ))
    # abduced_cause(both_numeric_and_symbolic) ← hallucinating ∧ numeric_mismatch ∧ symbolic_mismatch
    kb.add_clause(Clause(
        Literal(Atom("abduced_cause", (Const("both_numeric_and_symbolic"),)), False),
        [
            Literal(Atom("hallucinating", (Var("q"),)), False),
            Literal(Atom("numeric_mismatch", (Var("G"), Var("L"))), False),
            Literal(Atom("symbolic_mismatch", (Var("G2"), Var("L2"))), False),
        ],
    ))
    # abduced_cause(form_difference_only) ← hallucinating ∧ numeric_match ∧ symbolic_mismatch
    kb.add_clause(Clause(
        Literal(Atom("abduced_cause", (Const("form_difference_only"),)), False),
        [
            Literal(Atom("hallucinating", (Var("q"),)), False),
            Literal(Atom("numeric_match", (Var("G"), Var("L"))), False),
            Literal(Atom("symbolic_mismatch", (Var("G2"), Var("L2"))), False),
        ],
    ))

    hypothesis_forms = {
        "numeric_error": "LLM may have made a numerical mistake (wrong constant, wrong operation, or rounding).",
        "symbolic_error": "LLM may have used a different (incorrect) symbolic form or expression.",
        "both_numeric_and_symbolic": "Both numeric and symbolic mismatch; hallucination likely substantive.",
        "form_difference_only": "Numbers agree but form differs (e.g. equivalent expression in different form).",
    }
    causes = []
    all_proofs = []
    for hyp in ["numeric_error", "symbolic_error", "both_numeric_and_symbolic", "form_difference_only"]:
        goal = Literal(Atom("abduced_cause", (Const(hyp),)), False)
        success, trace = kb.prove(goal)
        if success:
            causes.append({
                "predicate": "abduced_cause",
                "hypothesis": hyp,
                "form": hypothesis_forms[hyp],
                "fopc_proof": trace,
            })
            all_proofs.extend(trace)
    return (causes, all_proofs)


# -------------------------
# Evidence strength (confidence) from relative error
# -------------------------
def fopc_deduce_evidence_strength(
    rel_err: float | None,
    match: bool,
    tolerance: float = 1e-6,
) -> tuple[str, str, list[dict[str, Any]]]:
    """
    FOPC deduction: derive evidence_strength (confidence) from rel_err facts.
    Rules:
      evidence_strength(high)     ← rel_err_below(tolerance)           # match
      evidence_strength(medium)   ← rel_err_below(0.01) ∧ ¬match
      evidence_strength(high)     ← rel_err_below(0.5) ∧ rel_err_above(0.01)
      evidence_strength(very_high)← rel_err_above(0.5)
      evidence_strength(unknown)  ← no_rel_err
    Oracle asserts rel_err_below/above from numeric rel_err; FOPC derives level.
    Returns (confidence_level, explanation, proof_trace).
    """
    kb = KnowledgeBase()

    if rel_err is None:
        kb.add_fact(Atom("no_rel_err", (Const("q"),)))
        kb.add_clause(Clause(
            Literal(Atom("evidence_strength", (Const("unknown"),)), False),
            [Literal(Atom("no_rel_err", (Var("q"),)), False)],
        ))
        goal = Literal(Atom("evidence_strength", (Const("unknown"),)), False)
        success, trace = kb.prove(goal)
        return ("unknown", "Cannot compute relative error (missing numbers).", trace)

    # Oracle: assert rel_err facts from thresholds
    if rel_err <= tolerance:
        kb.add_fact(Atom("rel_err_below", (Const("tolerance"),)))
        goal = Literal(Atom("evidence_strength", (Const("high"),)), False)
        success, trace = kb.prove(goal)
        return ("high", f"Relative error ≈ {rel_err:.2e} (match).", trace)

    kb.add_fact(Atom("not_match", (Const("q"),)))
    if rel_err < 0.01:
        kb.add_fact(Atom("rel_err_below", (Const("0.01"),)))
        kb.add_fact(Atom("rel_err_above", (Const("tolerance"),)))
    elif rel_err < 0.5:
        kb.add_fact(Atom("rel_err_below", (Const("0.5"),)))
        kb.add_fact(Atom("rel_err_above", (Const("0.01"),)))
    else:
        kb.add_fact(Atom("rel_err_above", (Const("0.5"),)))

    # Rules
    kb.add_clause(Clause(
        Literal(Atom("evidence_strength", (Const("high"),)), False),
        [Literal(Atom("rel_err_below", (Const("tolerance"),)), False)],
    ))
    kb.add_clause(Clause(
        Literal(Atom("evidence_strength", (Const("medium"),)), False),
        [
            Literal(Atom("rel_err_below", (Const("0.01"),)), False),
            Literal(Atom("not_match", (Var("q"),)), False),
        ],
    ))
    kb.add_clause(Clause(
        Literal(Atom("evidence_strength", (Const("high"),)), False),
        [
            Literal(Atom("rel_err_below", (Const("0.5"),)), False),
            Literal(Atom("rel_err_above", (Const("0.01"),)), False),
        ],
    ))
    kb.add_clause(Clause(
        Literal(Atom("evidence_strength", (Const("very_high"),)), False),
        [Literal(Atom("rel_err_above", (Const("0.5"),)), False)],
    ))

    for level in ["very_high", "high", "medium"]:
        goal = Literal(Atom("evidence_strength", (Const(level),)), False)
        success, trace = kb.prove(goal)
        if success:
            if level == "very_high":
                expl = f"Relative error ≈ {rel_err:.2%}; strong evidence of hallucination."
            elif level == "medium":
                expl = f"Relative error ≈ {rel_err:.2%}; small numeric deviation."
            else:
                expl = f"Relative error ≈ {rel_err:.2%}; moderate evidence of hallucination." if rel_err > tolerance else f"Relative error ≈ {rel_err:.2e} (match)."
            return (level, expl, trace)

    return ("unknown", f"Relative error ≈ {rel_err:.2%}", [])


# -------------------------
# Inference chain from FOPC proof trace
# -------------------------
def fopc_build_inference_chain(
    gt_text: str,
    llm_answer: str,
    num_ok: bool,
    sym_ok: bool,
    verdict: str,
    fopc_proof: list[dict[str, Any]],
) -> list[dict]:
    """
    Build inference chain from FOPC proof trace instead of fixed template.
    Maps fact_match/rule_match steps to premise → observation → rule → conclusion.
    """
    chain: list[dict] = []
    step_num = 1

    # Premises
    chain.append({"step": step_num, "type": "premise", "form": "P1: ground_truth(G)", "value": gt_text[:80]})
    step_num += 1
    chain.append({"step": step_num, "type": "premise", "form": "P2: llm_answer(L)", "value": llm_answer[:80]})
    step_num += 1

    # Observations (from oracle / comparison)
    chain.append({"step": step_num, "type": "observation", "form": f"P3: numeric_match(G,L) = {num_ok}", "value": num_ok})
    step_num += 1
    chain.append({"step": step_num, "type": "observation", "form": f"P4: symbolic_match(G,L) = {sym_ok}", "value": sym_ok})
    step_num += 1

    # Rule applications from FOPC proof trace
    for i, proof_step in enumerate(fopc_proof):
        if proof_step.get("step") == "rule_match":
            chain.append({
                "step": step_num,
                "type": "rule",
                "form": f"R{i+1}: {proof_step.get('clause', '')}",
                "value": None,
                "fopc_proof_step": proof_step,
            })
            step_num += 1
        elif proof_step.get("step") == "fact_match":
            chain.append({
                "step": step_num,
                "type": "fact_match",
                "form": f"F{i+1}: {proof_step.get('goal', '')} ← {proof_step.get('fact', '')}",
                "value": proof_step.get("subst", {}),
                "fopc_proof_step": proof_step,
            })
            step_num += 1

    # Conclusion
    chain.append({"step": step_num, "type": "conclusion", "form": f"C: verdict = {verdict}", "value": verdict})
    return chain


# -------------------------
# Aggregation rules
# -------------------------
def fopc_aggregate_result(
    n: int,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    FOPC deduction: aggregate_result(N, K, R) ← all_processed(N) ∧ count_hallucinating(K) ∧ rate(K, N, R).
    Oracle computes R = K/N; FOPC derives aggregate_result via resolution.
    Returns (aggregation_dict with fopc_aggregation, proof_trace).
    """
    rate = (k / n) if n else 0
    kb = KnowledgeBase()

    kb.add_fact(Atom("all_processed", (Const(str(n)),)))
    kb.add_fact(Atom("count_hallucinating", (Const(str(k)),)))
    kb.add_fact(Atom("rate", (Const(str(k)), Const(str(n)), Const(str(rate)))))

    kb.add_clause(Clause(
        Literal(Atom("aggregate_result", (Var("N"), Var("K"), Var("R"))), False),
        [
            Literal(Atom("all_processed", (Var("N"),)), False),
            Literal(Atom("count_hallucinating", (Var("K"),)), False),
            Literal(Atom("rate", (Var("K"), Var("N"), Var("R"))), False),
        ],
    ))

    goal = Literal(Atom("aggregate_result", (Const(str(n)), Const(str(k)), Const(str(rate)))), False)
    success, trace = kb.prove(goal)

    return (
        {
            "total_questions": n,
            "hallucination_count": k,
            "hallucination_rate": rate,
            "fopc_aggregation": [
                {"predicate": "all_processed", "args": [n], "value": n, "form": f"all_processed({n})"},
                {"predicate": "count_hallucinating", "args": [k], "value": k, "form": f"count_hallucinating({k})"},
                {"predicate": "rate", "args": [k, n], "value": rate, "form": f"rate({k}, {n}) = {rate:.2%}"},
                {"predicate": "aggregate_result", "args": [n, k, rate], "value": rate, "form": f"aggregate_result({n}, {k}, {rate:.2%})", "fopc_proof": trace},
            ],
        },
        trace,
    )


# -------------------------
# Equation: where diverges / first wrong step
# -------------------------
def fopc_deduce_diverges_at(
    hallucination_location: list[dict],
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    FOPC deduction: diverges_at(Subexpr) ← step has significant_discrepancy ∧ first_in_eval_order.
    Oracle provides hallucination_location (step_index, subexpr, ...); FOPC derives diverges_at(Subexpr).
    Returns (list of subexprs where divergence occurs, proof_trace).
    """
    if not hallucination_location:
        return ([], [])

    kb = KnowledgeBase()

    # Find first in eval order (min step_index) that accounts for total error, or largest discrepancy
    first_wrong = [loc for loc in hallucination_location if loc.get("first_wrong") or loc.get("in_gap_combo")]
    if not first_wrong:
        # Fallback: largest abs_discrepancy, then earliest step_index
        sorted_loc = sorted(
            hallucination_location,
            key=lambda x: (-(x.get("abs_discrepancy") or 0), x.get("step_index", 9999)),
        )
        first_wrong = sorted_loc[:1] if sorted_loc else []

    for loc in hallucination_location:
        subexpr = loc.get("subexpr", "")
        step_idx = loc.get("step_index", 9999)
        if subexpr and loc.get("abs_discrepancy") is not None:
            kb.add_fact(Atom("step_with_discrepancy", (Const(subexpr), Const(str(step_idx)))))
        if loc in first_wrong or (first_wrong and loc.get("subexpr") == first_wrong[0].get("subexpr")):
            kb.add_fact(Atom("first_in_eval_order", (Const(subexpr),)))

    kb.add_clause(Clause(
        Literal(Atom("diverges_at", (Var("S"),)), False),
        [
            Literal(Atom("step_with_discrepancy", (Var("S"), Var("I"))), False),
            Literal(Atom("first_in_eval_order", (Var("S"),)), False),
        ],
    ))

    diverges: list[str] = []
    all_traces: list[dict] = []
    for loc in first_wrong:
        subexpr = loc.get("subexpr", "")
        if subexpr:
            goal = Literal(Atom("diverges_at", (Const(subexpr),)), False)
            success, trace = kb.prove(goal)
            if success:
                diverges.append(subexpr)
                all_traces.extend(trace)

    return (diverges, all_traces)


# -------------------------
# Meta-rules: prioritization
# -------------------------
def fopc_deduce_priority(
    hallucinating: bool,
    evidence_strength: str,
    abduced_causes: list[dict],
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    FOPC deduction: high_priority_hallucination ← hallucinating ∧ evidence_strength(very_high)
                    needs_human_review ← hallucinating ∧ abduced_cause(both_numeric_and_symbolic)
    Returns (list of priority flags, proof_trace).
    """
    kb = KnowledgeBase()

    if hallucinating:
        kb.add_fact(Atom("hallucinating", (Const("q"),)))
    if evidence_strength == "very_high":
        kb.add_fact(Atom("evidence_strength", (Const("very_high"),)))
    has_both = any(c.get("hypothesis") == "both_numeric_and_symbolic" for c in abduced_causes)
    if has_both:
        kb.add_fact(Atom("abduced_cause", (Const("both_numeric_and_symbolic"),)))

    kb.add_clause(Clause(
        Literal(Atom("high_priority_hallucination", (Var("q"),)), False),
        [
            Literal(Atom("hallucinating", (Var("q"),)), False),
            Literal(Atom("evidence_strength", (Const("very_high"),)), False),
        ],
    ))
    kb.add_clause(Clause(
        Literal(Atom("needs_human_review", (Var("q"),)), False),
        [
            Literal(Atom("hallucinating", (Var("q"),)), False),
            Literal(Atom("abduced_cause", (Const("both_numeric_and_symbolic"),)), False),
        ],
    ))

    flags: list[str] = []
    all_traces: list[dict] = []
    for pred in ["high_priority_hallucination", "needs_human_review"]:
        goal = Literal(Atom(pred, (Const("q"),)), False)
        success, trace = kb.prove(goal)
        if success:
            flags.append(pred)
            all_traces.extend(trace)

    return (flags, all_traces)


# -------------------------
# Error verdict
# -------------------------
def fopc_deduce_error_verdict(wolfram_failed: bool) -> tuple[str, list[dict[str, Any]]]:
    """
    FOPC deduction: verdict(error) ← wolfram_query_failed(Question)
    """
    kb = KnowledgeBase()
    if wolfram_failed:
        kb.add_fact(Atom("wolfram_query_failed", (Const("question"),)))

    kb.add_clause(Clause(
        Literal(Atom("verdict", (Const("error"),)), False),
        [Literal(Atom("wolfram_query_failed", (Var("Q"),)), False)],
    ))

    goal = Literal(Atom("verdict", (Const("error"),)), False)
    success, trace = kb.prove(goal)
    return ("error" if success else "unknown", trace)


# -------------------------
# Category-based rules
# -------------------------
def fopc_expect_symbolic(category: str) -> tuple[bool, list[dict[str, Any]]]:
    """
    FOPC deduction: expect_symbolic_answer(Q) ← category(Q, calculus)
    Categories that typically expect symbolic answers: calculus, algebra.
    """
    kb = KnowledgeBase()
    if category:
        kb.add_fact(Atom("category", (Const("q"), Const(category))))

    kb.add_clause(Clause(
        Literal(Atom("expect_symbolic_answer", (Var("q"),)), False),
        [Literal(Atom("category", (Var("q"), Const("calculus"))), False)],
    ))
    kb.add_clause(Clause(
        Literal(Atom("expect_symbolic_answer", (Var("q"),)), False),
        [Literal(Atom("category", (Var("q"), Const("algebra"))), False)],
    ))

    goal = Literal(Atom("expect_symbolic_answer", (Const("q"),)), False)
    success, trace = kb.prove(goal)
    return (success, trace)
