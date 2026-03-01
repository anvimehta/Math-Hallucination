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
