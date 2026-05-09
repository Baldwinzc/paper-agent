"""Heuristic comparison between baseline-paper evidence and repository evidence."""

from __future__ import annotations

import re
from typing import Iterable

from paper_agent.state import PaperState


class CodeBaselineComparisonAgent:
    """Turns baseline/code alignment into innovation-oriented evidence."""

    TECHNIQUES = (
        {
            "name": "survival prediction",
            "patterns": (r"\bsurvival\b", r"\bcox\b", r"\brisk\b"),
            "seed": "Model patient-level survival risk with evidence-grounded survival objectives.",
        },
        {
            "name": "prototype learning",
            "patterns": (r"\bprototype", r"\bprototypes\b", r"prototype bank"),
            "seed": "Use prototype-based representations as a central modeling component.",
        },
        {
            "name": "heterogeneous graph modeling",
            "patterns": (r"heterogeneous graph", r"\bgraph representation\b"),
            "seed": "Represent heterogeneous tissue relations through graph-structured modeling.",
        },
        {
            "name": "hypergraph modeling",
            "patterns": (r"\bhypergraph\b", r"\bhyperedge\b", r"\bincidence\b"),
            "seed": "Introduce hypergraph structure modeling for higher-order tissue and prototype relations.",
        },
        {
            "name": "optimal transport geometry",
            "patterns": (r"optimal transport", r"\bwasserstein\b", r"\bbarycenter\b", r"\bot\.emd\b"),
            "seed": "Construct adaptive prototype geometry with optimal-transport evidence.",
        },
        {
            "name": "bidirectional hyperedge convolution",
            "patterns": (r"\bhcon\b", r"bidirectional hyperedge", r"hyperedge-side"),
            "seed": "Use bidirectional hyperedge convolution to exchange node- and hyperedge-level context.",
        },
        {
            "name": "cross-attention fusion",
            "patterns": (r"cross[- ]attention", r"\bcrossattention\b", r"proto_fusion"),
            "seed": "Fuse prototype and instance/context representations through cross-attention.",
        },
        {
            "name": "incidence reconstruction",
            "patterns": (r"incidence reconstruction", r"binary reconstruction", r"target_h", r"\bh_recon\b"),
            "seed": "Regularize learned hypergraph structure with incidence reconstruction.",
        },
        {
            "name": "reconstruction objective",
            "patterns": (r"reconstruction objective", r"\bl_rec\b", r"hcon_rec_loss", r"binary_cross_entropy"),
            "seed": "Jointly optimize survival prediction with reconstruction evidence from the implementation.",
        },
        {
            "name": "regularizer simplification",
            "patterns": (r"legacy loss removal", r"compatibility loss", r"orthogonal_regularization"),
            "seed": "Simplify the training objective by removing unsupported legacy regularizers.",
        },
    )

    def run(self, state: PaperState) -> PaperState:
        state.setdefault("artifacts", {})["code_baseline_comparison"] = self._compare(state)
        return state

    def _compare(self, state: PaperState) -> dict[str, object]:
        baseline = state.get("baseline")
        code = state.get("code")
        if not baseline or not code:
            return {
                "mode": "insufficient_evidence",
                "overlapping_terms": [],
                "baseline_only_terms": [],
                "code_only_terms": [],
                "likely_method_shifts": [],
                "innovation_seeds": [],
                "cautions": ["Both baseline and code evidence are required for comparison."],
            }

        baseline_terms = self._present_techniques(self._baseline_text(baseline))
        code_terms = self._present_techniques(self._code_text(code))
        overlap = [term for term in code_terms if term in baseline_terms]
        code_only = [term for term in code_terms if term not in baseline_terms]
        baseline_only = [term for term in baseline_terms if term not in code_terms]
        shifts = self._method_shifts(code_only, code, overlap)
        seeds = self._innovation_seeds(shifts, overlap)
        cautions = [
            "This comparison is lexical and code-evidence based; authors must confirm novelty before final claims."
        ]
        if not shifts:
            cautions.append("No code-side method shift was detected beyond the extracted baseline terms.")
        return {
            "mode": "compared",
            "overlapping_terms": overlap,
            "baseline_only_terms": baseline_only,
            "code_only_terms": code_only,
            "likely_method_shifts": shifts,
            "innovation_seeds": seeds,
            "cautions": cautions,
        }

    def _baseline_text(self, baseline) -> str:
        sections = " ".join((baseline.structured_sections or {}).values())
        return " ".join(
            [
                baseline.title or "",
                baseline.problem or "",
                baseline.method or "",
                baseline.experiments or "",
                " ".join(baseline.limitations or []),
                " ".join(baseline.related_terms or []),
                sections,
            ]
        )

    def _code_text(self, code) -> str:
        return " ".join(
            [
                code.summary or "",
                " ".join(code.likely_method_files or []),
                " ".join(code.method_claims or []),
                " ".join(code.implementation_evidence or []),
            ]
        )

    def _present_techniques(self, text: str) -> list[str]:
        lowered = text.lower()
        present: list[str] = []
        for technique in self.TECHNIQUES:
            patterns = technique["patterns"]
            if any(re.search(pattern, lowered, flags=re.I) for pattern in patterns):
                present.append(str(technique["name"]))
        return present

    def _method_shifts(self, code_only_terms: list[str], code, overlap: list[str]) -> list[dict[str, object]]:
        shifts = []
        if overlap and code_only_terms:
            shifts.append(
                {
                    "technique": "problem-setting continuity",
                    "rationale": (
                        "The code keeps the same broad task context while adding implementation-backed "
                        "method components that should be framed as innovation points."
                    ),
                    "evidence": [f"Shared technical context: {', '.join(overlap[:4])}."],
                }
            )
        for term in code_only_terms[:6]:
            evidence = self._matching_evidence(term, code)
            shifts.append(
                {
                    "technique": term,
                    "rationale": (
                        "Repository evidence supports this as a proposed-method component; "
                        "phrase it as an innovation only after author novelty review."
                    ),
                    "evidence": evidence,
                }
            )
        return shifts

    def _matching_evidence(self, term: str, code) -> list[str]:
        patterns = self._patterns_for(term)
        candidates = list(code.implementation_evidence or []) + list(code.method_claims or [])
        matches = [item for item in candidates if self._matches_any(item, patterns)]
        if not matches and code.summary:
            matches = [code.summary]
        return matches[:3]

    def _patterns_for(self, term: str) -> tuple[str, ...]:
        for technique in self.TECHNIQUES:
            if technique["name"] == term:
                return tuple(str(pattern) for pattern in technique["patterns"])
        return (re.escape(term),)

    def _matches_any(self, text: str, patterns: Iterable[str]) -> bool:
        lowered = text.lower()
        return any(re.search(pattern, lowered, flags=re.I) for pattern in patterns)

    def _innovation_seeds(self, shifts: list[dict[str, object]], overlap: list[str]) -> list[str]:
        seeds: list[str] = []
        for shift in shifts:
            term = str(shift.get("technique", ""))
            if term == "problem-setting continuity":
                continue
            seed = self._seed_for(term)
            if seed:
                seeds.append(seed)
        if not seeds and overlap:
            seeds.append(
                f"Retain the {', '.join(overlap[:3])} setting while requiring explicit author notes for novelty."
            )
        return list(dict.fromkeys(seeds))[:5]

    def _seed_for(self, term: str) -> str:
        for technique in self.TECHNIQUES:
            if technique["name"] == term:
                return str(technique["seed"])
        return ""
