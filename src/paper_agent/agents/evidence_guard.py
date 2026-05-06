"""Evidence guard for generated paper drafts."""

from __future__ import annotations

import re

from paper_agent.state import DraftSections, PaperState, ReviewFinding


class EvidenceGuardAgent:
    """Checks generated text against known evidence and marks risky claims.

    This is intentionally conservative. When evidence is missing, the guard does not try
    to be clever; it replaces unsupported experimental claims with placeholders and records
    findings for the reviewer.
    """

    NO_COX_PATTERNS = [
        r"\bno\s+cox\s+loss\s+is\s+used\b",
        r"\bwithout\s+(?:a\s+)?cox\s+loss\b",
        r"\bonly\s+(?:a\s+)?single\s+reconstruction\s+loss\b",
        r"\buses\s+only\s+one\s+supervised\s+signal\b",
        r"\bsurvival outcome .* incorporated implicitly\b",
    ]
    UNSUPPORTED_EXPERIMENT_PATTERNS = [
        r"\bdemonstrat(?:e|es|ed|ing)\s+(?:improved|superior|consistent|significant)",
        r"\bconfirm(?:s|ed|ing)?\s+(?:the|that|each)",
        r"\bachiev(?:e|es|ed|ing)\s+(?:improved|superior|state-of-the-art)",
        r"\bvalidate(?:s|d|ing)?\s+(?:the|our|framework|across|on)",
        r"\bvalidated\s+across\b",
        r"\bmeasurable\s+gains\b",
        r"\bconsistent\s+gains\b",
        r"\bgreater\s+robustness\b",
        r"\bimproved\s+(?:consistency|generalizability|robustness|performance)",
        r"\bdegrad(?:e|es|ed|ing)\b",
        r"\breduc(?:e|es|ed|ing)\s+(?:IBS|C-index|performance|error|loss|variance)",
        r"\bincreas(?:e|es|ed|ing)\s+(?:C-index|AUC|performance|accuracy|variance)",
        r"\bby\s*[≥><=]?\s*\d+(?:\.\d+)?\b",
        r"\bempirical validation\b",
    ]

    def run(self, state: PaperState) -> PaperState:
        sections = state.get("sections")
        if not sections:
            return state

        findings: list[ReviewFinding] = []
        guarded = sections.model_copy(deep=True)

        if self._evidence_says_cox_is_used(state):
            guarded, cox_findings = self._guard_cox_claims(guarded)
            findings.extend(cox_findings)

        experiments = state.get("experiments")
        if experiments and experiments.missing_details:
            guarded, experiment_findings = self._guard_unsupported_experiment_claims(guarded)
            findings.extend(experiment_findings)
            guarded.experiments = self._prepend_experiment_placeholder(
                guarded.experiments,
                experiments.missing_details,
            )

        state["sections"] = guarded
        if findings:
            state.setdefault("artifacts", {})["evidence_guard_findings"] = [
                finding.model_dump() for finding in findings
            ]
            existing = state.get("review_findings", [])
            state["review_findings"] = [*existing, *findings]
        return state

    def _evidence_says_cox_is_used(self, state: PaperState) -> bool:
        evidence_parts = []
        code = state.get("code")
        baseline = state.get("baseline")
        if code:
            evidence_parts.append(code.summary)
            evidence_parts.extend(code.method_claims)
        if baseline:
            evidence_parts.extend([baseline.method, baseline.experiments, baseline.extracted_text_preview])
        evidence = "\n".join(evidence_parts).lower()
        return "cox" in evidence and any(term in evidence for term in ["l_surv", "cox ph", "partial likelihood"])

    def _guard_cox_claims(self, sections: DraftSections) -> tuple[DraftSections, list[ReviewFinding]]:
        findings: list[ReviewFinding] = []
        for field in ["abstract", "introduction", "method", "conclusion"]:
            text = getattr(sections, field)
            updated = text
            for pattern in self.NO_COX_PATTERNS:
                updated = re.sub(
                    pattern,
                    "[EVIDENCE_GUARD: revise loss description; repository evidence indicates Cox survival loss is retained alongside L_rec]",
                    updated,
                    flags=re.I,
                )
            if updated != text:
                setattr(sections, field, updated)
                findings.append(
                    ReviewFinding(
                        severity="major",
                        issue=f"Potentially incorrect Cox-loss claim in {field}.",
                        suggestion="State the objective as L_surv (Cox PH) + lambda_rec * L_rec unless updated evidence says otherwise.",
                    )
                )
        return sections, findings

    def _guard_unsupported_experiment_claims(
        self, sections: DraftSections
    ) -> tuple[DraftSections, list[ReviewFinding]]:
        findings: list[ReviewFinding] = []
        for field in ["abstract", "introduction", "experiments", "conclusion"]:
            text = getattr(sections, field)
            updated = text
            for pattern in self.UNSUPPORTED_EXPERIMENT_PATTERNS:
                updated = re.sub(
                    pattern,
                    "[EVIDENCE_GUARD: unsupported empirical claim; insert exact result after experiment table is provided]",
                    updated,
                    flags=re.I,
                )
            if updated != text:
                setattr(sections, field, updated)
                findings.append(
                    ReviewFinding(
                        severity="major",
                        issue=f"Unsupported empirical claim in {field}.",
                        suggestion="Provide exact result tables before claiming improvements, consistency, or validation.",
                    )
                )
        return sections, findings

    def _prepend_experiment_placeholder(self, text: str, missing_details: list[str]) -> str:
        placeholder = (
            "[EVIDENCE_GUARD: experiment results are incomplete. Fill in exact datasets, "
            f"metrics, baseline rows, and numerical values before final submission. Missing: {', '.join(missing_details)}.]\n\n"
        )
        if text.startswith("[EVIDENCE_GUARD: experiment results are incomplete."):
            return text
        return placeholder + text
