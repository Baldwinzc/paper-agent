"""Reviewer simulation."""

from __future__ import annotations

from paper_agent.state import PaperState, ReviewFinding


class ReviewerAgent:
    """Flags risks before the draft is treated as paper-ready."""

    def run(self, state: PaperState) -> PaperState:
        findings: list[ReviewFinding] = []
        experiments = state.get("experiments")
        innovations = state.get("innovations", [])
        sections = state.get("sections")

        for innovation in innovations:
            if "Needs manual confirmation" in innovation.risk or "Insufficient" in innovation.risk:
                findings.append(
                    ReviewFinding(
                        severity="major",
                        issue=f"{innovation.name} is not fully validated.",
                        suggestion="Confirm novelty and evidence before finalizing the Method section.",
                    )
                )

        if experiments and experiments.missing_details:
            findings.append(
                ReviewFinding(
                    severity="major",
                    issue="Experiment section lacks required details.",
                    suggestion="Fill in datasets, metrics, baselines, and exact result tables.",
                )
            )

        if sections and "Citation placeholders" in sections.related_work:
            findings.append(
                ReviewFinding(
                    severity="minor",
                    issue="Related Work still uses citation placeholders.",
                    suggestion="Add bibliography ingestion and replace placeholders with BibTeX keys.",
                )
            )

        if state.get("bibliography"):
            unverified = [
                entry
                for entry in state["bibliography"]
                if "Seed" in entry.note or "replace" in entry.note.lower() or not entry.year
            ]
            if unverified:
                findings.append(
                    ReviewFinding(
                        severity="minor",
                        issue="Bibliography contains seed entries that are not submission-ready.",
                        suggestion="Replace generated seed BibTeX entries with verified metadata from real papers.",
                    )
                )

        state["review_findings"] = [*state.get("review_findings", []), *findings]
        return state
