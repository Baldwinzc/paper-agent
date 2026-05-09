"""Submission readiness scoring for generated drafts."""

from __future__ import annotations

from paper_agent.state import PaperState


class SubmissionReadinessAgent:
    """Summarizes draft readiness into scores and concrete action items."""

    SECTION_NAMES = ("abstract", "introduction", "related_work", "method", "experiments", "conclusion")

    def run(self, state: PaperState) -> PaperState:
        state.setdefault("artifacts", {})["submission_readiness"] = self._readiness(state)
        return state

    def _readiness(self, state: PaperState) -> dict[str, object]:
        scores = {
            "evidence_grounding": self._evidence_grounding_score(state),
            "writing_completeness": self._writing_completeness_score(state),
            "citation_readiness": self._citation_readiness_score(state),
            "venue_package": self._venue_package_score(state),
        }
        overall = round(
            scores["evidence_grounding"] * 0.35
            + scores["writing_completeness"] * 0.25
            + scores["citation_readiness"] * 0.25
            + scores["venue_package"] * 0.15
        )
        blocking_items = self._blocking_items(state)
        action_items = self._action_items(state, blocking_items)
        return {
            "overall_score": overall,
            "status": self._status(overall, blocking_items),
            "scores": scores,
            "blocking_items": blocking_items,
            "action_items": action_items,
        }

    def _evidence_grounding_score(self, state: PaperState) -> int:
        score = 100
        experiments = state.get("experiments")
        if not state.get("baseline"):
            score -= 15
        if not state.get("code"):
            score -= 10
        if not experiments:
            score -= 25
        elif experiments.missing_details:
            score -= min(30, len(experiments.missing_details) * 10)
        elif not experiments.result_tables:
            score -= 15

        findings = state.get("review_findings", [])
        score -= sum(20 if finding.severity == "major" else 5 for finding in findings)
        consistency = state.get("artifacts", {}).get("factual_consistency", [])
        score -= sum(10 for item in consistency if item.get("status") == "needs_review")
        guard_findings = state.get("artifacts", {}).get("evidence_guard_findings", [])
        score -= min(30, len(guard_findings) * 10)
        return self._clamp(score)

    def _writing_completeness_score(self, state: PaperState) -> int:
        sections = state.get("sections")
        if not sections:
            return 0
        section_values = sections.model_dump()
        present = sum(1 for name in self.SECTION_NAMES if len(section_values.get(name, "").strip()) >= 80)
        score = round(100 * present / len(self.SECTION_NAMES))
        section_errors = state.get("artifacts", {}).get("section_writer_section_errors", {})
        score -= min(25, len(section_errors) * 6)
        outline_hits = state.get("artifacts", {}).get("outline_language_hits", [])
        score -= min(20, len(outline_hits) * 5)
        return self._clamp(score)

    def _citation_readiness_score(self, state: PaperState) -> int:
        bibliography = state.get("bibliography", [])
        artifacts = state.get("artifacts", {})
        if not bibliography:
            return 35
        verification = artifacts.get("reference_verification", {})
        if not verification:
            score = 55 if artifacts.get("reference_resolver_mode") == "disabled" else 60
        else:
            resolved = int(verification.get("resolved_count", 0) or 0)
            unresolved = int(verification.get("unresolved_count", 0) or 0)
            total = max(1, resolved + unresolved)
            score = round(100 * resolved / total)
            score -= min(30, unresolved * 8)

        coverage = artifacts.get("related_work_citation_coverage", [])
        missing_coverage = [
            item
            for item in coverage
            if item.get("requires_citation") and not item.get("covered_by_real_citation")
        ]
        score -= min(30, len(missing_coverage) * 10)
        score -= min(30, len(artifacts.get("undefined_citation_keys", [])) * 15)
        return self._clamp(score)

    def _venue_package_score(self, state: PaperState) -> int:
        score = 30
        if state.get("venue_template"):
            score += 25
        if state.get("latex_output_path"):
            score += 25
        if state.get("latex_project_dir"):
            score += 10
        if state.get("latex_zip_path"):
            score += 10
        return self._clamp(score)

    def _blocking_items(self, state: PaperState) -> list[str]:
        items: list[str] = []
        experiments = state.get("experiments")
        if experiments and experiments.missing_details:
            items.append("Experiment details are incomplete.")
        if experiments and not experiments.result_tables:
            items.append("No structured trained-model result table was parsed.")
        for finding in state.get("review_findings", []):
            if finding.severity == "major":
                items.append(finding.issue)
        for item in state.get("artifacts", {}).get("factual_consistency", []):
            if item.get("status") == "needs_review":
                items.append(f"Factual consistency needs review: {item.get('check')}.")
        undefined = state.get("artifacts", {}).get("undefined_citation_keys", [])
        if undefined:
            items.append("Undefined citation keys remain: " + ", ".join(undefined[:5]) + ".")
        return list(dict.fromkeys(items))

    def _action_items(self, state: PaperState, blocking_items: list[str]) -> list[str]:
        items = list(blocking_items)
        artifacts = state.get("artifacts", {})
        verification = artifacts.get("reference_verification", {})
        unresolved = int(verification.get("unresolved_count", 0) or 0)
        if unresolved:
            items.append(f"Resolve or remove {unresolved} unresolved bibliography seed entries.")
        elif artifacts.get("reference_resolver_mode") == "disabled":
            items.append("Run with online reference resolution before citation-sensitive submission.")
        missing_coverage = [
            item
            for item in artifacts.get("related_work_citation_coverage", [])
            if item.get("requires_citation") and not item.get("covered_by_real_citation")
        ]
        if missing_coverage:
            labels = ", ".join(str(item.get("thread")) for item in missing_coverage[:3])
            items.append(f"Add real citations for related-work threads: {labels}.")
        section_errors = artifacts.get("section_writer_section_errors", {})
        if section_errors:
            sections = ", ".join(section_errors.keys())
            items.append(f"Review fallback sections after rejected LLM output: {sections}.")
        if not items:
            items.append("Perform a final human pass for wording, author metadata, figures, and venue rules.")
        return list(dict.fromkeys(items))[:8]

    def _status(self, score: int, blocking_items: list[str]) -> str:
        if blocking_items:
            return "needs_evidence"
        if score >= 88:
            return "reviewable"
        if score >= 72:
            return "needs_author_pass"
        return "needs_evidence"

    def _clamp(self, value: int) -> int:
        return max(0, min(100, value))
