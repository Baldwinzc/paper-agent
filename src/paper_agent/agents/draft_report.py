"""Draft quality report writer."""

from __future__ import annotations

from paper_agent.state import PaperState


class DraftReportAgent:
    """Writes a human-readable report beside the generated LaTeX project."""

    def run(self, state: PaperState) -> PaperState:
        project_dir = state.get("latex_project_dir")
        if not project_dir:
            return state
        report_path = project_dir / "DRAFT_REPORT.md"
        report_path.write_text(self._report(state), encoding="utf-8")
        state.setdefault("artifacts", {})["draft_report_path"] = str(report_path)
        return state

    def _report(self, state: PaperState) -> str:
        request = state["request"]
        artifacts = state.get("artifacts", {})
        review_findings = state.get("review_findings", [])
        baseline = state.get("baseline")
        experiments = state.get("experiments")
        bibliography = state.get("bibliography", [])

        lines = [
            "# Draft Report",
            "",
            f"- Project: {request.project_name}",
            f"- Target venue: {request.target_venue}",
            f"- Section writer mode: {artifacts.get('section_writer_mode', 'unknown')}",
            f"- Template source: {state.get('venue_template').template_source if state.get('venue_template') else 'unknown'}",
            f"- LaTeX tables: {artifacts.get('latex_table_count', 0)}",
            f"- Bibliography entries: {len(bibliography)}",
            f"- Reference resolver: {artifacts.get('reference_resolver_mode', 'not run')}",
            "",
            "## Review Findings",
            "",
        ]
        if review_findings:
            for finding in review_findings:
                lines.append(f"- [{finding.severity}] {finding.issue} Suggestion: {finding.suggestion}")
        else:
            lines.append("- No reviewer findings recorded.")

        if baseline:
            lines.extend(["", "## Baseline Evidence", ""])
            lines.append(f"- Title: {baseline.title or 'not detected'}")
            lines.append(f"- Problem: {self._clip(baseline.problem)}")
            lines.append(f"- Method: {self._clip(baseline.method)}")
            lines.append(f"- Experiments: {self._clip(baseline.experiments)}")
            if baseline.limitations:
                lines.append("- Limitations:")
                for limitation in baseline.limitations[:3]:
                    lines.append(f"  - {self._clip(limitation)}")
            if baseline.structured_sections:
                section_names = ", ".join(baseline.structured_sections.keys())
                lines.append(f"- Structured sections: {section_names}")

        lines.extend(["", "## Missing Experiment Details", ""])
        if experiments and experiments.missing_details:
            lines.extend(f"- {item}" for item in experiments.missing_details)
        else:
            lines.append("- No missing experiment details detected by the analyzer.")

        traceability = artifacts.get("innovation_traceability", [])
        if traceability:
            lines.extend(["", "## Innovation Traceability", ""])
            for item in traceability:
                status = "covered" if item.get("mentioned_in_method") else "missing from Method"
                lines.append(f"- `{item.get('name')}`: {status}; evidence items: {item.get('evidence_count', 0)}")
                for evidence in item.get("evidence_preview", [])[:2]:
                    lines.append(f"  Evidence: {evidence}")

        related_candidates = artifacts.get("related_work_candidates", [])
        if related_candidates:
            lines.extend(["", "## Related Work Discovery", ""])
            for candidate in related_candidates:
                query = candidate.get("query")
                query_text = f"; query: {query}" if query and query != candidate.get("title") else ""
                lines.append(
                    "- "
                    f"[{candidate.get('category', 'unknown')}] "
                    f"`{candidate.get('key')}`: {candidate.get('title')} "
                    f"({candidate.get('year') or 'year unknown'}, "
                    f"cited by {candidate.get('cited_by_count', 0)})"
                    f"{query_text}"
                )

        citation_coverage = artifacts.get("related_work_citation_coverage", [])
        if citation_coverage:
            lines.extend(["", "## Related Work Citation Coverage", ""])
            for item in citation_coverage:
                if not item.get("requires_citation", True):
                    status = "not required"
                elif item.get("covered_by_real_citation"):
                    status = "covered"
                else:
                    status = "missing real citation"
                citations = ", ".join(item.get("real_citation_keys") or item.get("citation_keys") or [])
                lines.append(f"- `{item.get('thread')}`: {status}; citations: {citations or 'none'}")

        consistency = artifacts.get("factual_consistency", [])
        if consistency:
            lines.extend(["", "## Factual Consistency", ""])
            for item in consistency:
                values = ", ".join(item.get("values", []))
                detail = f"; values: {values}" if values else ""
                lines.append(f"- `{item.get('check')}`: {item.get('status')}{detail}")

        lines.extend(["", "## Bibliography Verification", ""])
        verification = artifacts.get("reference_verification", {})
        if verification:
            lines.extend(
                [
                    f"- Auto-resolved entries: {verification.get('resolved_count', 0)}",
                    f"- Unresolved seed entries: {verification.get('unresolved_count', 0)}",
                    "- Auto-resolved metadata still requires manual relevance checking.",
                    "",
                ]
            )

        unresolved_keys = set(verification.get("unresolved_seed_keys", [])) if verification else set()
        resolved_keys = set(verification.get("resolved_keys", [])) if verification else set()
        unresolved = [
            entry
            for entry in bibliography
            if entry.key in unresolved_keys
            or (
                not verification
                and ("seed" in entry.note.lower() or "placeholder" in entry.note.lower())
            )
        ]
        if unresolved:
            lines.append("Unresolved entries:")
            for entry in unresolved:
                lines.append(f"- `{entry.key}` (seed): {entry.title}")
                lines.append(f"  Note: {entry.note}")
        else:
            lines.append("- No unresolved seed entries detected.")

        resolved = [entry for entry in bibliography if entry.key in resolved_keys]
        if resolved:
            lines.extend(["", "Auto-resolved entries:"])
            for entry in resolved:
                venue = f", {entry.venue}" if entry.venue else ""
                year = f", {entry.year}" if entry.year else ""
                lines.append(f"- `{entry.key}`: {entry.title}{year}{venue}")

        aliases = artifacts.get("citation_key_aliases", {})
        if aliases:
            lines.extend(["", "## Citation Aliases", ""])
            for old_key, new_key in aliases.items():
                lines.append(f"- `{old_key}` merged into `{new_key}`")

        undefined_citations = artifacts.get("undefined_citation_keys", [])
        if undefined_citations:
            lines.extend(["", "## Undefined Citations", ""])
            for key in undefined_citations:
                lines.append(f"- `{key}`")

        section_errors = artifacts.get("section_writer_section_errors", {})
        if section_errors:
            lines.extend(["", "## Section Writer Errors", ""])
            for section, error in section_errors.items():
                lines.append(f"- {section}: {error}")

        llm_review = artifacts.get("llm_self_review", {})
        if llm_review:
            lines.extend(["", "## LLM Self Review", ""])
            lines.append(f"- Mode: {llm_review.get('mode', 'unknown')}")
            if llm_review.get("error"):
                lines.append(f"- Error: {llm_review.get('error')}")
            claims = llm_review.get("unsupported_claims", [])
            if claims:
                for claim in claims:
                    lines.append(
                        "- "
                        f"[{claim.get('severity', 'major')}] "
                        f"{claim.get('section', 'unknown')}: {claim.get('claim', '')}"
                    )
                    if claim.get("reason"):
                        lines.append(f"  Reason: {claim.get('reason')}")
                    if claim.get("evidence_needed"):
                        lines.append(f"  Evidence needed: {claim.get('evidence_needed')}")
            elif llm_review.get("mode") == "llm":
                lines.append("- No unsupported claims returned by LLM self-review.")
            notes = llm_review.get("section_quality_notes", [])
            for note in notes:
                lines.append(f"- Note: {note}")

        lines.extend(
            [
                "",
                "## Submission Reminder",
                "",
                "- Add real trained-model performance tables before making empirical comparison claims.",
                "- Verify every bibliography entry against the actual paper.",
                "- Confirm the venue template and author instructions before submission.",
                "",
            ]
        )
        return "\n".join(lines)

    def _clip(self, text: str, limit: int = 280) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact or "not detected"
        return compact[: limit - 3].rstrip() + "..."
