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
                lines.append(
                    "- "
                    f"[{candidate.get('category', 'unknown')}] "
                    f"`{candidate.get('key')}`: {candidate.get('title')} "
                    f"({candidate.get('year') or 'year unknown'}, "
                    f"cited by {candidate.get('cited_by_count', 0)})"
                )

        lines.extend(["", "## Bibliography Verification", ""])
        unverified = [
            entry
            for entry in bibliography
            if "verify" in entry.note.lower() or "seed" in entry.note.lower() or entry.year == "TODO"
        ]
        if unverified:
            for entry in unverified:
                status = "resolved" if entry.doi or (entry.year and entry.year != "TODO") else "seed"
                lines.append(f"- `{entry.key}` ({status}): {entry.title}")
                lines.append(f"  Note: {entry.note}")
        else:
            lines.append("- All bibliography entries appear resolved; still verify manually before submission.")

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

        lines.extend(
            [
                "",
                "## Submission Reminder",
                "",
                "- Replace synthetic or mock experiment results with real measured results.",
                "- Verify every bibliography entry against the actual paper.",
                "- Confirm the venue template and author instructions before submission.",
                "",
            ]
        )
        return "\n".join(lines)
