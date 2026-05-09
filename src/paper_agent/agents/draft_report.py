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
            f"- LLM-written sections: {len(artifacts.get('section_writer_llm_successes', []))}",
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

        readiness = artifacts.get("submission_readiness", {})
        if readiness:
            lines.extend(["", "## Submission Readiness", ""])
            lines.append(f"- Status: {readiness.get('status', 'unknown')}")
            lines.append(f"- Overall score: {readiness.get('overall_score', 0)}/100")
            scores = readiness.get("scores", {})
            for name, value in scores.items():
                lines.append(f"- {name.replace('_', ' ').title()}: {value}/100")
            blocking_items = readiness.get("blocking_items", [])
            if blocking_items:
                lines.append("- Blocking items:")
                for item in blocking_items[:5]:
                    lines.append(f"  - {item}")
            action_items = readiness.get("action_items", [])
            if action_items:
                lines.append("- Next actions:")
                for item in action_items[:5]:
                    lines.append(f"  - {item}")

        package = artifacts.get("submission_package", {})
        if package:
            checks = package.get("checks", {})
            zip_check = checks.get("zip", {}) if isinstance(checks, dict) else {}
            compile_check = checks.get("compile", {}) if isinstance(checks, dict) else {}
            lines.extend(["", "## Submission Package", ""])
            lines.append(f"- Status: {package.get('status', 'unknown')}")
            lines.append(f"- Errors: {len(package.get('errors', []))}")
            lines.append(f"- Warnings: {len(package.get('warnings', []))}")
            if isinstance(zip_check, dict):
                lines.append(
                    f"- Zip: {'present' if zip_check.get('present') else 'not generated'}"
                    f"; entries: {zip_check.get('entries', 0)}"
                )
            if isinstance(compile_check, dict):
                lines.append(
                    f"- Compile check: {compile_check.get('status', 'unknown')}"
                    f" ({compile_check.get('tool') or 'no tool'})"
                )
            for item in package.get("errors", [])[:5]:
                lines.append(f"- Error: {item}")
            for item in package.get("warnings", [])[:5]:
                lines.append(f"- Warning: {item}")

        presentation = artifacts.get("presentation_plan", {})
        if presentation:
            figures = presentation.get("figures", [])
            tables = presentation.get("tables", [])
            open_items = presentation.get("open_items", [])
            generated_figures = [
                figure for figure in figures if figure.get("status") == "generated"
            ]
            lines.extend(["", "## Figure and Table Plan", ""])
            lines.append(f"- Planned figures: {len(figures)}")
            lines.append(f"- Generated figures: {len(generated_figures)}")
            lines.append(f"- Planned/rendered tables: {len(tables)}")
            plan_path = artifacts.get("presentation_plan_path", "")
            if plan_path:
                lines.append(f"- Plan file: {plan_path}")
            for figure in figures[:5]:
                lines.append(
                    f"- Figure `{figure.get('label')}` ({figure.get('section')}): "
                    f"{self._clip(figure.get('caption', ''))}"
                )
            for table in tables[:5]:
                lines.append(
                    f"- Table `{table.get('label')}`: {self._clip(table.get('caption', ''))}"
                )
            if open_items:
                lines.append("- Open presentation items:")
                for item in open_items[:5]:
                    lines.append(f"  - {item}")

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

        comparison = artifacts.get("code_baseline_comparison", {})
        if comparison:
            lines.extend(["", "## Code-Baseline Comparison", ""])
            lines.append(f"- Mode: {comparison.get('mode', 'unknown')}")
            overlap = ", ".join(comparison.get("overlapping_terms", [])[:6])
            code_only = ", ".join(comparison.get("code_only_terms", [])[:6])
            if overlap:
                lines.append(f"- Shared technical context: {overlap}")
            if code_only:
                lines.append(f"- Code-side innovation candidates: {code_only}")
            shifts = comparison.get("likely_method_shifts", [])
            if shifts:
                lines.append("- Method-shift evidence:")
                for item in shifts[:5]:
                    lines.append(f"  - {item.get('technique')}: {self._clip(item.get('rationale', ''))}")
                    for evidence in item.get("evidence", [])[:2]:
                        lines.append(f"    Evidence: {self._clip(evidence)}")
            seeds = comparison.get("innovation_seeds", [])
            if seeds:
                lines.append("- Innovation seeds:")
                for seed in seeds[:5]:
                    lines.append(f"  - {self._clip(seed)}")

        lines.extend(["", "## Missing Experiment Details", ""])
        if experiments and experiments.missing_details:
            lines.extend(f"- {item}" for item in experiments.missing_details)
        else:
            lines.append("- No missing experiment details detected by the analyzer.")

        if experiments and experiments.result_tables:
            lines.extend(["", "## Parsed Experiment Results", ""])
            for table in experiments.result_tables:
                lines.append(
                    f"- {table.caption}: {table.method} vs {table.baseline}; "
                    f"comparisons: {len(table.comparisons)}"
                )
                for comparison in table.comparisons[:5]:
                    metric = f" {comparison.metric}" if comparison.metric else ""
                    lines.append(
                        f"  - {comparison.dataset or 'reported column'}{metric}: "
                        f"{comparison.method_value:.3f} vs {comparison.baseline_value:.3f} "
                        f"(signed improvement {comparison.signed_improvement:+.3f})"
                    )

        if experiments and experiments.ablation_evidence:
            lines.extend(["", "## Ablation Evidence", ""])
            for item in experiments.ablation_evidence[:8]:
                metric = f" {item.metric}" if item.metric else ""
                support = f"; supports: {', '.join(item.supports)}" if item.supports else ""
                lines.append(
                    f"- {item.variant}: {item.reference_value:.3f} -> "
                    f"{item.variant_value:.3f} on {item.dataset or 'reported column'}{metric} "
                    f"(signed drop {item.signed_drop:+.3f}){support}"
                )

        traceability = artifacts.get("innovation_traceability", [])
        if traceability:
            lines.extend(["", "## Innovation Traceability", ""])
            for item in traceability:
                status = "covered" if item.get("mentioned_in_method") else "missing from Method"
                lines.append(f"- `{item.get('name')}`: {status}; evidence items: {item.get('evidence_count', 0)}")
                for evidence in item.get("evidence_preview", [])[:2]:
                    lines.append(f"  Evidence: {evidence}")
                for evidence in item.get("ablation_evidence_preview", [])[:2]:
                    lines.append(f"  Ablation: {evidence}")

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

        resolution_trace = artifacts.get("reference_resolution_trace", [])
        if resolution_trace:
            lines.extend(["", "Reference resolution trace:"])
            for item in resolution_trace[:10]:
                source = item.get("source", "unknown")
                status = item.get("status", "unknown")
                retained = "" if item.get("retained", True) else f"; merged into `{item.get('retained_key')}`"
                doi = f"; doi: {item.get('doi')}" if item.get("doi") else ""
                lines.append(
                    f"- `{item.get('key')}`: {status} via {source}{retained}{doi}"
                )
                lines.append(f"  Query: {item.get('query')}")
                lines.append(f"  Match: {item.get('resolved_title')}")

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
        section_successes = artifacts.get("section_writer_llm_successes", [])
        repaired_sections = artifacts.get("section_writer_repaired_sections", [])
        if section_successes:
            lines.extend(["", "## LLM Section Drafting", ""])
            lines.append(f"- Successful sections: {', '.join(section_successes)}")
            if repaired_sections:
                lines.append(f"- Repaired sections: {', '.join(repaired_sections)}")
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
