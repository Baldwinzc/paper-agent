"""LaTeX composer."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from paper_agent.figures import (
    write_bar_chart_pdf,
    write_flow_diagram_pdf,
    write_prototype_hypergraph_pdf,
)
from paper_agent.state import PaperState, VenueTemplate
from paper_agent.tables import extract_markdown_tables, markdown_table_to_latex


class LatexComposerAgent:
    """Composes draft sections into a venue-compatible LaTeX project."""

    CORE_LATEX_PACKAGES = {
        "algorithm",
        "algorithmic",
        "amsfonts",
        "amsmath",
        "amssymb",
        "array",
        "babel",
        "balance",
        "booktabs",
        "calc",
        "caption",
        "cite",
        "color",
        "dblfloatfix",
        "enumitem",
        "epsfig",
        "float",
        "fontenc",
        "geometry",
        "graphicx",
        "hyperref",
        "ifthen",
        "inputenc",
        "mathtools",
        "microtype",
        "multirow",
        "natbib",
        "stfloats",
        "subfigure",
        "textcomp",
        "times",
        "url",
        "xcolor",
    }

    def run(self, state: PaperState) -> PaperState:
        request = state["request"]
        venue_template = state["venue_template"]
        sections = state["sections"]
        outline = state["outline"]

        output_root = Path("outputs") / self._slug(request.project_name)
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        template_dir = Path(__file__).resolve().parents[1] / "latex_templates" / venue_template.family
        if not template_dir.exists():
            template_dir = Path(__file__).resolve().parents[1] / "latex_templates" / "generic"

        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(disabled_extensions=("tex", "j2")),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("main.tex.j2")
        self._copy_template_assets(template_dir, output_root)
        self._copy_cached_template_assets(Path(venue_template.template_dir), output_root)
        experiment_tables = extract_markdown_tables(request.experiment_results)
        self._generate_presentation_figures(output_root, state)
        method_figures_latex = self._figure_latex_for_section(state, "Method")
        experiment_figures_latex = self._figure_latex_for_section(state, "Experiments")
        experiment_table_latex = "\n\n".join(
            markdown_table_to_latex(table) for table in experiment_tables
        )
        citation_keys = {entry.key for entry in state.get("bibliography", [])}
        citation_aliases = state.get("artifacts", {}).get("citation_key_aliases", {})
        undefined_citations: set[str] = set()
        experiments_latex = self._latex_escape(
            sections.experiments,
            citation_keys=citation_keys,
            citation_aliases=citation_aliases,
            undefined_citations=undefined_citations,
        )
        if experiment_table_latex:
            experiments_latex = experiments_latex + "\n\n" + experiment_table_latex
        if experiment_figures_latex:
            experiments_latex = experiments_latex + "\n\n" + experiment_figures_latex

        title = outline.title_candidates[0] if outline.title_candidates else request.project_name
        template_values = {
            "title": title,
            "venue": request.target_venue,
            "abstract": self._latex_escape(
                sections.abstract,
                citation_keys=citation_keys,
                citation_aliases=citation_aliases,
                undefined_citations=undefined_citations,
            ),
            "introduction": self._latex_escape(
                sections.introduction,
                citation_keys=citation_keys,
                citation_aliases=citation_aliases,
                undefined_citations=undefined_citations,
            ),
            "related_work": self._latex_escape(
                sections.related_work,
                citation_keys=citation_keys,
                citation_aliases=citation_aliases,
                undefined_citations=undefined_citations,
            ),
            "method": self._latex_escape(
                sections.method,
                citation_keys=citation_keys,
                citation_aliases=citation_aliases,
                undefined_citations=undefined_citations,
            )
            + (("\n\n" + method_figures_latex) if method_figures_latex else ""),
            "experiments": experiments_latex,
            "conclusion": self._latex_escape(
                sections.conclusion,
                citation_keys=citation_keys,
                citation_aliases=citation_aliases,
                undefined_citations=undefined_citations,
            ),
        }
        if venue_template.sample_main_tex:
            rendered = self._render_from_sample_main(
                Path(venue_template.sample_main_tex),
                template_values,
                output_root,
                state,
            )
        else:
            rendered = template.render(**template_values)
        self._write_project_helpers(output_root, venue_template, state)
        self._write_presentation_plan(output_root, state, experiment_tables)

        output_path = output_root / "main.tex"
        output_path.write_text(rendered, encoding="utf-8")
        state["latex_project_dir"] = output_root
        state["latex_output_path"] = output_path
        state.setdefault("artifacts", {})["latex_table_count"] = len(experiment_tables)
        state["artifacts"]["generated_figure_count"] = len(
            state.get("artifacts", {}).get("generated_figures", [])
        )
        state["artifacts"]["latex_tables"] = [
            {
                "label": table.label,
                "caption": table.caption,
                "columns": len(table.headers),
                "rows": len(table.rows),
            }
            for table in experiment_tables
        ]
        state["artifacts"]["undefined_citation_keys"] = sorted(undefined_citations)
        state["final_markdown"] = self._markdown(state)
        return state

    def _slug(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower()).strip("-") or "paper"

    def _latex_escape(
        self,
        text: str,
        citation_keys: set[str] | None = None,
        citation_aliases: dict[str, str] | None = None,
        undefined_citations: set[str] | None = None,
    ) -> str:
        converted_lines = []
        for line in text.splitlines():
            heading = re.match(r"^(#{3,6})\s+(.+?)\s*$", line)
            if heading:
                command = r"\subsection" if len(heading.group(1)) == 3 else r"\subsubsection"
                title = heading.group(2).strip()
                converted_lines.append(command + "{" + self._escape_inline(title) + "}")
            else:
                converted_lines.append(
                    self._escape_inline(
                        line,
                        citation_keys=citation_keys,
                        citation_aliases=citation_aliases,
                        undefined_citations=undefined_citations,
                    )
                )
        return "\n".join(converted_lines)

    def _escape_inline(
        self,
        text: str,
        citation_keys: set[str] | None = None,
        citation_aliases: dict[str, str] | None = None,
        undefined_citations: set[str] | None = None,
    ) -> str:
        if citation_keys:
            text = self._convert_known_citations(
                text,
                citation_keys,
                citation_aliases or {},
                undefined_citations=undefined_citations,
            )
        text = re.sub(r"\[PLACEHOLDER:\s*(.+?)\]", r"\\textbf{TODO:} \1", text, flags=re.I)
        text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
        protected_commands: dict[str, str] = {}

        def protect_command(match: re.Match[str]) -> str:
            token = f"PAPERAGENTCMD{len(protected_commands)}TOKEN"
            protected_commands[token] = match.group(0)
            return token

        text = re.sub(r"\\cite\{[^}]+\}", protect_command, text)
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = self._replace_unicode_math_symbols(text)
        for token, command in protected_commands.items():
            text = text.replace(token, command)
        return text

    def _replace_unicode_math_symbols(self, text: str) -> str:
        math_replacements = {
            "λ\\_rec": r"\(\lambda_{\mathrm{rec}}\)",
            "λ": r"\(\lambda\)",
        }
        for old, new in math_replacements.items():
            text = text.replace(old, new)
        return text

    def _convert_known_citations(
        self,
        text: str,
        citation_keys: set[str],
        citation_aliases: dict[str, str],
        undefined_citations: set[str] | None = None,
    ) -> str:
        def normalize(raw: str) -> tuple[list[str], list[str]]:
            keys = [citation_aliases.get(part.strip(), part.strip()) for part in raw.split(",")]
            deduped = list(dict.fromkeys(keys))
            missing = [key for key in deduped if key not in citation_keys]
            return deduped, missing

        def remember_missing(missing: list[str]) -> None:
            if undefined_citations is not None:
                undefined_citations.update(key for key in missing if key)

        def replace_markdown_cite(match: re.Match[str]) -> str:
            deduped, missing = normalize(match.group(1))
            if deduped and not missing:
                return r"\cite{" + ",".join(deduped) + "}"
            remember_missing(missing)
            return match.group(0)

        def replace_latex_cite(match: re.Match[str]) -> str:
            deduped, missing = normalize(match.group(1))
            remember_missing(missing)
            if deduped and not missing:
                return r"\cite{" + ",".join(deduped) + "}"
            return match.group(0)

        text = re.sub(r"\\cite\{([A-Za-z0-9_,\s-]+)\}", replace_latex_cite, text)
        return re.sub(r"\[([A-Za-z0-9,\s]+)\]", replace_markdown_cite, text)

    def _copy_template_assets(self, template_dir: Path, output_root: Path) -> None:
        for path in template_dir.iterdir():
            if path.name.endswith(".j2"):
                continue
            destination = output_root / path.name
            if path.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(path, destination)
            else:
                shutil.copy2(path, destination)

    def _copy_cached_template_assets(self, cached_template_dir: Path, output_root: Path) -> None:
        if not cached_template_dir.exists():
            return

        allowed_suffixes = {
            ".bbx",
            ".bib",
            ".bst",
            ".cbx",
            ".clo",
            ".cls",
            ".def",
            ".eps",
            ".jpg",
            ".jpeg",
            ".pdf",
            ".png",
            ".sty",
            ".svg",
            ".tex",
        }
        for path in sorted(cached_template_dir.rglob("*")):
            if not path.is_file() or path.name == "SOURCE_URL.txt":
                continue
            relative_parts = path.relative_to(cached_template_dir).parts
            if relative_parts and relative_parts[0] == "user":
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            if path.name.lower() == "main.tex":
                continue
            destination = output_root / path.name
            if destination.exists():
                continue
            shutil.copy2(path, destination)

    def _render_from_sample_main(
        self,
        sample_main: Path,
        values: dict[str, str],
        output_root: Path,
        state: PaperState,
    ) -> str:
        sample_text = sample_main.read_text(encoding="utf-8", errors="ignore")
        begin_match = re.search(r"\\begin\{document\}", sample_text)
        if not begin_match:
            return self._render_minimal_document(values)

        preamble = sample_text[: begin_match.start()].strip()
        preamble = self._replace_or_append_command(preamble, "title", values["title"])
        preamble = self._ensure_package(preamble, "amsmath")
        preamble = self._ensure_package(preamble, "graphicx")
        preamble = self._ensure_package(preamble, "booktabs")
        preamble, dropped_packages = self._drop_missing_template_packages(preamble, output_root)
        if dropped_packages:
            state.setdefault("artifacts", {})["dropped_missing_template_packages"] = dropped_packages

        return "\n\n".join(
            [
                preamble,
                r"\begin{document}",
                r"\maketitle",
                "",
                r"\begin{abstract}",
                values["abstract"],
                r"\end{abstract}",
                "",
                r"\section{Introduction}",
                values["introduction"],
                "",
                r"\section{Related Work}",
                values["related_work"],
                "",
                r"\section{Method}",
                values["method"],
                "",
                r"\section{Experiments}",
                values["experiments"],
                "",
                r"\section{Conclusion}",
                values["conclusion"],
                "",
                r"\bibliographystyle{IEEEtran}",
                r"\bibliography{references}",
                "",
                r"\end{document}",
            ]
        )

    def _render_minimal_document(self, values: dict[str, str]) -> str:
        return "\n\n".join(
            [
                r"\documentclass{article}",
                r"\usepackage{amsmath}",
                r"\usepackage{graphicx}",
                r"\usepackage{booktabs}",
                rf"\title{{{self._escape_inline(values['title'])}}}",
                r"\author{Anonymous Authors}",
                r"\begin{document}",
                r"\maketitle",
                r"\begin{abstract}",
                values["abstract"],
                r"\end{abstract}",
                r"\section{Introduction}",
                values["introduction"],
                r"\section{Related Work}",
                values["related_work"],
                r"\section{Method}",
                values["method"],
                r"\section{Experiments}",
                values["experiments"],
                r"\section{Conclusion}",
                values["conclusion"],
                r"\bibliographystyle{plain}",
                r"\bibliography{references}",
                r"\end{document}",
            ]
        )

    def _replace_or_append_command(self, preamble: str, command: str, value: str) -> str:
        escaped_value = self._escape_inline(value)
        pattern = re.compile(rf"\\{command}(?:\[[^\]]*\])?\{{.*?\}}", re.S)
        replacement = rf"\{command}{{{escaped_value}}}"
        if pattern.search(preamble):
            return pattern.sub(lambda _: replacement, preamble, count=1)
        return preamble + "\n" + replacement

    def _ensure_package(self, preamble: str, package: str) -> str:
        if re.search(rf"\\usepackage(?:\[[^\]]*\])?\{{[^}}]*\b{re.escape(package)}\b[^}}]*\}}", preamble):
            return preamble
        return preamble + "\n" + rf"\usepackage{{{package}}}"

    def _drop_missing_template_packages(self, preamble: str, output_root: Path) -> tuple[str, list[str]]:
        dropped: list[str] = []

        def replace(match: re.Match[str]) -> str:
            options = match.group(1) or ""
            packages = [item.strip() for item in match.group(2).split(",") if item.strip()]
            keep = []
            for package in packages:
                if self._package_available(package, output_root):
                    keep.append(package)
                else:
                    dropped.append(package)
            if not keep:
                return ""
            if keep == packages:
                return match.group(0)
            return rf"\usepackage{options}{{{','.join(keep)}}}"

        cleaned = re.sub(
            r"\\usepackage(\[[^\]]*\])?\{([^}]+)\}",
            replace,
            preamble,
        )
        return cleaned, list(dict.fromkeys(dropped))

    def _package_available(self, package: str, output_root: Path) -> bool:
        if package in self.CORE_LATEX_PACKAGES:
            return True
        package_path = Path(package)
        candidates = [
            output_root / f"{package}.sty",
            output_root / f"{package_path.name}.sty",
        ]
        return any(candidate.is_file() for candidate in candidates)

    def _write_project_helpers(
        self,
        output_root: Path,
        venue_template: VenueTemplate,
        state: PaperState,
    ) -> None:
        references = output_root / "references.bib"
        references.write_text(self._bibtex(state), encoding="utf-8")

        readme = output_root / "README_OVERLEAF.md"
        readme.write_text(
            "\n".join(
                [
                    "# Overleaf Upload Notes",
                    "",
                    "Upload the generated zip file to Overleaf with New Project > Upload Project.",
                    "Set `main.tex` as the main document if Overleaf does not detect it automatically.",
                    "",
                    f"- Target venue: {venue_template.venue}",
                    f"- Template: {venue_template.template_name or venue_template.family}",
                    f"- Template source: {venue_template.template_source}",
                    f"- Overleaf template page: {venue_template.overleaf_url or 'not configured'}",
                    "- `references.bib` is a placeholder; add real BibTeX entries before submission.",
                    "- Generated text should be reviewed against the actual experiments and baseline paper.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        source_notes = output_root / "TEMPLATE_SOURCE.md"
        source_notes.write_text(
            "\n".join(
                [
                    "# Template Source",
                    "",
                    f"- Venue: {venue_template.venue}",
                    f"- Family: {venue_template.family}",
                    f"- Template: {venue_template.template_name or venue_template.family}",
                    f"- Source: {venue_template.template_source}",
                    f"- Overleaf: {venue_template.overleaf_url or 'not configured'}",
                    f"- Sample main: {venue_template.sample_main_tex or 'not detected'}",
                    "",
                    "## Notes",
                    "",
                    *[f"- {note}" for note in venue_template.notes],
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _write_presentation_plan(self, output_root: Path, state: PaperState, experiment_tables) -> None:
        plan = state.setdefault("artifacts", {}).get("presentation_plan", {})
        figures = plan.get("figures", []) if isinstance(plan, dict) else []
        planned_tables = plan.get("tables", []) if isinstance(plan, dict) else []
        rendered_tables = [
            {
                "label": table.label,
                "caption": table.caption,
                "columns": len(table.headers),
                "rows": len(table.rows),
                "status": "rendered_from_markdown",
            }
            for table in experiment_tables
        ]
        table_by_label = {
            str(item.get("label")): item
            for item in [*planned_tables, *rendered_tables]
            if item.get("label")
        }
        lines = [
            "# Figure and Table Plan",
            "",
            "This file records planned presentation assets for author review. Planned figures are not inserted into `main.tex` until the asset exists.",
            "",
            "## Planned Figures",
            "",
        ]
        if figures:
            for figure in figures:
                lines.extend(
                    [
                        f"- `{figure.get('label')}`: {figure.get('title')}",
                        f"  - Section: {figure.get('section')}",
                        f"  - Asset: `{figure.get('asset_path')}`",
                        f"  - Status: {figure.get('status', 'planned')}",
                        f"  - Caption: {figure.get('caption')}",
                    ]
                )
                evidence = figure.get("evidence", [])
                if evidence:
                    lines.append("  - Evidence:")
                    lines.extend(f"    - {item}" for item in evidence[:3])
        else:
            lines.append("- No planned figures.")
        lines.extend(["", "## Tables", ""])
        if table_by_label:
            for table in table_by_label.values():
                lines.append(
                    f"- `{table.get('label')}`: {table.get('caption')} "
                    f"({table.get('rows', 0)} rows, {table.get('columns', 0)} columns; "
                    f"{table.get('status', 'planned')})"
                )
        else:
            lines.append("- No planned or rendered tables.")
        open_items = plan.get("open_items", []) if isinstance(plan, dict) else []
        if open_items:
            lines.extend(["", "## Open Items", ""])
            lines.extend(f"- {item}" for item in open_items)
        path = output_root / "FIGURE_TABLE_PLAN.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        state.setdefault("artifacts", {})["presentation_plan_path"] = str(path)

    def _generate_presentation_figures(self, output_root: Path, state: PaperState) -> None:
        plan = state.setdefault("artifacts", {}).get("presentation_plan", {})
        if not isinstance(plan, dict):
            return
        figures = plan.get("figures", [])
        if not figures:
            return
        experiments = state.get("experiments")

        generated: list[dict[str, str]] = []
        for figure in figures:
            label = figure.get("label")
            asset_path = str(figure.get("asset_path") or "")
            if not asset_path:
                continue
            output_path = output_root / asset_path

            if label == "fig:method-overview":
                steps = self._method_overview_steps(state)
                if len(steps) < 2:
                    continue
                write_flow_diagram_pdf(
                    output_path,
                    title=str(figure.get("title") or label),
                    steps=steps,
                    footer=self._method_overview_footer(state),
                )
            elif label == "fig:prototype-hypergraph":
                if not self._has_prototype_hypergraph_evidence(state):
                    continue
                write_prototype_hypergraph_pdf(
                    output_path,
                    title=str(figure.get("title") or label),
                    notes=self._prototype_hypergraph_notes(state),
                )
            elif label == "fig:main-results" and experiments:
                bars = self._main_result_bars(experiments)
                if not bars:
                    continue
                write_bar_chart_pdf(
                    output_path,
                    title=str(figure.get("title") or label),
                    bars=bars,
                    y_label="Signed improvement",
                )
            elif label == "fig:ablation-summary" and experiments:
                bars = self._ablation_bars(experiments)
                if not bars:
                    continue
                write_bar_chart_pdf(
                    output_path,
                    title=str(figure.get("title") or label),
                    bars=bars,
                    y_label="Signed drop",
                )
            else:
                continue

            figure["status"] = "generated"
            figure["asset_exists"] = True
            figure["generated_path"] = str(output_path)
            generated.append(
                {
                    "label": str(label),
                    "asset_path": asset_path,
                    "path": str(output_path),
                }
            )

        state.setdefault("artifacts", {})["generated_figures"] = generated
        plan["open_items"] = [
            item
            for item in plan.get("open_items", [])
            if not any(generated_item["asset_path"] in item for generated_item in generated)
        ]

    def _method_overview_steps(self, state: PaperState) -> list[str]:
        innovations = state.get("innovations", [])
        if not innovations:
            return []
        steps = ["WSI patch features"]
        for innovation in innovations[:3]:
            label = re.sub(r"^Innovation\s+\d+:\s*", "", innovation.name, flags=re.I)
            steps.append(label or innovation.technical_idea)
        steps.append("Survival risk prediction")
        return list(dict.fromkeys(self._figure_step_text(step) for step in steps if step))

    def _method_overview_footer(self, state: PaperState) -> str:
        innovations = state.get("innovations", [])
        evidence = [
            item
            for innovation in innovations
            for item in innovation.evidence[:1]
        ]
        return "Evidence: " + "; ".join(evidence[:2]) if evidence else ""

    def _has_prototype_hypergraph_evidence(self, state: PaperState) -> bool:
        text_parts: list[str] = []
        for innovation in state.get("innovations", []):
            text_parts.extend([innovation.name, innovation.technical_idea, *innovation.evidence])
        code = state.get("code")
        if code:
            text_parts.extend(code.likely_method_files)
            text_parts.extend(code.implementation_evidence)
            text_parts.extend(code.method_claims)
        comparison = state.get("artifacts", {}).get("code_baseline_comparison", {})
        if isinstance(comparison, dict):
            text_parts.extend(str(term) for term in comparison.get("code_only_terms", []))
            text_parts.extend(str(seed) for seed in comparison.get("innovation_seeds", []))
        lowered = " ".join(text_parts).lower()
        return any(
            token in lowered
            for token in ["hypergraph", "hyperedge", "optimal transport", "wasserstein", "barycenter"]
        )

    def _prototype_hypergraph_notes(self, state: PaperState) -> list[str]:
        notes = []
        for innovation in state.get("innovations", []):
            if self._contains_any(
                f"{innovation.name} {innovation.technical_idea}",
                ["hypergraph", "hyperedge", "transport", "wasserstein", "barycenter"],
            ):
                notes.append(re.sub(r"^Innovation\s+\d+:\s*", "", innovation.name, flags=re.I))
        code = state.get("code")
        if code:
            notes.extend(
                evidence
                for evidence in code.implementation_evidence[:2]
                if self._contains_any(evidence, ["hypergraph", "hyperedge", "transport", "wasserstein"])
            )
        return notes[:3]

    def _figure_step_text(self, text: str) -> str:
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip(" .")
        return text[:70].rstrip()

    def _contains_any(self, text: str, needles: list[str]) -> bool:
        lowered = text.lower()
        return any(needle in lowered for needle in needles)

    def _main_result_bars(self, experiments) -> list[tuple[str, float]]:
        bars: list[tuple[str, float]] = []
        for table in experiments.result_tables:
            for comparison in table.comparisons:
                label = " ".join(
                    part
                    for part in [comparison.dataset, comparison.metric]
                    if part
                ) or comparison.table_caption or "Result"
                bars.append((label, comparison.signed_improvement))
        return bars[:10]

    def _ablation_bars(self, experiments) -> list[tuple[str, float]]:
        return [
            (item.variant, item.signed_drop)
            for item in experiments.ablation_evidence[:10]
        ]

    def _figure_latex_for_section(self, state: PaperState, section: str) -> str:
        plan = state.get("artifacts", {}).get("presentation_plan", {})
        if not isinstance(plan, dict):
            return ""
        snippets = []
        for figure in plan.get("figures", []):
            if figure.get("section") != section or figure.get("status") != "generated":
                continue
            asset_path = str(figure.get("asset_path") or "")
            if not asset_path:
                continue
            snippets.append(
                "\n".join(
                    [
                        r"\begin{figure}[t]",
                        r"\centering",
                        rf"\includegraphics[width=\columnwidth]{{{asset_path}}}",
                        rf"\caption{{{self._escape_inline(str(figure.get('caption') or 'Generated figure.'))}}}",
                        rf"\label{{{figure.get('label')}}}",
                        r"\end{figure}",
                    ]
                )
            )
        return "\n\n".join(snippets)

    def _markdown(self, state: PaperState) -> str:
        sections = state["sections"]
        outline = state["outline"]
        title = outline.title_candidates[0] if outline.title_candidates else state["request"].project_name
        references = "\n".join(
            f"- `{entry.key}`: {entry.title}" for entry in state.get("bibliography", [])
        )
        presentation = self._presentation_markdown(state)
        return (
            f"# {title}\n\n"
            f"## Abstract\n\n{sections.abstract}\n\n"
            f"## Introduction\n\n{sections.introduction}\n\n"
            f"## Related Work\n\n{sections.related_work}\n\n"
            f"## Method\n\n{sections.method}\n\n"
            f"## Experiments\n\n{sections.experiments}\n\n"
            f"## Conclusion\n\n{sections.conclusion}\n\n"
            f"{presentation}"
            f"## Reference Seeds\n\n{references or '- To be completed.'}\n"
        )

    def _presentation_markdown(self, state: PaperState) -> str:
        plan = state.get("artifacts", {}).get("presentation_plan", {})
        if not isinstance(plan, dict):
            return ""
        figures = plan.get("figures", [])
        tables = plan.get("tables", [])
        if not figures and not tables:
            return ""
        lines = ["## Figure and Table Plan", ""]
        if figures:
            lines.append("Planned figures:")
            for figure in figures[:6]:
                lines.append(
                    f"- `{figure.get('label')}` ({figure.get('section')}): "
                    f"{figure.get('caption')}"
                )
        if tables:
            lines.append("")
            lines.append("Planned/rendered tables:")
            for table in tables[:6]:
                lines.append(
                    f"- `{table.get('label')}`: {table.get('caption')}"
                )
        return "\n".join(lines) + "\n\n"

    def _bibtex(self, state: PaperState) -> str:
        entries = state.get("bibliography", [])
        if not entries:
            return "% Add baseline and related-work BibTeX entries here.\n"
        return "\n\n".join(self._bibtex_entry(entry) for entry in entries) + "\n"

    def _bibtex_entry(self, entry) -> str:
        fields = {
            "title": entry.title,
            "author": " and ".join(entry.authors) if entry.authors else "To be completed",
            "year": entry.year or "TODO",
            "journal": entry.venue,
            "doi": entry.doi,
            "url": entry.url,
            "note": entry.note or "Verify metadata before submission.",
        }
        lines = [f"@misc{{{entry.key},"]
        for name, value in fields.items():
            if value:
                lines.append(f"  {name} = {{{self._bibtex_escape(value)}}},")
        lines.append("}")
        return "\n".join(lines)

    def _bibtex_escape(self, text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "{": r"\{",
            "}": r"\}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
        }
        escaped = text
        for old, new in replacements.items():
            escaped = escaped.replace(old, new)
        return escaped
