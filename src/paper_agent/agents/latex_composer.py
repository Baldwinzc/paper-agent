"""LaTeX composer."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from paper_agent.state import PaperState, VenueTemplate
from paper_agent.tables import extract_markdown_tables, markdown_table_to_latex


class LatexComposerAgent:
    """Composes draft sections into a venue-compatible LaTeX project."""

    def run(self, state: PaperState) -> PaperState:
        request = state["request"]
        venue_template = state["venue_template"]
        sections = state["sections"]
        outline = state["outline"]

        output_root = Path("outputs") / self._slug(request.project_name)
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
        experiment_tables = extract_markdown_tables(request.experiment_results)
        experiment_table_latex = "\n\n".join(
            markdown_table_to_latex(table) for table in experiment_tables
        )
        experiments_latex = self._latex_escape(sections.experiments)
        if experiment_table_latex:
            experiments_latex = experiments_latex + "\n\n" + experiment_table_latex

        title = outline.title_candidates[0] if outline.title_candidates else request.project_name
        template_values = {
            "title": title,
            "venue": request.target_venue,
            "abstract": self._latex_escape(sections.abstract),
            "introduction": self._latex_escape(sections.introduction),
            "related_work": self._latex_escape(sections.related_work),
            "method": self._latex_escape(sections.method),
            "experiments": experiments_latex,
            "conclusion": self._latex_escape(sections.conclusion),
        }
        if venue_template.sample_main_tex:
            rendered = self._render_from_sample_main(Path(venue_template.sample_main_tex), template_values)
        else:
            rendered = template.render(**template_values)
        self._copy_template_assets(template_dir, output_root)
        self._copy_cached_template_assets(Path(venue_template.template_dir), output_root)
        self._write_project_helpers(output_root, venue_template)

        output_path = output_root / "main.tex"
        output_path.write_text(rendered, encoding="utf-8")
        state["latex_project_dir"] = output_root
        state["latex_output_path"] = output_path
        state.setdefault("artifacts", {})["latex_table_count"] = len(experiment_tables)
        state["final_markdown"] = self._markdown(state)
        return state

    def _slug(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower()).strip("-") or "paper"

    def _latex_escape(self, text: str) -> str:
        converted_lines = []
        for line in text.splitlines():
            if line.startswith("### "):
                title = line[4:].strip()
                converted_lines.append(r"\subsection{" + self._escape_inline(title) + "}")
            else:
                converted_lines.append(self._escape_inline(line))
        return "\n".join(converted_lines)

    def _escape_inline(self, text: str) -> str:
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

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
            if path.suffix.lower() not in allowed_suffixes:
                continue
            if path.name.lower() == "main.tex":
                continue
            destination = output_root / path.name
            if destination.exists():
                continue
            shutil.copy2(path, destination)

    def _render_from_sample_main(self, sample_main: Path, values: dict[str, str]) -> str:
        sample_text = sample_main.read_text(encoding="utf-8", errors="ignore")
        begin_match = re.search(r"\\begin\{document\}", sample_text)
        if not begin_match:
            return self._render_minimal_document(values)

        preamble = sample_text[: begin_match.start()].strip()
        preamble = self._replace_or_append_command(preamble, "title", values["title"])
        preamble = self._ensure_package(preamble, "amsmath")
        preamble = self._ensure_package(preamble, "graphicx")
        preamble = self._ensure_package(preamble, "booktabs")

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

    def _write_project_helpers(self, output_root: Path, venue_template: VenueTemplate) -> None:
        references = output_root / "references.bib"
        if not references.exists():
            references.write_text(
                "% Add baseline and related-work BibTeX entries here.\n",
                encoding="utf-8",
            )

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

    def _markdown(self, state: PaperState) -> str:
        sections = state["sections"]
        outline = state["outline"]
        title = outline.title_candidates[0] if outline.title_candidates else state["request"].project_name
        return (
            f"# {title}\n\n"
            f"## Abstract\n\n{sections.abstract}\n\n"
            f"## Introduction\n\n{sections.introduction}\n\n"
            f"## Related Work\n\n{sections.related_work}\n\n"
            f"## Method\n\n{sections.method}\n\n"
            f"## Experiments\n\n{sections.experiments}\n\n"
            f"## Conclusion\n\n{sections.conclusion}\n"
        )
