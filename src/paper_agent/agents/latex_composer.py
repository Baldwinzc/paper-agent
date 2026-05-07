"""LaTeX composer."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from paper_agent.state import PaperState


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
        rendered = template.render(
            title=outline.title_candidates[0] if outline.title_candidates else request.project_name,
            venue=request.target_venue,
            abstract=self._latex_escape(sections.abstract),
            introduction=self._latex_escape(sections.introduction),
            related_work=self._latex_escape(sections.related_work),
            method=self._latex_escape(sections.method),
            experiments=self._latex_escape(sections.experiments),
            conclusion=self._latex_escape(sections.conclusion),
        )
        self._copy_template_assets(template_dir, output_root)
        self._write_project_helpers(output_root, request.target_venue, venue_template.template_source)

        output_path = output_root / "main.tex"
        output_path.write_text(rendered, encoding="utf-8")
        state["latex_project_dir"] = output_root
        state["latex_output_path"] = output_path
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

    def _write_project_helpers(self, output_root: Path, venue: str, template_source: str) -> None:
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
                    f"- Target venue: {venue}",
                    f"- Template source: {template_source}",
                    "- `references.bib` is a placeholder; add real BibTeX entries before submission.",
                    "- Generated text should be reviewed against the actual experiments and baseline paper.",
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
