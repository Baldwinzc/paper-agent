"""LaTeX composer."""

from __future__ import annotations

import re
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
        output_path = output_root / "main.tex"
        output_path.write_text(rendered, encoding="utf-8")
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
