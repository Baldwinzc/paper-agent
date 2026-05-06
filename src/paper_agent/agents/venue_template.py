"""Venue template selection and retrieval."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from paper_agent.state import PaperRequest, PaperState, VenueTemplate


class VenueTemplateAgent:
    """Selects a venue family and prepares a LaTeX template directory."""

    TEMPLATE_REGISTRY = {
        "ieee": "https://template-selector.ieee.org/secure/templateSelector/downloadTemplate?publicationTypeId=1&titleId=1",
        "cvpr": "https://github.com/cvpr-org/author-kit/releases/latest/download/cvpr-author-kit.zip",
        "acl": "https://github.com/acl-org/acl-style-files/archive/refs/heads/master.zip",
        "neurips": "https://media.neurips.cc/Conferences/NeurIPS2025/Styles/neurips_2025.sty",
        "icml": "https://media.icml.cc/Conferences/ICML2025/Styles/icml2025.zip",
    }

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        family = self._classify(request.target_venue)
        template_root = Path(os.getenv("PAPER_AGENT_DATA_DIR", "data")) / "templates" / family
        template_root.mkdir(parents=True, exist_ok=True)
        source = "built-in"
        notes = []

        url = self.TEMPLATE_REGISTRY.get(family)
        if url:
            try:
                marker = template_root / "SOURCE_URL.txt"
                if not marker.exists():
                    response = httpx.get(url, timeout=20, follow_redirects=True)
                    response.raise_for_status()
                    suffix = ".zip" if "zip" in response.headers.get("content-type", "") or url.endswith(".zip") else ".sty"
                    (template_root / f"remote_template{suffix}").write_bytes(response.content)
                    marker.write_text(url, encoding="utf-8")
                source = url
                notes.append("Remote venue template artifact cached. Extraction will be added in Phase 2.")
            except Exception as exc:
                notes.append(f"Remote template fetch failed; using built-in fallback: {exc}")

        state["venue_template"] = VenueTemplate(
            venue=request.target_venue,
            family=family,
            template_source=source,
            template_dir=str(template_root),
            main_template="main.tex.j2",
            notes=notes,
        )
        return state

    def _classify(self, venue: str) -> str:
        lowered = venue.lower()
        if "ieee" in lowered:
            return "ieee"
        if "cvpr" in lowered or "iccv" in lowered or "eccv" in lowered:
            return "cvpr"
        if "acl" in lowered or "emnlp" in lowered or "naacl" in lowered:
            return "acl"
        if "neurips" in lowered:
            return "neurips"
        if "icml" in lowered:
            return "icml"
        if "iclr" in lowered:
            return "iclr"
        if "acm" in lowered:
            return "acm"
        if "springer" in lowered or "lncs" in lowered:
            return "springer"
        return "generic"

