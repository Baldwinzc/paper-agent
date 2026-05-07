"""Venue template selection and retrieval."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import httpx

from paper_agent.state import PaperRequest, PaperState, VenueTemplate


@dataclass(frozen=True)
class TemplateSpec:
    family: str
    name: str
    remote_url: str = ""
    overleaf_url: str = ""
    notes: list[str] = field(default_factory=list)


class VenueTemplateAgent:
    """Selects a venue family and prepares a LaTeX template directory."""

    TEMPLATE_REGISTRY = {
        "generic": TemplateSpec(family="generic", name="Built-in generic article"),
        "ieee": TemplateSpec(
            family="ieee",
            name="IEEE conference template",
            remote_url="https://template-selector.ieee.org/secure/templateSelector/downloadTemplate?publicationTypeId=1&titleId=1",
            overleaf_url="https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn",
            notes=[
                "IEEE Author Center points conference authors to IEEE Word/LaTeX templates and Overleaf.",
            ],
        ),
        "ieee_journal": TemplateSpec(
            family="ieee_journal",
            name="IEEE journal paper template",
            remote_url="https://template-selector.ieee.org/secure/templateSelector/downloadTemplate?publicationTypeId=2&titleId=1002148",
            overleaf_url="https://www.overleaf.com/org/ieee",
            notes=[
                "For TPAMI, use an IEEE Computer Society journal-compatible IEEEtran layout.",
                "Overleaf lists official IEEE journal and Computer Society journal templates.",
            ],
        ),
        "cvpr": TemplateSpec(
            family="cvpr",
            name="CVPR author kit",
            remote_url="https://github.com/cvpr-org/author-kit/releases/latest/download/cvpr-author-kit.zip",
            overleaf_url="https://www.overleaf.com/gallery/tagged/cvpr-official",
        ),
        "acl": TemplateSpec(
            family="acl",
            name="ACL style files",
            remote_url="https://github.com/acl-org/acl-style-files/archive/refs/heads/master.zip",
            overleaf_url="https://www.overleaf.com/gallery/tagged/acl-official",
        ),
        "neurips": TemplateSpec(
            family="neurips",
            name="NeurIPS year-specific template fallback",
            overleaf_url="https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc",
            notes=[
                "NeurIPS templates are year-specific; use the current-year Overleaf or NeurIPS style page before submission.",
            ],
        ),
        "icml": TemplateSpec(
            family="icml",
            name="ICML year-specific template fallback",
            overleaf_url="https://www.overleaf.com/gallery/tagged/icml-official",
            notes=[
                "ICML templates are year-specific; use the exact target-year template before submission.",
            ],
        ),
        "iclr": TemplateSpec(
            family="iclr",
            name="ICLR-style template fallback",
            overleaf_url="https://www.overleaf.com/gallery/tagged/iclr",
            notes=[
                "ICLR templates change by year; using the built-in generic fallback until a year-specific source is selected.",
            ],
        ),
        "acm": TemplateSpec(
            family="acm",
            name="ACM template fallback",
            overleaf_url="https://www.overleaf.com/gallery/tagged/acm-official",
            notes=[
                "ACM templates vary by publication type; using the built-in generic fallback until the exact venue is selected.",
            ],
        ),
        "springer": TemplateSpec(
            family="springer",
            name="Springer LNCS template fallback",
            overleaf_url="https://www.overleaf.com/gallery/tagged/springer-official",
            notes=[
                "Springer templates vary by book or journal series; using the built-in generic fallback until the exact series is selected.",
            ],
        ),
    }

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        family = self._classify(request.target_venue)
        spec = self.TEMPLATE_REGISTRY.get(family, self.TEMPLATE_REGISTRY["generic"])
        template_root = Path(os.getenv("PAPER_AGENT_DATA_DIR", "data")) / "templates" / family
        template_root.mkdir(parents=True, exist_ok=True)

        source = "built-in"
        notes = list(spec.notes)
        template_fetch_disabled = os.getenv("PAPER_AGENT_DISABLE_TEMPLATE_FETCH", "").strip().lower()
        if template_fetch_disabled in {"1", "true", "yes", "on"}:
            notes.append("Remote template fetch disabled by PAPER_AGENT_DISABLE_TEMPLATE_FETCH.")
        elif spec.remote_url:
            source, fetch_notes = self._cache_remote_template(spec.remote_url, template_root)
            notes.extend(fetch_notes)

        state["venue_template"] = VenueTemplate(
            venue=request.target_venue,
            family=family,
            template_name=spec.name,
            template_source=source,
            overleaf_url=spec.overleaf_url,
            template_dir=str(template_root),
            main_template="main.tex.j2",
            notes=notes,
        )
        return state

    def _cache_remote_template(self, url: str, template_root: Path) -> tuple[str, list[str]]:
        notes: list[str] = []
        marker = template_root / "SOURCE_URL.txt"
        artifact = template_root / "remote_template"
        extracted = template_root / "extracted"

        try:
            if not marker.exists():
                response = httpx.get(
                    url,
                    timeout=20,
                    follow_redirects=True,
                    headers={"User-Agent": "paper-agent/0.1 template fetcher"},
                )
                response.raise_for_status()
                suffix = self._suffix_from_response(url, response)
                artifact = artifact.with_suffix(suffix)
                artifact.write_bytes(response.content)
                marker.write_text(url, encoding="utf-8")
            else:
                artifacts = sorted(template_root.glob("remote_template.*"))
                artifact = artifacts[0] if artifacts else artifact

            if artifact.suffix == ".zip":
                self._extract_zip(artifact, extracted)
                notes.append(f"Remote template archive cached and extracted: {artifact.name}")
            elif artifact.exists():
                notes.append(f"Remote template artifact cached: {artifact.name}")
            return url, notes
        except Exception as exc:
            notes.append(f"Remote template fetch failed; using built-in fallback: {exc}")
            return "built-in", notes

    def _suffix_from_response(self, url: str, response: httpx.Response) -> str:
        content_type = response.headers.get("content-type", "").lower()
        if url.endswith(".zip") or "zip" in content_type:
            return ".zip"
        if url.endswith(".cls"):
            return ".cls"
        return ".sty"

    def _extract_zip(self, artifact: Path, extracted: Path) -> None:
        if extracted.exists() and any(extracted.iterdir()):
            return
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir(parents=True, exist_ok=True)
        try:
            with ZipFile(artifact) as archive:
                for member in archive.infolist():
                    destination = (extracted / member.filename).resolve()
                    try:
                        destination.relative_to(extracted.resolve())
                    except ValueError:
                        continue
                    archive.extract(member, extracted)
        except BadZipFile as exc:
            raise ValueError(f"Downloaded template is not a valid zip: {artifact}") from exc

    def _classify(self, venue: str) -> str:
        lowered = venue.lower()
        if any(name in lowered for name in ["tpami", "t-pami", "pattern analysis and machine intelligence"]):
            return "ieee_journal"
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
