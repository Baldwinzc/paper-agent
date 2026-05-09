"""Shared state and public models for the paper generation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class PaperRequest(BaseModel):
    """User-provided research materials."""

    project_name: str = "untitled-paper"
    target_venue: str
    baseline_pdf_path: str | None = None
    code_path: str | None = None
    template_zip_path: str | None = None
    template_dir_path: str | None = None
    method_notes: str = ""
    experiment_results: str = ""
    keywords: list[str] = Field(default_factory=list)
    skip_llm_self_review: bool = False


class BaselineSummary(BaseModel):
    title: str = ""
    problem: str = ""
    method: str = ""
    experiments: str = ""
    limitations: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    structured_sections: dict[str, str] = Field(default_factory=dict)
    references: dict[str, str] = Field(default_factory=dict)
    extracted_text_preview: str = ""


class CodeSummary(BaseModel):
    path: str = ""
    languages: dict[str, int] = Field(default_factory=dict)
    likely_entrypoints: list[str] = Field(default_factory=list)
    likely_method_files: list[str] = Field(default_factory=list)
    implementation_evidence: list[str] = Field(default_factory=list)
    method_claims: list[str] = Field(default_factory=list)
    summary: str = ""


class ExperimentComparison(BaseModel):
    table_caption: str = ""
    dataset: str = ""
    metric: str = ""
    method: str = ""
    baseline: str = ""
    method_value: float
    baseline_value: float
    signed_improvement: float
    higher_is_better: bool = True
    improved: bool = False


class ExperimentTableSummary(BaseModel):
    caption: str = ""
    metric: str = ""
    method: str = ""
    baseline: str = ""
    comparisons: list[ExperimentComparison] = Field(default_factory=list)


class AblationEvidence(BaseModel):
    table_caption: str = ""
    dataset: str = ""
    metric: str = ""
    reference: str = ""
    variant: str = ""
    reference_value: float
    variant_value: float
    signed_drop: float
    higher_is_better: bool = True
    supports: list[str] = Field(default_factory=list)


class ExperimentSummary(BaseModel):
    raw_preview: str = ""
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    result_tables: list[ExperimentTableSummary] = Field(default_factory=list)
    ablation_evidence: list[AblationEvidence] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    missing_details: list[str] = Field(default_factory=list)


class CitationEntry(BaseModel):
    key: str
    title: str
    query: str = ""
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    note: str = ""


class InnovationPoint(BaseModel):
    name: str
    motivation: str
    technical_idea: str
    evidence: list[str] = Field(default_factory=list)
    risk: str = ""


class VenueTemplate(BaseModel):
    venue: str
    family: Literal[
        "generic",
        "ieee",
        "ieee_journal",
        "acm",
        "springer",
        "neurips",
        "icml",
        "iclr",
        "acl",
        "cvpr",
    ] = "generic"
    template_name: str = ""
    template_source: str = "built-in"
    overleaf_url: str = ""
    template_dir: str = ""
    sample_main_tex: str = ""
    main_template: str = "main.tex.j2"
    notes: list[str] = Field(default_factory=list)


class PaperOutline(BaseModel):
    title_candidates: list[str] = Field(default_factory=list)
    central_claim: str = ""
    section_plan: dict[str, list[str]] = Field(default_factory=dict)


class DraftSections(BaseModel):
    abstract: str = ""
    introduction: str = ""
    related_work: str = ""
    method: str = ""
    experiments: str = ""
    conclusion: str = ""


class ReviewFinding(BaseModel):
    severity: Literal["major", "minor"]
    issue: str
    suggestion: str


class PaperState(TypedDict, total=False):
    request: PaperRequest
    baseline: BaselineSummary
    code: CodeSummary
    experiments: ExperimentSummary
    bibliography: list[CitationEntry]
    innovations: list[InnovationPoint]
    venue_template: VenueTemplate
    outline: PaperOutline
    sections: DraftSections
    latex_project_dir: Path
    latex_output_path: Path
    latex_zip_path: Path
    review_findings: list[ReviewFinding]
    final_markdown: str
    artifacts: dict[str, Any]
