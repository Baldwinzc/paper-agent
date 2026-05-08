import os
from pathlib import Path
from zipfile import ZipFile

from paper_agent.export import zip_latex_project
from paper_agent.tables import extract_markdown_tables, markdown_tables_to_latex
from paper_agent.state import CitationEntry, PaperOutline, PaperRequest, VenueTemplate
from paper_agent.workflow import PaperWorkflow
from paper_agent.agents.baseline_reader import BaselineReaderAgent
from paper_agent.agents.evidence_guard import EvidenceGuardAgent
from paper_agent.agents.experiment_analyzer import ExperimentAnalyzerAgent
from paper_agent.agents.latex_composer import LatexComposerAgent
from paper_agent.agents.draft_report import DraftReportAgent
from paper_agent.agents.reference_resolver import ReferenceResolverAgent
from paper_agent.agents.reviewer import ReviewerAgent
from paper_agent.state import CodeSummary, DraftSections, ExperimentSummary


os.environ.setdefault("PAPER_AGENT_DISABLE_TEMPLATE_FETCH", "1")
os.environ.setdefault("PAPER_AGENT_DISABLE_LLM", "1")
os.environ.setdefault("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "1")


def test_baseline_reader_uses_descriptive_pdf_filename_for_truncated_title():
    reader = BaselineReaderAgent()
    text_title = "Leveraging Tumor Heterogeneity: Heterogeneous"
    path_title = reader._guess_title_from_path(
        "NeurIPS-2024-leveraging-tumor-heterogeneity-heterogeneous-graph-representation-learning-for-cancer-survival-prediction-in-whole-slide-images-Paper-Conference.pdf"
    )

    assert (
        path_title
        == "Leveraging Tumor Heterogeneity Heterogeneous Graph Representation Learning for Cancer Survival Prediction in Whole Slide Images"
    )
    assert reader._best_title(text_title, path_title) == path_title


def test_workflow_generates_latex_and_sections():
    request = PaperRequest(
        project_name="demo-paper",
        target_venue="IEEE Conference",
        method_notes="Adaptive feature calibration",
        experiment_results=(
            "| Method | DatasetA Accuracy |\n"
            "|---|---:|\n"
            "| baseline | 80 |\n"
            "| ours | 83 |\n"
        ),
    )

    state = PaperWorkflow().run(request)

    assert state["sections"].abstract
    assert state["innovations"]
    assert state["venue_template"].family == "ieee"
    assert state["venue_template"].overleaf_url
    assert state["latex_project_dir"].name == "demo-paper"
    assert state["latex_output_path"].name == "main.tex"
    assert r"\begin{table}" in state["latex_output_path"].read_text(encoding="utf-8")
    assert state["bibliography"]
    assert (state["latex_project_dir"] / "references.bib").read_text(encoding="utf-8")
    report = state["latex_project_dir"] / "DRAFT_REPORT.md"
    assert report.exists()
    assert "Draft Report" in report.read_text(encoding="utf-8")


def test_tpami_uses_ieee_journal_template():
    request = PaperRequest(
        project_name="demo-paper",
        target_venue="TPAMI",
        method_notes="Adaptive feature calibration",
    )

    state = PaperWorkflow().run(request)

    assert state["venue_template"].family == "ieee_journal"
    assert state["venue_template"].template_name == "IEEE journal paper template"
    assert state["venue_template"].overleaf_url == "https://www.overleaf.com/org/ieee"


def test_evidence_guard_blocks_no_cox_claim_when_cox_evidence_exists():
    state = {
        "code": CodeSummary(method_claims=["L = L_surv (Cox PH) + lambda_rec * L_rec"]),
        "experiments": ExperimentSummary(missing_details=[]),
        "sections": DraftSections(method="No Cox loss is used. The method uses only one supervised signal."),
        "artifacts": {},
    }

    guarded = EvidenceGuardAgent().run(state)

    assert "Cox survival loss is retained" in guarded["sections"].method
    assert guarded["artifacts"]["evidence_guard_findings"]


def test_evidence_guard_blocks_numeric_experiment_claims_without_results():
    state = {
        "experiments": ExperimentSummary(missing_details=["Baseline comparison rows should be made explicit."]),
        "sections": DraftSections(
            experiments="Removing BHE degrades C-index by ≥0.023 across cohorts.",
            conclusion="The method achieves greater robustness and measurable gains.",
        ),
        "artifacts": {},
    }

    guarded = EvidenceGuardAgent().run(state)

    assert "unsupported empirical claim" in guarded["sections"].experiments
    assert "unsupported empirical claim" in guarded["sections"].conclusion


def test_evidence_guard_blocks_validate_on_and_outperforms_without_results():
    state = {
        "experiments": ExperimentSummary(missing_details=["Exact result table required."]),
        "sections": DraftSections(
            abstract=(
                "We validate Hyper-ProtoSurv on multiple cohorts. "
                "Preliminary analyses indicate it outperforms ProtoSurv."
            ),
        ),
        "artifacts": {},
    }

    guarded = EvidenceGuardAgent().run(state)

    assert "unsupported empirical claim" in guarded["sections"].abstract


def test_experiment_analyzer_extracts_tcga_mock_results():
    raw = """
    | Method | BLCA | BRCA | LGG | LUAD | UCEC |
    |---|---:|---:|---:|---:|---:|
    | ProtoSurv baseline | 0.646 | 0.669 | 0.724 | 0.636 | 0.658 |
    | Hyper-ProtoSurv ours | 0.671 | 0.691 | 0.746 | 0.661 | 0.681 |
    Metric: C-index and IBS. Ablation w/o L_rec is lower.
    """
    state = {
        "request": PaperRequest(
            project_name="demo",
            target_venue="TPAMI",
            experiment_results=raw,
        )
    }

    state = ExperimentAnalyzerAgent().run(state)

    assert state["experiments"].datasets == ["BLCA", "BRCA", "LGG", "LUAD", "UCEC"]
    assert "C-INDEX" in state["experiments"].metrics
    assert "IBS" in state["experiments"].metrics
    assert not state["experiments"].missing_details


def test_zip_latex_project_contains_overleaf_files(tmp_path):
    project = tmp_path / "paper"
    project.mkdir()
    (project / "main.tex").write_text(r"\documentclass{article}", encoding="utf-8")
    (project / "references.bib").write_text("% refs", encoding="utf-8")

    zip_path = zip_latex_project(project, tmp_path / "paper.zip")

    assert zip_path.exists()
    with ZipFile(zip_path) as archive:
        assert set(archive.namelist()) == {"main.tex", "references.bib"}


def test_workflow_writes_template_source_notes():
    request = PaperRequest(
        project_name="template-notes-demo",
        target_venue="TPAMI",
        method_notes="Adaptive feature calibration",
    )

    state = PaperWorkflow().run(request)
    source_notes = state["latex_project_dir"] / "TEMPLATE_SOURCE.md"

    assert source_notes.exists()
    content = source_notes.read_text(encoding="utf-8")
    assert "IEEE journal paper template" in content
    assert "https://www.overleaf.com/org/ieee" in content


def test_user_template_directory_supplies_preamble_and_assets(tmp_path):
    template_dir = tmp_path / "official-template"
    template_dir.mkdir()
    (template_dir / "main.tex").write_text(
        "\n".join(
            [
                r"\documentclass[journal]{IEEEtran}",
                r"\usepackage{officialstyle}",
                r"\title{Official Sample Title}",
                r"\begin{document}",
                r"\maketitle",
                r"Sample body.",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    (template_dir / "officialstyle.sty").write_text(r"\ProvidesPackage{officialstyle}", encoding="utf-8")

    state = PaperWorkflow().run(
        PaperRequest(
            project_name="manual-template-dir-demo",
            target_venue="TPAMI",
            method_notes="Adaptive feature calibration",
            template_dir_path=str(template_dir),
        )
    )
    tex = state["latex_output_path"].read_text(encoding="utf-8")
    source_notes = (state["latex_project_dir"] / "TEMPLATE_SOURCE.md").read_text(encoding="utf-8")

    assert state["venue_template"].template_source.startswith("user-dir:")
    assert r"\documentclass[journal]{IEEEtran}" in tex
    assert r"\usepackage{officialstyle}" in tex
    assert "\\title{" in tex
    assert "\title" not in tex
    assert "Official Sample Title" not in tex
    assert (state["latex_project_dir"] / "officialstyle.sty").exists()
    assert "user-dir:" in source_notes


def test_user_template_zip_is_extracted_and_detected(tmp_path):
    source_dir = tmp_path / "zip-source"
    source_dir.mkdir()
    (source_dir / "main.tex").write_text(
        r"\documentclass{article}\begin{document}Sample\end{document}",
        encoding="utf-8",
    )
    (source_dir / "custom.cls").write_text(r"\NeedsTeXFormat{LaTeX2e}", encoding="utf-8")
    zip_path = tmp_path / "official-template.zip"
    with ZipFile(zip_path, "w") as archive:
        for path in source_dir.rglob("*"):
            archive.write(path, Path("template") / path.name)

    state = PaperWorkflow().run(
        PaperRequest(
            project_name="manual-template-zip-demo",
            target_venue="TPAMI",
            method_notes="Adaptive feature calibration",
            template_zip_path=str(zip_path),
        )
    )

    assert state["venue_template"].template_source.startswith("user-zip:")
    assert state["venue_template"].sample_main_tex.endswith("main.tex")
    assert (state["latex_project_dir"] / "custom.cls").exists()


def test_markdown_experiment_tables_render_as_booktabs_latex():
    raw = """
    ## Main Results
    Metric: C-index.

    | Method | BLCA | Average |
    |---|---:|---:|
    | ProtoSurv baseline | 0.646 +/- 0.024 | 0.667 |
    | Hyper-ProtoSurv ours | 0.671 +/- 0.021 | 0.690 |

    ## Ablation Results

    | Variant | Average C-index |
    |---|---:|
    | Full | 0.690 |
    | w/o L_rec | 0.672 |
    """

    tables = extract_markdown_tables(raw)
    latex = markdown_tables_to_latex(raw)

    assert len(tables) == 2
    assert tables[0].caption == "Main Results. Metric: C-index."
    assert r"\toprule" in latex
    assert r"$\pm$" in latex
    assert r"w/o L\_rec" in latex


def test_bibliography_seeds_are_written_to_markdown_and_bibtex():
    request = PaperRequest(
        project_name="citation-demo",
        target_venue="TPAMI",
        method_notes="Adaptive feature calibration",
        keywords=["survival prediction", "whole-slide images"],
    )

    state = PaperWorkflow().run(request)
    markdown = state["final_markdown"]
    bibtex = (state["latex_project_dir"] / "references.bib").read_text(encoding="utf-8")

    assert "## Reference Seeds" in markdown
    assert "@misc{" in bibtex
    assert "Verify metadata before submission" in bibtex
    assert any("Bibliography contains seed entries" in finding.issue for finding in state["review_findings"])


def test_reference_resolver_enriches_seed_entry(monkeypatch):
    def fake_query(self, query):
        return {
            "results": [
                {
                    "title": "Computational pathology with whole-slide images",
                    "doi": "https://doi.org/10.1234/example",
                    "publication_year": 2024,
                    "primary_location": {"source": {"display_name": "IEEE Transactions"}},
                    "authorships": [
                        {"author": {"display_name": "Ada Lovelace"}},
                        {"author": {"display_name": "Grace Hopper"}},
                    ],
                    "ids": {"openalex": "https://openalex.org/W123"},
                }
            ]
        }

    monkeypatch.setenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "0")
    monkeypatch.setattr(ReferenceResolverAgent, "_query_openalex", fake_query)
    state = PaperWorkflow().run(
        PaperRequest(
            project_name="resolver-demo",
            target_venue="TPAMI",
            method_notes="Adaptive feature calibration",
            keywords=["whole-slide images"],
        )
    )
    bibtex = (state["latex_project_dir"] / "references.bib").read_text(encoding="utf-8")

    assert "Ada Lovelace and Grace Hopper" in bibtex
    assert "year = {2024}" in bibtex
    assert "doi = {10.1234/example}" in bibtex
    assert "journal = {IEEE Transactions}" in bibtex
    assert state["artifacts"]["reference_resolver_resolved"] >= 1


def test_reference_resolver_rejects_low_confidence_match(monkeypatch):
    def fake_query(self, query):
        return {
            "results": [
                {
                    "title": "Unrelated work on software maintenance",
                    "doi": "https://doi.org/10.1234/unrelated",
                    "publication_year": 2024,
                    "authorships": [],
                }
            ]
        }

    monkeypatch.setenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "0")
    monkeypatch.setattr(ReferenceResolverAgent, "_query_openalex", fake_query)
    monkeypatch.setattr(ReferenceResolverAgent, "_query_semantic_scholar", lambda self, query: {"data": []})
    entry = ReferenceResolverAgent()._resolve_entry(
        CitationEntry(
            key="x",
            title="Representative work on whole slide images",
            query="whole slide images cancer survival prediction",
        )
    )

    assert not entry.doi
    assert "low-confidence" in entry.note


def test_reference_resolver_falls_back_to_semantic_scholar(monkeypatch):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "0")
    monkeypatch.setattr(ReferenceResolverAgent, "_query_openalex", lambda self, query: {"results": []})
    monkeypatch.setattr(
        ReferenceResolverAgent,
        "_query_semantic_scholar",
        lambda self, query: {
            "data": [
                {
                    "title": "Whole slide image survival prediction with hypergraph learning",
                    "year": 2025,
                    "venue": "Medical Image Analysis",
                    "externalIds": {"DOI": "10.1234/s2"},
                    "url": "https://www.semanticscholar.org/paper/test",
                    "authors": [{"name": "Ada Lovelace"}],
                }
            ]
        },
    )
    entry = ReferenceResolverAgent()._resolve_entry(
        CitationEntry(
            key="x",
            title="Representative work on hypergraph learning",
            query="whole slide image survival prediction hypergraph learning",
        )
    )

    assert entry.doi == "10.1234/s2"
    assert entry.year == "2025"
    assert entry.venue == "Medical Image Analysis"
    assert "Semantic Scholar" in entry.note


def test_reference_resolver_handles_semantic_scholar_rate_limit(monkeypatch):
    class FakeResponse:
        status_code = 429

    def raise_rate_limit(self, query):
        raise __import__("httpx").HTTPStatusError("rate limited", request=None, response=FakeResponse())

    monkeypatch.setenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "0")
    monkeypatch.setattr(ReferenceResolverAgent, "_query_openalex", lambda self, query: {"results": []})
    monkeypatch.setattr(ReferenceResolverAgent, "_query_semantic_scholar", raise_rate_limit)
    resolver = ReferenceResolverAgent()

    first = resolver._resolve_entry(CitationEntry(key="a", title="A", query="whole slide image survival prediction"))
    second = resolver._resolve_entry(CitationEntry(key="b", title="B", query="hypergraph learning"))

    assert "rate limited" in first.note
    assert "skipped after rate limit" in second.note
    assert "https://api.semanticscholar.org" not in first.note


def test_reference_resolver_deduplicates_repeated_dois(monkeypatch):
    def fake_query(self, query):
        return {
            "results": [
                {
                    "title": f"{query} study",
                    "doi": "https://doi.org/10.1234/shared",
                    "publication_year": 2024,
                    "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                }
            ]
        }

    monkeypatch.setenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "0")
    monkeypatch.setattr(ReferenceResolverAgent, "_query_openalex", fake_query)
    state = PaperWorkflow().run(
        PaperRequest(
            project_name="dedupe-demo",
            target_venue="TPAMI",
            method_notes="Adaptive feature calibration",
            keywords=["whole-slide images", "survival prediction"],
        )
    )

    dois = [entry.doi for entry in state["bibliography"] if entry.doi]
    assert dois == ["10.1234/shared"]
    assert state["artifacts"]["citation_key_aliases"]


def test_known_markdown_citations_convert_to_latex_cite():
    request = PaperRequest(
        project_name="citation-conversion-demo",
        target_venue="TPAMI",
        method_notes="Adaptive feature calibration",
        keywords=["survival prediction"],
    )

    state = PaperWorkflow().run(request)
    state["sections"].related_work = "Prior survival work [survivalprediction] motivates this setting."
    state = LatexComposerAgent().run(state)

    tex = state["latex_output_path"].read_text(encoding="utf-8")
    assert r"\cite{survivalprediction}" in tex


def test_latex_composer_converts_markdown_bold():
    request = PaperRequest(
        project_name="bold-demo",
        target_venue="TPAMI",
        method_notes="Adaptive feature calibration",
    )

    state = PaperWorkflow().run(request)
    state["sections"].related_work = "**Important thread** discusses the baseline."
    state = LatexComposerAgent().run(state)

    tex = state["latex_output_path"].read_text(encoding="utf-8")
    assert r"\textbf{Important thread}" in tex


def test_latex_composer_converts_placeholder_markup_to_todo():
    request = PaperRequest(
        project_name="placeholder-demo",
        target_venue="TPAMI",
        method_notes="Adaptive feature calibration",
    )

    state = PaperWorkflow().run(request)
    state["sections"].experiments = "[PLACEHOLDER: Insert the real ablation table.]"
    state = LatexComposerAgent().run(state)

    tex = state["latex_output_path"].read_text(encoding="utf-8")
    assert r"\textbf{TODO:} Insert the real ablation table." in tex
    assert "[PLACEHOLDER" not in tex


def test_reviewer_flags_placeholders():
    state = {
        "experiments": ExperimentSummary(),
        "innovations": [],
        "sections": DraftSections(method="The architecture is shown in Fig. [placeholder for figure]."),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)

    assert any("placeholders" in finding.issue for finding in reviewed["review_findings"])


def test_citation_aliases_convert_to_retained_key():
    request = PaperRequest(
        project_name="citation-alias-demo",
        target_venue="TPAMI",
        method_notes="Adaptive feature calibration",
        keywords=["whole-slide images", "survival prediction"],
    )

    state = PaperWorkflow().run(request)
    state["artifacts"]["citation_key_aliases"] = {"survivalprediction": "wholeslideimages"}
    state["bibliography"] = [entry for entry in state["bibliography"] if entry.key != "survivalprediction"]
    state["sections"].related_work = "Prior work [survivalprediction] motivates this setting."
    state = LatexComposerAgent().run(state)

    tex = state["latex_output_path"].read_text(encoding="utf-8")
    assert r"\cite{wholeslideimages}" in tex
    assert "survivalprediction" not in tex


def test_citation_aliases_convert_raw_latex_cite_to_retained_key(tmp_path):
    state = {
        "request": PaperRequest(project_name="raw-citation-alias-demo", target_venue="TPAMI"),
        "venue_template": VenueTemplate(
            venue="TPAMI",
            family="ieee_journal",
            template_dir=str(tmp_path / "missing-template"),
        ),
        "outline": PaperOutline(title_candidates=["Raw Citation Alias Demo"]),
        "sections": DraftSections(),
        "bibliography": [
            CitationEntry(key="wholeslideimages", title="Whole slide images", query="whole slide images")
        ],
        "artifacts": {"citation_key_aliases": {"survivalprediction": "wholeslideimages"}},
    }
    state["sections"].related_work = r"Prior work \cite{survivalprediction} motivates this setting."
    state = LatexComposerAgent().run(state)

    tex = state["latex_output_path"].read_text(encoding="utf-8")
    assert r"\cite{wholeslideimages}" in tex
    assert "survivalprediction" not in tex


def test_undefined_raw_latex_citation_is_reported(tmp_path):
    state = {
        "request": PaperRequest(project_name="undefined-citation-demo", target_venue="TPAMI"),
        "venue_template": VenueTemplate(
            venue="TPAMI",
            family="ieee_journal",
            template_dir=str(tmp_path / "missing-template"),
        ),
        "outline": PaperOutline(title_candidates=["Undefined Citation Demo"]),
        "sections": DraftSections(
            related_work=r"Prior work \cite{missing_key} should be resolved before submission."
        ),
        "bibliography": [
            CitationEntry(key="wholeslideimages", title="Whole slide images", query="whole slide images")
        ],
        "artifacts": {},
    }
    state = LatexComposerAgent().run(state)
    state = DraftReportAgent().run(state)

    report = (state["latex_project_dir"] / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    tex = state["latex_output_path"].read_text(encoding="utf-8")
    assert state["artifacts"]["undefined_citation_keys"] == ["missing_key"]
    assert r"\cite{missing_key}" in tex
    assert "## Undefined Citations" in report
    assert "`missing_key`" in report
