import json
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from paper_agent import api as api_module
from paper_agent import cli as cli_module
from paper_agent.export import zip_latex_project
from paper_agent.tables import extract_markdown_tables, markdown_tables_to_latex
from paper_agent.state import CitationEntry, InnovationPoint, PaperOutline, PaperRequest, VenueTemplate
from paper_agent.workflow import PaperWorkflow
from paper_agent.agents.baseline_reader import BaselineReaderAgent
from paper_agent.agents.bibliography import BibliographyAgent
from paper_agent.agents.evidence_guard import EvidenceGuardAgent
from paper_agent.agents.experiment_analyzer import ExperimentAnalyzerAgent
from paper_agent.agents.innovation_analyzer import InnovationAnalyzerAgent
from paper_agent.agents.latex_composer import LatexComposerAgent
from paper_agent.agents.llm_self_review import LLMSelfReviewAgent
from paper_agent.agents.draft_report import DraftReportAgent
from paper_agent.agents.reference_resolver import ReferenceResolverAgent
from paper_agent.agents.related_work_discovery import RelatedWorkDiscoveryAgent
from paper_agent.agents.reviewer import ReviewerAgent
from paper_agent.agents.section_writer import SectionWriterAgent
from paper_agent.state import BaselineSummary, CodeSummary, DraftSections, ExperimentSummary


os.environ.setdefault("PAPER_AGENT_DISABLE_TEMPLATE_FETCH", "1")
os.environ.setdefault("PAPER_AGENT_DISABLE_LLM", "1")
os.environ.setdefault("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "1")
os.environ.setdefault("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY", "1")


class FakeLLMClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return SimpleNamespace(content=self.content, model="fake", usage={}, raw={})


class FakeSequenceLLMClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = []

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return SimpleNamespace(content=self.contents.pop(0), model="fake", usage={}, raw={})


def _write_pdf(path: Path, text: str) -> None:
    import fitz  # type: ignore

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


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


def test_baseline_reader_strips_pdf_section_prefix_noise():
    reader = BaselineReaderAgent()
    text = (
        "author@example.edu.cn\n"
        "cn\n"
        "Abstract\n"
        "Survival prediction is a significant challenge in cancer management.\n"
        "Introduction The method section describes the proposed framework."
    )

    cleaned = reader._clean_extracted_text(text)
    problem = reader._guess_sentence(cleaned, ["challenge"])

    assert "Abstract" not in cleaned
    assert not cleaned.startswith("cn")
    assert problem.startswith("Survival prediction")


def test_innovation_name_uses_readable_hypergraph_title():
    name = InnovationAnalyzerAgent()._innovation_name(
        "Hyper-ProtoSurv explores adaptive hypergraph prototype learning, "
        "bidirectional hyperedge updates, cross-attention fusion, and reconstruction "
        "regularization as reflected by the code structure and mock ablation table."
    )

    assert name == "Adaptive hypergraph prototype learning with bidirectional updates"


def test_innovation_name_truncates_at_word_boundary():
    name = InnovationAnalyzerAgent()._innovation_name(
        "A long contribution title with adaptive calibration, hierarchical modeling, "
        "regularized survival objectives, uncertainty handling, and efficient inference"
    )

    assert len(name) <= 90
    assert name.endswith("efficient") is False
    assert not name.endswith("infer")


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


def test_acceptance_flow_inputs_code_baseline_venue_outputs_paper(tmp_path):
    baseline_pdf = tmp_path / "Baseline-Survival-Prediction-with-Prototype-Graphs.pdf"
    _write_pdf(
        baseline_pdf,
        (
            "Prototype Graphs for Survival Prediction\n"
            "Abstract Survival prediction is a significant challenge in computational pathology.\n"
            "The method uses graph prototypes for whole-slide image representation learning.\n"
            "Experiments evaluate C-index on TCGA cohorts.\n"
        ),
    )
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "train.py").write_text(
        "class AdaptivePrototypeSurvivalModel:\n"
        "    def forward(self, graph):\n"
        "        return self.hypergraph_attention(graph)\n",
        encoding="utf-8",
    )
    request = PaperRequest(
        project_name="acceptance-flow-demo",
        target_venue="TPAMI",
        baseline_pdf_path=str(baseline_pdf),
        code_path=str(code_dir),
        method_notes="Adaptive prototype calibration for survival prediction",
        experiment_results=(
            "| Method | TCGA C-index |\n"
            "|---|---:|\n"
            "| baseline | 0.62 |\n"
            "| ours | 0.66 |\n"
        ),
        keywords=["whole-slide images", "survival prediction"],
        skip_llm_self_review=True,
    )

    state = PaperWorkflow().run(request)
    markdown_path = tmp_path / "draft.md"
    markdown_path.write_text(state["final_markdown"], encoding="utf-8")
    summary_path = cli_module._write_run_summary(state, tmp_path / "RUN_SUMMARY.json", markdown_path)

    assert state["baseline"].title == "Prototype Graphs for Survival Prediction"
    assert "train.py" in state["code"].likely_entrypoints
    assert state["venue_template"].family == "ieee_journal"
    assert state["sections"].abstract
    assert state["sections"].method
    assert state["sections"].experiments
    assert state["latex_output_path"].exists()
    assert (state["latex_project_dir"] / "DRAFT_REPORT.md").exists()
    assert (state["latex_project_dir"] / "references.bib").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["inputs"]["code_path"] == str(code_dir)
    assert summary["inputs"]["baseline_pdf_path"] == str(baseline_pdf)
    assert summary["inputs"]["target_venue"] == "TPAMI"
    assert summary["inputs"]["experiment_results_provided"]
    assert summary["outputs"]["markdown"] == str(markdown_path)


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
    assert state["artifacts"]["experiment_result_findings"]
    assert "5/5 numeric comparisons" in state["experiments"].observations[0]
    assert "average signed improvement +0.023" in state["experiments"].observations[0]


def test_experiment_analyzer_handles_lower_is_better_metrics():
    raw = """
    | Method | DatasetA IBS | DatasetB IBS |
    |---|---:|---:|
    | baseline | 0.180 | 0.210 |
    | ours | 0.160 | 0.190 |
    """
    state = {
        "request": PaperRequest(
            project_name="demo",
            target_venue="TPAMI",
            experiment_results=raw,
        )
    }

    state = ExperimentAnalyzerAgent().run(state)

    assert "2/2 numeric comparisons" in state["experiments"].observations[0]
    assert "average signed improvement +0.020" in state["experiments"].observations[0]


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
    assert any(
        "Bibliography contains" in finding.issue and "unresolved seed" in finding.issue
        for finding in state["review_findings"]
    )


def test_bibliography_uses_technical_queries_for_innovation_threads():
    state = {
        "request": PaperRequest(
            project_name="citation-demo",
            target_venue="TPAMI",
            keywords=["whole-slide images", "survival prediction"],
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: OT-driven adaptive hyperedges with bidirectional hyperedge convolution",
                technical_idea="Build OT-driven adaptive hyperedges for survival prediction.",
                motivation="The baseline uses static prototypes.",
            ),
            InnovationPoint(
                name="Innovation 2: Wasserstein-barycenter prototype geometry",
                technical_idea="Use Wasserstein-barycenter prototype geometry.",
                motivation="The baseline uses static prototypes.",
            ),
            InnovationPoint(
                name="Innovation 3: Minimal survival-reconstruction training objective",
                technical_idea="Use a survival-reconstruction objective for representation learning.",
                motivation="The baseline objective is incomplete.",
            ),
        ],
        "artifacts": {},
    }

    state = BibliographyAgent().run(state)

    queries = {entry.query for entry in state["bibliography"]}
    keys = {entry.key for entry in state["bibliography"]}
    assert any("optimal transport hypergraph learning" in query for query in queries)
    assert any("prototype learning" in query for query in queries)
    assert any("survival prediction representation learning" in query for query in queries)
    assert not any(key.startswith("innovation") for key in keys)
    assert not any("Innovation 1" in entry.title for entry in state["bibliography"])


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
    assert state["artifacts"]["reference_verification"]["resolved_count"] >= 1


def test_reference_resolver_selects_best_openalex_candidate(monkeypatch):
    def fake_query(self, query):
        return {
            "results": [
                {
                    "title": "Unrelated work on software maintenance",
                    "doi": "https://doi.org/10.1234/unrelated",
                    "publication_year": 2024,
                    "authorships": [],
                },
                {
                    "title": "Whole slide image survival prediction with hypergraph learning",
                    "doi": "https://doi.org/10.1234/relevant",
                    "publication_year": 2025,
                    "primary_location": {"source": {"display_name": "Medical Image Analysis"}},
                    "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                },
            ]
        }

    monkeypatch.setenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "0")
    monkeypatch.setattr(ReferenceResolverAgent, "_query_openalex", fake_query)

    entry = ReferenceResolverAgent()._resolve_entry(
        CitationEntry(
            key="x",
            title="Representative work on hypergraph learning",
            query="whole slide image survival prediction hypergraph learning",
        )
    )

    assert entry.doi == "10.1234/relevant"
    assert entry.title == "Whole slide image survival prediction with hypergraph learning"


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


def test_related_work_discovery_adds_categorized_candidates(monkeypatch):
    def work(title, identifier, year, cited_by_count, referenced_works=None):
        return {
            "id": f"https://openalex.org/{identifier}",
            "title": title,
            "doi": f"https://doi.org/10.1234/{identifier.lower()}",
            "publication_year": year,
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "primary_location": {"source": {"display_name": "IEEE Transactions"}},
            "ids": {"openalex": f"https://openalex.org/{identifier}"},
            "referenced_works": referenced_works or [],
            "cited_by_count": cited_by_count,
        }

    def fake_query(self, params):
        if params.get("search") == "Baseline Survival Paper":
            return {
                "results": [
                    work(
                        "Baseline Survival Paper",
                        "WBASE",
                        2024,
                        42,
                        referenced_works=["https://openalex.org/WCLASSIC"],
                    )
                ]
            }
        if str(params.get("filter", "")).startswith("openalex_id:"):
            return {"results": [work("Classic survival analysis for whole-slide images", "WCLASSIC", 2018, 500)]}
        if str(params.get("filter", "")).startswith("cites:"):
            return {"results": [work("Recent extension that cites the baseline", "WFOLLOW", 2026, 7)]}
        if params.get("sort") == "cited_by_count:desc":
            return {"results": [work("Influential computational pathology survey", "WINFLUENTIAL", 2020, 900)]}
        if params.get("sort") == "publication_date:desc":
            return {
                "results": [
                    work("New whole-slide survival prediction model", "WRECENT", 2026, 3),
                    work("New whole-slide survival prediction model", "WRECENTDUP", 2026, 3),
                ]
            }
        return {"results": []}

    monkeypatch.setenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "0")
    monkeypatch.setenv("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY", "0")
    monkeypatch.setattr(RelatedWorkDiscoveryAgent, "_query_openalex", fake_query)
    state = {
        "request": PaperRequest(
            project_name="discovery-demo",
            target_venue="TPAMI",
            keywords=["whole-slide images", "survival prediction"],
        ),
        "baseline": BaselineSummary(
            title="Baseline Survival Paper",
            related_terms=["computational pathology"],
        ),
        "bibliography": [
            CitationEntry(
                key="baseline",
                title="Baseline Survival Paper",
                query="Baseline Survival Paper",
            )
        ],
        "artifacts": {},
    }

    state = RelatedWorkDiscoveryAgent().run(state)

    categories = {item["category"] for item in state["artifacts"]["related_work_candidates"]}
    titles = [item["title"] for item in state["artifacts"]["related_work_candidates"]]
    assert {"baseline_reference", "baseline_citing", "influential", "recent"} <= categories
    assert titles.count("New whole-slide survival prediction model") == 1
    assert any(entry.title == "Classic survival analysis for whole-slide images" for entry in state["bibliography"])
    assert state["artifacts"]["citation_keys"]


def test_section_writer_uses_related_work_discovery_in_fallback():
    state = {
        "request": PaperRequest(project_name="related-work-demo", target_venue="TPAMI"),
        "sections": DraftSections(),
        "innovations": [],
        "artifacts": {
            "citation_keys": ["baseline"],
            "related_work_candidates": [
                {
                    "key": "classicpaper",
                    "category": "baseline_reference",
                    "title": "Classic paper",
                },
                {
                    "key": "recentpaper",
                    "category": "recent",
                    "title": "Recent paper",
                },
            ],
        },
    }

    sections = SectionWriterAgent()._run_fallback(state)

    assert r"\cite{classicpaper}" in sections.related_work
    assert r"\cite{recentpaper}" in sections.related_work
    assert "### Baseline Lineage" in sections.related_work
    assert "### Recent Developments" in sections.related_work
    assert "Classic paper" in sections.related_work
    assert "should be discussed" not in sections.related_work
    assert "should be positioned" not in sections.related_work


def test_section_writer_fallback_uses_paper_prose_for_core_sections():
    state = {
        "request": PaperRequest(project_name="prose-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(
            title="Baseline Survival Paper",
            problem="Whole-slide survival prediction estimates patient risk from pathology images.",
            limitations=["The baseline uses static prototypes."],
        ),
        "experiments": ExperimentSummary(
            datasets=["BLCA", "BRCA"],
            metrics=["C-INDEX"],
            observations=[
                "Table 1: Hyper-ProtoSurv ours improves over ProtoSurv baseline on 2/2 numeric comparisons (average signed improvement +0.020)."
            ],
            missing_details=[],
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype calibration",
                motivation="The baseline uses static prototypes.",
                technical_idea="Calibrate prototypes with uncertainty-aware adaptation.",
                evidence=["Repository exposes adaptive prototype code."],
                risk="Inferred from repository text; user should confirm novelty and wording.",
            )
        ],
        "outline": PaperOutline(
            central_claim="This paper improves the baseline setting through adaptive prototype calibration."
        ),
        "artifacts": {},
    }

    sections = SectionWriterAgent()._run_fallback(state)
    combined = "\n".join([sections.introduction, sections.experiments, sections.conclusion, sections.method])

    assert "### Experimental Setup" in sections.experiments
    assert "### Main Results" in sections.experiments
    assert "The contributions are organized as follows" in sections.introduction
    assert "Validation note" in sections.method
    assert "user should confirm" not in combined
    assert "The introduction should" not in combined
    assert "The experiments section should" not in combined
    assert "Current missing details" not in combined
    assert "to be refined" not in combined
    assert "final conclusion should" not in combined


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


def test_reviewer_flags_only_unresolved_bibliography_seeds():
    state = {
        "experiments": ExperimentSummary(),
        "innovations": [],
        "sections": DraftSections(),
        "bibliography": [
            CitationEntry(
                key="resolved",
                title="Resolved Paper",
                authors=["Ada Lovelace"],
                year="2024",
                doi="10.1234/resolved",
                note="Resolved by OpenAlex. Verify relevance before submission.",
            ),
            CitationEntry(
                key="seed",
                title="Representative work on whole slide images",
                authors=["Related work authors"],
                note="Seed related-work entry generated from project keywords.",
            ),
        ],
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)

    issue = next(finding.issue for finding in reviewed["review_findings"] if "Bibliography" in finding.issue)
    assert "1 unresolved seed" in issue
    assert issue.endswith("entries: seed.")


def test_reviewer_flags_related_work_threads_without_real_citations():
    state = {
        "experiments": ExperimentSummary(),
        "innovations": [],
        "sections": DraftSections(
            related_work=(
                "### Classic Thread\n"
                "This thread discusses prior survival prediction without a citation.\n\n"
                "### Recent Thread\n"
                "Recent work is cited with a resolved entry \\cite{resolved}.\n\n"
                "### Seed Thread\n"
                "This thread cites an unresolved generated seed \\cite{seed}.\n\n"
                "### Relation to the Proposed Method\n"
                "This positioning paragraph compares the proposed method to the cited threads."
            )
        ),
        "bibliography": [
            CitationEntry(
                key="resolved",
                title="Resolved Paper",
                authors=["Ada Lovelace"],
                year="2024",
                doi="10.1234/resolved",
            ),
            CitationEntry(
                key="seed",
                title="Representative work on whole slide images",
                authors=["Related work authors"],
                note="Seed related-work entry generated from project keywords.",
            ),
        ],
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)
    coverage = reviewed["artifacts"]["related_work_citation_coverage"]

    assert any("Related Work threads lack real citation coverage" in f.issue for f in reviewed["review_findings"])
    assert [item["thread"] for item in coverage if not item["covered_by_real_citation"]] == [
        "Classic Thread",
        "Seed Thread",
    ]
    relation = next(item for item in coverage if item["thread"] == "Relation to the Proposed Method")
    assert not relation["requires_citation"]


def test_reviewer_flags_unsupported_experiment_facts():
    state = {
        "experiments": ExperimentSummary(
            raw_preview=(
                "| Method | BLCA C-index | BRCA C-index |\n"
                "|---|---:|---:|\n"
                "| ProtoSurv baseline | 0.646 | 0.669 |\n"
                "| Ours | 0.671 | 0.691 |\n"
            ),
            datasets=["BLCA", "BRCA"],
            metrics=["C-INDEX"],
            observations=["Ours improves over baseline with average signed improvement +0.023."],
            missing_details=[],
        ),
        "innovations": [],
        "sections": DraftSections(
            experiments=(
                "We evaluate on BLCA, BRCA, and XYZ using C-index and AUC. "
                "The proposed method obtains 0.999 on XYZ."
            )
        ),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)
    consistency = {
        item["check"]: item
        for item in reviewed["artifacts"]["factual_consistency"]
    }

    assert "XYZ" in consistency["unsupported_datasets"]["values"]
    assert "AUC" in consistency["unsupported_metrics"]["values"]
    assert "0.999" in consistency["unsupported_experiment_numbers"]["values"]
    assert any("not supported by supplied evidence" in f.issue for f in reviewed["review_findings"])


def test_reviewer_accepts_supported_experiment_facts():
    state = {
        "experiments": ExperimentSummary(
            raw_preview=(
                "| Method | BLCA C-index | BRCA C-index |\n"
                "|---|---:|---:|\n"
                "| ProtoSurv baseline | 0.646 | 0.669 |\n"
                "| Ours | 0.671 | 0.691 |\n"
            ),
            datasets=["BLCA", "BRCA"],
            metrics=["C-INDEX"],
            observations=["Ours improves over baseline with average signed improvement +0.023."],
            missing_details=[],
        ),
        "innovations": [],
        "sections": DraftSections(
            experiments="We evaluate on BLCA and BRCA using C-index and obtain 0.671 on BLCA."
        ),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)

    assert all(
        item["status"] == "ok"
        for item in reviewed["artifacts"]["factual_consistency"]
    )
    assert not any("not supported by supplied evidence" in f.issue for f in reviewed["review_findings"])


def test_reviewer_flags_method_threads_without_innovation_support():
    state = {
        "experiments": ExperimentSummary(),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype calibration",
                motivation="The baseline uses static prototypes.",
                technical_idea="Calibrate prototypes with uncertainty-aware adaptation.",
                evidence=["Method notes mention adaptive prototype calibration."],
            )
        ],
        "sections": DraftSections(
            method=(
                "### Adaptive prototype calibration\n"
                "We calibrate prototypes before prediction.\n\n"
                "### Contrastive memory bank\n"
                "We add a memory bank that is not present in the accepted innovations."
            )
        ),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)
    consistency = {
        item["check"]: item
        for item in reviewed["artifacts"]["factual_consistency"]
    }

    assert consistency["unsupported_method_threads"]["values"] == ["Contrastive memory bank"]
    assert any("not supported by supplied evidence" in f.issue for f in reviewed["review_findings"])


def test_reviewer_flags_outline_language():
    state = {
        "experiments": ExperimentSummary(),
        "innovations": [],
        "sections": DraftSections(
            introduction="The introduction should open with the research problem.",
            experiments="Current missing details: dataset names are not explicit.",
        ),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)

    assert any("outline or procedural language" in finding.issue for finding in reviewed["review_findings"])
    assert reviewed["artifacts"]["outline_language_hits"] == ["introduction", "experiments"]


def test_draft_report_includes_related_work_citation_coverage(tmp_path):
    state = {
        "request": PaperRequest(project_name="coverage-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "related_work_citation_coverage": [
                {
                    "thread": "Classic Thread",
                    "requires_citation": True,
                    "citation_keys": [],
                    "real_citation_keys": [],
                    "covered_by_real_citation": False,
                },
                {
                    "thread": "Recent Thread",
                    "requires_citation": True,
                    "citation_keys": ["resolved"],
                    "real_citation_keys": ["resolved"],
                    "covered_by_real_citation": True,
                },
            ]
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Related Work Citation Coverage" in report
    assert "`Classic Thread`: missing real citation" in report
    assert "`Recent Thread`: covered; citations: resolved" in report


def test_draft_report_includes_factual_consistency(tmp_path):
    state = {
        "request": PaperRequest(project_name="consistency-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "factual_consistency": [
                {"check": "unsupported_datasets", "status": "needs_review", "values": ["XYZ"]},
                {"check": "unsupported_metrics", "status": "ok", "values": []},
            ]
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Factual Consistency" in report
    assert "`unsupported_datasets`: needs_review; values: XYZ" in report
    assert "`unsupported_metrics`: ok" in report


def test_llm_self_review_adds_unsupported_claim_finding(monkeypatch):
    monkeypatch.delenv("PAPER_AGENT_DISABLE_LLM_SELF_REVIEW", raising=False)
    client = FakeLLMClient(
        """
        {
          "unsupported_claims": [
            {
              "section": "experiments",
              "claim": "The method obtains 0.999 on XYZ.",
              "reason": "XYZ and 0.999 are absent from the supplied experiment table.",
              "evidence_needed": "Add an experiment row for XYZ with this value.",
              "severity": "major"
            }
          ],
          "section_quality_notes": ["Experiments contain a likely hallucinated number."]
        }
        """
    )
    state = {
        "request": PaperRequest(project_name="llm-review-demo", target_venue="TPAMI"),
        "sections": DraftSections(experiments="The method obtains 0.999 on XYZ."),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    reviewed = LLMSelfReviewAgent(llm_client=client).run(state)

    assert reviewed["artifacts"]["llm_self_review"]["mode"] == "llm"
    assert reviewed["artifacts"]["llm_self_review"]["unsupported_claims"][0]["section"] == "experiments"
    assert any("LLM self-review flagged unsupported claim" in finding.issue for finding in reviewed["review_findings"])
    assert client.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}


def test_cli_llm_self_review_smoke_reports_unavailable_without_llm(monkeypatch, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "1")
    monkeypatch.setattr("sys.argv", ["paper-agent", "llm-self-review-smoke"])

    cli_module.main()

    output = capsys.readouterr().out
    assert "LLM self-review mode: unavailable" in output
    assert "Unsupported claims: 0" in output


def test_llm_self_review_can_be_skipped_per_request(monkeypatch):
    monkeypatch.delenv("PAPER_AGENT_DISABLE_LLM_SELF_REVIEW", raising=False)
    client = FakeLLMClient('{"unsupported_claims": [], "section_quality_notes": []}')
    state = {
        "request": PaperRequest(
            project_name="llm-review-skip-demo",
            target_venue="TPAMI",
            skip_llm_self_review=True,
        ),
        "sections": DraftSections(abstract="A draft."),
        "artifacts": {},
    }

    reviewed = LLMSelfReviewAgent(llm_client=client).run(state)

    assert reviewed["artifacts"]["llm_self_review"]["mode"] == "disabled"
    assert client.calls == []


def test_cli_skip_llm_self_review_sets_request_flag(monkeypatch, tmp_path):
    captured = {}

    class FakeWorkflow:
        def run(self, request):
            captured["skip_llm_self_review"] = request.skip_llm_self_review
            return {
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI"),
                "bibliography": [],
                "artifacts": {"llm_self_review": {"mode": "disabled"}},
                "latex_output_path": tmp_path / "main.tex",
                "latex_project_dir": tmp_path,
                "review_findings": [],
            }

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        ["paper-agent", "demo", "--output", str(tmp_path / "out"), "--skip-llm-self-review"],
    )

    cli_module.main()

    assert captured["skip_llm_self_review"]


def test_api_returns_llm_self_review_summary(monkeypatch):
    class FakeWorkflow:
        def run(self, request):
            assert request.skip_llm_self_review
            return {
                "artifacts": {"llm_self_review": {"mode": "disabled"}},
                "review_findings": [],
                "final_markdown": "# Draft",
            }

    monkeypatch.setattr(api_module, "PaperWorkflow", FakeWorkflow)

    response = api_module.draft_paper(
        PaperRequest(project_name="api-demo", target_venue="TPAMI", skip_llm_self_review=True)
    )

    assert response["llm_self_review"] == {"mode": "disabled"}


def test_run_summary_reports_core_metrics(tmp_path):
    state = {
        "request": PaperRequest(project_name="summary-demo", target_venue="TPAMI"),
        "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
        "bibliography": [CitationEntry(key="paper", title="Paper")],
        "review_findings": [SimpleNamespace()],
        "latex_project_dir": tmp_path / "latex",
        "latex_output_path": tmp_path / "latex" / "main.tex",
        "latex_zip_path": tmp_path / "paper.zip",
        "artifacts": {
            "section_writer_mode": "fallback",
            "llm_self_review": {"mode": "disabled", "unsupported_claims": []},
            "reference_verification": {"resolved_count": 1, "unresolved_count": 2},
            "related_work_candidates": [{"title": "A"}],
            "latex_table_count": 3,
            "undefined_citation_keys": ["missing"],
            "draft_report_path": str(tmp_path / "latex" / "DRAFT_REPORT.md"),
        },
    }

    summary = cli_module._build_run_summary(state, tmp_path / "draft.md")

    assert summary["project_name"] == "summary-demo"
    assert summary["llm_self_review_mode"] == "disabled"
    assert summary["bibliography_entries"] == 1
    assert summary["reference_unresolved"] == 2
    assert summary["related_work_candidates"] == 1
    assert summary["outputs"]["markdown"].endswith("draft.md")


def test_cli_sample_hyper_protosurv_writes_showcase_artifacts(monkeypatch, tmp_path):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    experiment_path = tmp_path / "experiments.md"
    experiment_path.write_text("| Method | BLCA |\n|---|---:|\n| ours | 0.67 |\n", encoding="utf-8")
    latex_dir = tmp_path / "latex"
    latex_dir.mkdir()
    (latex_dir / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [CitationEntry(key="paper", title="Paper")],
                "artifacts": {
                    "llm_self_review": {"mode": "disabled"},
                    "latex_table_count": 1,
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    output_dir = tmp_path / "out"
    zip_path = tmp_path / "sample.zip"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "sample-hyper-protosurv",
            "--example-root",
            str(example_root),
            "--experiment-results",
            str(experiment_path),
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
        ],
    )

    cli_module.main()

    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert (output_dir / "draft.md").read_text(encoding="utf-8") == "# Draft"
    assert zip_path.exists()
    assert captured["request"].project_name == output_dir.name
    assert captured["request"].baseline_pdf_path.endswith("baseline.pdf")
    assert captured["request"].code_path.endswith("hyper-protosurv")
    assert captured["request"].skip_llm_self_review
    assert summary["llm_self_review_mode"] == "disabled"


def test_llm_self_review_records_bad_json_error(monkeypatch):
    monkeypatch.delenv("PAPER_AGENT_DISABLE_LLM_SELF_REVIEW", raising=False)
    state = {
        "request": PaperRequest(project_name="llm-review-error-demo", target_venue="TPAMI"),
        "sections": DraftSections(abstract="A draft."),
        "artifacts": {},
    }

    reviewed = LLMSelfReviewAgent(llm_client=FakeLLMClient("not json")).run(state)

    assert reviewed["artifacts"]["llm_self_review"]["mode"] == "error"
    assert "review_findings" not in reviewed


def test_llm_self_review_repairs_invalid_json_once(monkeypatch):
    monkeypatch.delenv("PAPER_AGENT_DISABLE_LLM_SELF_REVIEW", raising=False)
    client = FakeSequenceLLMClient(
        [
            '{"unsupported_claims": [{"section": "method", "claim": "Broken',
            """
            {
              "unsupported_claims": [
                {
                  "section": "method",
                  "claim": "Uses NVIDIA GPUs.",
                  "reason": "Hardware is absent from supplied evidence.",
                  "evidence_needed": "Add hardware details.",
                  "severity": "minor"
                }
              ],
              "section_quality_notes": []
            }
            """,
        ]
    )
    state = {
        "request": PaperRequest(project_name="llm-review-repair-demo", target_venue="TPAMI"),
        "sections": DraftSections(method="Uses NVIDIA GPUs."),
        "artifacts": {},
    }

    reviewed = LLMSelfReviewAgent(llm_client=client).run(state)

    assert reviewed["artifacts"]["llm_self_review"]["mode"] == "llm"
    assert reviewed["artifacts"]["llm_self_review"]["repaired_from_invalid_json"]
    assert reviewed["artifacts"]["llm_self_review"]["unsupported_claims"][0]["section"] == "method"
    assert len(client.calls) == 2


def test_draft_report_includes_llm_self_review(tmp_path):
    state = {
        "request": PaperRequest(project_name="llm-review-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "llm_self_review": {
                "mode": "llm",
                "unsupported_claims": [
                    {
                        "section": "experiments",
                        "claim": "The method obtains 0.999 on XYZ.",
                        "reason": "No such dataset or value was supplied.",
                        "evidence_needed": "Add the missing experiment result.",
                        "severity": "major",
                    }
                ],
                "section_quality_notes": ["Check experiment claims."],
            }
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## LLM Self Review" in report
    assert "experiments: The method obtains 0.999 on XYZ." in report
    assert "Evidence needed: Add the missing experiment result." in report


def test_reviewer_flags_method_missing_innovation():
    state = {
        "experiments": ExperimentSummary(),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype calibration",
                motivation="The baseline uses static prototypes.",
                technical_idea="Calibrate prototypes with uncertainty-aware adaptation.",
                evidence=["Method notes mention adaptive prototype calibration."],
            ),
            InnovationPoint(
                name="Innovation 2: Survival-aware objective",
                motivation="The baseline objective is incomplete.",
                technical_idea="Use a Cox survival term with reconstruction regularization.",
                evidence=["Repository exposes L_surv and L_rec."],
            ),
        ],
        "sections": DraftSections(method="### Adaptive prototype calibration\nWe calibrate prototypes before prediction."),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)

    assert any("omits innovation points" in finding.issue for finding in reviewed["review_findings"])
    assert reviewed["artifacts"]["innovation_traceability"][0]["mentioned_in_method"]
    assert not reviewed["artifacts"]["innovation_traceability"][1]["mentioned_in_method"]


def test_draft_report_includes_innovation_traceability(tmp_path):
    state = {
        "request": PaperRequest(project_name="traceability-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "innovation_traceability": [
                {
                    "name": "Innovation 1: Adaptive prototype calibration",
                    "mentioned_in_method": True,
                    "evidence_count": 1,
                    "evidence_preview": ["Method notes mention adaptive prototype calibration."],
                },
                {
                    "name": "Innovation 2: Survival-aware objective",
                    "mentioned_in_method": False,
                    "evidence_count": 1,
                    "evidence_preview": ["Repository exposes L_surv and L_rec."],
                },
            ]
        },
    }

    state = DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Innovation Traceability" in report
    assert "missing from Method" in report
    assert "Repository exposes L_surv and L_rec." in report


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
