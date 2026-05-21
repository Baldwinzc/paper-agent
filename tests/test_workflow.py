import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from paper_agent import api as api_module
from paper_agent import cli as cli_module
from paper_agent.config import LLMConfig
from paper_agent.export import zip_latex_project
from paper_agent.tables import extract_markdown_tables, markdown_tables_to_latex
from paper_agent.state import CitationEntry, InnovationPoint, PaperOutline, PaperRequest, VenueTemplate
from paper_agent.workflow import PaperWorkflow
from paper_agent.agents.baseline_reader import BaselineReaderAgent
from paper_agent.agents.bibliography import BibliographyAgent
from paper_agent.agents.code_baseline_comparison import CodeBaselineComparisonAgent
from paper_agent.agents.code_understanding import CodeUnderstandingAgent
from paper_agent.agents.evidence_guard import EvidenceGuardAgent
from paper_agent.agents.experiment_analyzer import ExperimentAnalyzerAgent
from paper_agent.agents.innovation_analyzer import InnovationAnalyzerAgent
from paper_agent.agents.latex_composer import LatexComposerAgent
from paper_agent.agents.llm_self_review import LLMSelfReviewAgent
from paper_agent.agents.paper_planner import PaperPlannerAgent
from paper_agent.agents.presentation_planner import PresentationPlannerAgent
from paper_agent.agents.draft_report import DraftReportAgent
from paper_agent.agents.reference_resolver import ReferenceResolverAgent
from paper_agent.agents.related_work_discovery import RelatedWorkDiscoveryAgent
from paper_agent.agents.reviewer import ReviewerAgent
from paper_agent.agents.section_writer import SectionWriterAgent
from paper_agent.agents.submission_package_validator import SubmissionPackageValidatorAgent
from paper_agent.agents.submission_readiness import SubmissionReadinessAgent
from paper_agent.state import (
    AblationEvidence,
    BaselineSummary,
    CodeSummary,
    DraftSections,
    ExperimentComparison,
    ExperimentSummary,
    ExperimentTableSummary,
    SensitivityEvidence,
    StatisticalTestEvidence,
)


os.environ.setdefault("PAPER_AGENT_DISABLE_TEMPLATE_FETCH", "1")
os.environ.setdefault("PAPER_AGENT_DISABLE_LLM", "1")
os.environ.setdefault("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "1")
os.environ.setdefault("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY", "1")


class FakeLLMClient:
    def __init__(self, content: str, *, model: str = "fake", usage: dict | None = None) -> None:
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.calls = []

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return SimpleNamespace(content=self.content, model=self.model, usage=dict(self.usage), raw={})


class FakeSequenceLLMClient:
    def __init__(self, contents: list[str], *, model: str = "fake", usage: dict | None = None) -> None:
        self.contents = contents
        self.model = model
        self.usage = usage or {}
        self.calls = []

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return SimpleNamespace(content=self.contents.pop(0), model=self.model, usage=dict(self.usage), raw={})


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


def test_baseline_reader_extracts_structured_sections(tmp_path):
    baseline_text = tmp_path / "baseline.txt"
    baseline_text.write_text(
        "ProtoSurv: Heterogeneous Graph Survival Prediction\n"
        "Abstract\n"
        "Survival prediction is a significant challenge in cancer management. "
        "However, current methods often neglect the fact that the contribution to prognosis differs with tissue types. "
        "In this paper, we propose ProtoSurv, a novel heterogeneous graph model for WSI survival prediction. "
        "We validate ProtoSurv across five different cancer types from TCGA.\n"
        "1\n"
        "Introduction\n"
        "Most current works are based on Multiple Instance Learning and lose structural information across tissues. "
        "Therefore, these methods struggle on prognostic prediction tasks.\n"
        "3\n"
        "Method\n"
        "The heterogeneous graph introduces a tissue category attribute to each node. "
        "The Structure View uses neighbor message passing, and the Histology View extracts prototypes from global features.\n"
        "4\n"
        "Experiments\n"
        "We conducted comprehensive evaluations on five public benchmark datasets: BRCA, LGG, LUAD, COAD and PAAD.\n",
        encoding="utf-8",
    )

    state = BaselineReaderAgent().run(
        {
            "request": PaperRequest(
                project_name="baseline-structure-demo",
                target_venue="TPAMI",
                baseline_pdf_path=str(baseline_text),
            )
        }
    )

    baseline = state["baseline"]
    assert "abstract" in baseline.structured_sections
    assert "introduction" in baseline.structured_sections
    assert "method" in baseline.structured_sections
    assert "experiments" in baseline.structured_sections
    assert "neglect" in baseline.problem.lower()
    assert "heterogeneous graph model" in baseline.method.lower()
    assert "TCGA" in baseline.experiments or "benchmark datasets" in baseline.experiments
    assert any("neglect" in limitation.lower() for limitation in baseline.limitations)
    assert "heterogeneous graph" in baseline.related_terms
    assert "prototype learning" in baseline.related_terms


def test_baseline_reader_extracts_numbered_references():
    text = (
        "References\n"
        "[5] Richard J Chen, Ming Y Lu, and Faisal Mahmood. Whole slide images are 2d point clouds: "
        "Context-aware survival prediction using patch-based graph convolutional networks. MICCAI, 2021.\n"
        "[18] Mobadersany Person. Predicting cancer outcomes from histology. Journal, 2018.\n"
    )

    references = BaselineReaderAgent()._extract_references(text)

    assert "5" in references
    assert "Whole slide images are 2d point clouds" in references["5"]
    assert "18" in references


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


def test_code_understanding_extracts_implementation_evidence(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "utils").mkdir()
    (tmp_path / "data_preparation").mkdir()
    (tmp_path / "models" / "model_protosurv_v1.py").write_text(
        "class LINKX_PROTO_HG:\n"
        "    def __init__(self):\n"
        "        self.hcon = HCoN(input_feat_x_dim=512)\n"
        "        self.proto_fusion_to_p = CrossAttention()\n"
        "        self.mean_pool_fusion_proj = nn.Linear(dim_proto * 2, dim_proto, bias=False)\n"
        "        self.risk_prediction_layer = nn.Linear(dim_proto, 1, bias=False)\n"
        "    def forward(self, data):\n"
        "        M_OT = data.prototypes\n"
        "        x_emb, m_emb = self.hcon(hx1, hx2, x0, hy1, hy2, M_OT, alpha=0.5, beta=0.5)\n"
        "        node_mean = torch.mean(x_context_batched, dim=1)\n"
        "        proto_mean = torch.mean(prototypes_q, dim=1)\n"
        "        S = self.risk_prediction_layer(h)\n"
        "        rec_loss = F.binary_cross_entropy_with_logits(logits, target_H)\n",
        encoding="utf-8",
    )
    (tmp_path / "utils" / "core_funcs.py").write_text(
        "loss = loss_surv + args.hcon_beta * hcon_rec_loss\n",
        encoding="utf-8",
    )
    (tmp_path / "data_preparation" / "hypergraph_construction_wb.py").write_text(
        "mask = (y_col != p_row)\n"
        "C[mask] *= np.exp(self.alpha)\n"
        "X_bar = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init)\n",
        encoding="utf-8",
    )

    state = CodeUnderstandingAgent().run(
        {
            "request": PaperRequest(
                project_name="code-evidence-demo",
                target_venue="TPAMI",
                code_path=str(tmp_path),
            )
        }
    )

    evidence = state["code"].implementation_evidence
    assert any("(BHE/HCoN module)" in item and "self.hcon" in item for item in evidence)
    assert any("(cross-attention fusion)" in item and "CrossAttention" in item for item in evidence)
    assert any("(mean-pool fusion head)" in item and "mean_pool_fusion_proj" in item for item in evidence)
    assert any("(survival risk head)" in item and "risk_prediction_layer" in item for item in evidence)
    assert any("(incidence reconstruction)" in item and "target_H" in item for item in evidence)
    assert any("(reconstruction objective)" in item and "hcon_rec_loss" in item for item in evidence)
    assert any("(OT/Wasserstein hypergraph construction)" in item for item in evidence)
    assert any("(cross-cluster cost mask)" in item for item in evidence)
    assert "implementation evidence snippets" in state["code"].summary


def test_code_baseline_comparison_generates_innovation_seeds():
    state = {
        "baseline": BaselineSummary(
            title="Prototype Graphs for Survival Prediction",
            method="The baseline uses heterogeneous graph prototypes for WSI survival prediction.",
            related_terms=["prototype learning", "survival prediction"],
        ),
        "code": CodeSummary(
            summary="Scanned method files.",
            implementation_evidence=[
                "models/model.py:14 (BHE/HCoN module) self.hcon = HCoN(input_feat_x_dim=512)",
                "data/hypergraph.py:20 (OT/Wasserstein hypergraph construction) X_bar = ot.lp.free_support_barycenter(...)",
                "models/model.py:52 (incidence reconstruction) rec_loss = F.binary_cross_entropy_with_logits(logits, target_H)",
            ],
            method_claims=[
                "Adaptive hypergraph prototype learning with optimal transport and incidence reconstruction."
            ],
        ),
        "artifacts": {},
    }

    CodeBaselineComparisonAgent().run(state)

    comparison = state["artifacts"]["code_baseline_comparison"]
    assert comparison["mode"] == "compared"
    assert "prototype learning" in comparison["overlapping_terms"]
    assert "hypergraph modeling" in comparison["code_only_terms"]
    assert "optimal transport geometry" in comparison["code_only_terms"]
    assert any("hypergraph" in seed for seed in comparison["innovation_seeds"])
    assert comparison["likely_method_shifts"]


def test_innovation_analyzer_prioritizes_code_baseline_seeds():
    state = {
        "request": PaperRequest(project_name="comparison-innovation-demo", target_venue="TPAMI"),
        "code": CodeSummary(
            summary="Scanned method files.",
            implementation_evidence=[
                "models/model.py:52 (incidence reconstruction) rec_loss = F.binary_cross_entropy_with_logits(logits, target_H)"
            ],
            method_claims=["A lower-priority implementation claim."],
        ),
        "artifacts": {
            "code_baseline_comparison": {
                "innovation_seeds": [
                    "Regularize learned hypergraph structure with incidence reconstruction."
                ],
                "likely_method_shifts": [
                    {
                        "technique": "incidence reconstruction",
                        "evidence": [
                            "models/model.py:52 (incidence reconstruction) rec_loss = F.binary_cross_entropy_with_logits(logits, target_H)"
                        ],
                    }
                ],
            }
        },
    }

    InnovationAnalyzerAgent().run(state)

    assert state["innovations"][0].technical_idea.startswith(
        "Regularize learned hypergraph structure"
    )
    assert any(
        "Innovation support: incidence reconstruction" in item
        for item in state["innovations"][0].evidence
    )


def test_innovation_evidence_includes_code_implementation_snippets():
    state = {
        "request": PaperRequest(project_name="innovation-evidence-demo", target_venue="TPAMI"),
        "code": CodeSummary(
            summary="Scanned method files.",
            implementation_evidence=[
                "models/model_protosurv_v1.py:140 (BHE/HCoN module) self.hcon = HCoN(...)"
            ],
            method_claims=["OT-driven adaptive hyperedges with bidirectional hyperedge convolution."],
        ),
    }

    InnovationAnalyzerAgent().run(state)

    evidence = state["innovations"][0].evidence
    assert "Scanned method files." in evidence
    assert any("(BHE/HCoN module)" in item for item in evidence)
    assert any("OT-driven adaptive hyperedges" in item for item in evidence)


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
    assert summary["inputs"]["experiment_results_source"] == "provided"
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
    assert state["experiments"].result_tables
    first_table = state["experiments"].result_tables[0]
    assert first_table.method == "Hyper-ProtoSurv ours"
    assert first_table.baseline == "ProtoSurv baseline"
    assert first_table.comparisons[0].dataset == "BLCA"
    assert first_table.comparisons[0].metric == "C-INDEX"
    assert first_table.comparisons[0].method_value == 0.671
    assert first_table.comparisons[0].baseline_value == 0.646
    assert round(first_table.comparisons[0].signed_improvement, 3) == 0.025
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
    comparisons = state["experiments"].result_tables[0].comparisons
    assert all(not item.higher_is_better for item in comparisons)
    assert round(comparisons[0].signed_improvement, 3) == 0.020


def test_experiment_analyzer_uses_nearby_metric_text_for_tables():
    raw = """
    ## Main Results

    Metric: C-index, higher is better.

    | Method | BLCA | BRCA |
    |---|---:|---:|
    | ProtoSurv baseline | 0.646 | 0.669 |
    | Hyper-ProtoSurv ours | 0.671 | 0.691 |

    Metric: IBS, lower is better.

    | Method | BLCA | BRCA |
    |---|---:|---:|
    | ProtoSurv baseline | 0.184 | 0.171 |
    | Hyper-ProtoSurv ours | 0.171 | 0.160 |
    """

    state = ExperimentAnalyzerAgent().run(
        {
            "request": PaperRequest(
                project_name="demo",
                target_venue="TPAMI",
                experiment_results=raw,
            )
        }
    )

    tables = state["experiments"].result_tables
    assert [table.metric for table in tables] == ["C-INDEX", "IBS"]
    assert all(item.higher_is_better for item in tables[0].comparisons)
    assert all(not item.higher_is_better for item in tables[1].comparisons)
    assert round(tables[1].comparisons[0].signed_improvement, 3) == 0.013


def test_experiment_analyzer_extracts_ablation_evidence():
    raw = """
    ## Ablation Results

    Metric: Average C-index across BLCA and BRCA.

    | Variant | Average C-index | Delta vs Full |
    |---|---:|---:|
    | Full Hyper-ProtoSurv | 0.690 | 0.000 |
    | w/o OT-driven adaptive hyperedges | 0.674 | -0.016 |
    | w/o bidirectional hyperedge update | 0.678 | -0.012 |
    | mean-pool fusion instead of cross-attention | 0.681 | -0.009 |
    | w/o L_rec | 0.672 | -0.018 |
    """

    state = ExperimentAnalyzerAgent().run(
        {
            "request": PaperRequest(
                project_name="demo",
                target_venue="TPAMI",
                experiment_results=raw,
            )
        }
    )

    evidence = state["experiments"].ablation_evidence
    assert len(evidence) == 4
    assert state["artifacts"]["experiment_ablation_evidence"]
    assert evidence[0].reference == "Full Hyper-ProtoSurv"
    assert evidence[0].variant == "w/o OT-driven adaptive hyperedges"
    assert evidence[0].metric == "C-INDEX"
    assert evidence[0].dataset == "Average"
    assert round(evidence[0].signed_drop, 3) == 0.016
    assert "adaptive hypergraph prototype learning" in evidence[0].supports
    assert any("Ablation evidence includes 4 component comparisons" in item for item in state["experiments"].observations)


def test_experiment_analyzer_extracts_sensitivity_and_statistical_tests():
    raw = """
    ## Sensitivity Analysis

    Metric: Average C-index.

    | lambda_rec | Average C-index |
    |---:|---:|
    | 0.1 | 0.681 |
    | 0.5 | 0.687 |
    | 1.0 | 0.690 |
    | 2.0 | 0.686 |

    ## Statistical Testing

    | Comparison | Metric | Test | p-value |
    |---|---|---|---:|
    | Hyper-ProtoSurv vs ProtoSurv | C-index | Wilcoxon signed-rank | 0.018 |
    """

    state = ExperimentAnalyzerAgent().run(
        {
            "request": PaperRequest(
                project_name="demo",
                target_venue="TPAMI",
                experiment_results=raw,
            )
        }
    )

    sensitivity = state["experiments"].sensitivity_evidence
    tests = state["experiments"].statistical_tests
    assert len(sensitivity) == 1
    assert sensitivity[0].parameter == "lambda_rec"
    assert sensitivity[0].metric == "C-INDEX"
    assert sensitivity[0].best_parameter_value == "1.0"
    assert sensitivity[0].best_metric_value == 0.690
    assert sensitivity[0].tested_values == ["0.1", "0.5", "1.0", "2.0"]
    assert sensitivity[0].metric_values == [0.681, 0.687, 0.690, 0.686]
    assert len(tests) == 1
    assert tests[0].comparison == "Hyper-ProtoSurv vs ProtoSurv"
    assert tests[0].metric == "C-INDEX"
    assert tests[0].test == "Wilcoxon signed-rank"
    assert tests[0].p_value_text == "p=0.018"
    assert tests[0].significant
    assert state["artifacts"]["experiment_sensitivity_evidence"]
    assert state["artifacts"]["experiment_statistical_tests"]
    assert any("Sensitivity analysis for lambda_rec" in item for item in state["experiments"].observations)
    assert any("Statistical test evidence includes 1 comparisons" in item for item in state["experiments"].observations)


def test_experiment_contract_warns_when_optional_tables_are_missing():
    raw = """
    ## Main Results

    Metric: C-index. Higher is better.

    | Method | BLCA C-index |
    |---|---:|
    | ProtoSurv baseline | 0.646 |
    | Hyper-ProtoSurv ours | 0.671 |
    """

    state = ExperimentAnalyzerAgent().run(
        {
            "request": PaperRequest(
                project_name="contract-demo",
                target_venue="TPAMI",
                experiment_results=raw,
            )
        }
    )

    contract = state["artifacts"]["experiment_contract"]
    assert contract["status"] == "needs_attention"
    assert not contract["errors"]
    assert contract["checks"]["result_tables"] == 1
    assert contract["checks"]["numeric_comparisons"] == 1
    assert any("Missing ablation table" in item for item in contract["warnings"])
    assert any("Missing statistical-test table" in item for item in contract["warnings"])


def test_hyper_protosurv_mock_example_covers_full_experiment_contract():
    raw = (Path(__file__).resolve().parents[1] / "examples" / "hyper_protosurv_mock_experiments.md").read_text(
        encoding="utf-8"
    )

    state = ExperimentAnalyzerAgent().run(
        {
            "request": PaperRequest(
                project_name="hyper-protosurv-mock",
                target_venue="TPAMI",
                experiment_results=raw,
            )
        }
    )

    assert state["experiments"].datasets == ["BLCA", "BRCA", "LGG", "LUAD", "UCEC"]
    assert len(state["experiments"].result_tables) == 2
    assert len(state["experiments"].ablation_evidence) == 4
    assert len(state["experiments"].sensitivity_evidence) == 1
    assert len(state["experiments"].statistical_tests) == 2
    assert state["experiments"].statistical_tests[0].significant
    assert not state["experiments"].missing_details
    assert state["artifacts"]["experiment_contract"]["status"] == "complete"


def test_section_writer_uses_structured_result_tables():
    raw = """
    | Method | BLCA C-index | BRCA C-index |
    |---|---:|---:|
    | ProtoSurv baseline | 0.646 | 0.669 |
    | Hyper-ProtoSurv ours | 0.671 | 0.691 |
    """
    state = ExperimentAnalyzerAgent().run(
        {
            "request": PaperRequest(
                project_name="demo",
                target_venue="TPAMI",
                experiment_results=raw,
            )
        }
    )

    sections = SectionWriterAgent()._run_fallback(
        {
            "request": PaperRequest(project_name="demo", target_venue="TPAMI"),
            "experiments": state["experiments"],
            "innovations": [],
            "artifacts": {},
        }
    )

    assert "0.671 vs 0.646" in sections.experiments
    assert "average signed improvement of +0.023" in sections.experiments
    assert "copied from the supplied experiment tables" in sections.experiments

    reviewed = ReviewerAgent().run(
        {
            "experiments": state["experiments"],
            "innovations": [],
            "sections": DraftSections(experiments=sections.experiments),
            "artifacts": {},
        }
    )
    consistency = {
        item["check"]: item
        for item in reviewed["artifacts"]["factual_consistency"]
    }
    assert consistency["unsupported_experiment_numbers"]["status"] == "ok"


def test_section_writer_uses_sensitivity_and_statistical_evidence():
    raw = """
    | Method | BLCA C-index |
    |---|---:|
    | ProtoSurv baseline | 0.646 |
    | Hyper-ProtoSurv ours | 0.671 |

    | lambda_rec | Average C-index |
    |---:|---:|
    | 0.5 | 0.687 |
    | 1.0 | 0.690 |

    | Comparison | Metric | Test | p-value |
    |---|---|---|---:|
    | Hyper-ProtoSurv vs ProtoSurv | C-index | Wilcoxon signed-rank | 0.018 |
    """
    state = ExperimentAnalyzerAgent().run(
        {
            "request": PaperRequest(
                project_name="demo",
                target_venue="TPAMI",
                experiment_results=raw,
            )
        }
    )

    sections = SectionWriterAgent()._run_fallback(
        {
            "request": PaperRequest(project_name="demo", target_venue="TPAMI"),
            "experiments": state["experiments"],
            "innovations": [],
            "artifacts": {},
        }
    )

    assert "### Sensitivity Analysis" in sections.experiments
    assert "lambda_rec is tested over 0.5, 1.0" in sections.experiments
    assert "### Statistical Testing" in sections.experiments
    assert "p=0.018" in sections.experiments


def test_section_writer_uses_ablation_evidence():
    experiments = ExperimentSummary(
        datasets=["BLCA"],
        metrics=["C-INDEX"],
        ablation_evidence=[
            AblationEvidence(
                table_caption="Ablation Results",
                dataset="Average",
                metric="C-INDEX",
                reference="Full Hyper-ProtoSurv",
                variant="w/o bidirectional hyperedge update",
                reference_value=0.690,
                variant_value=0.678,
                signed_drop=0.012,
                supports=["bidirectional hyperedge updates"],
            )
        ],
    )

    sections = SectionWriterAgent()._run_fallback(
        {
            "request": PaperRequest(project_name="demo", target_venue="TPAMI"),
            "experiments": experiments,
            "innovations": [],
            "artifacts": {},
        }
    )

    assert "### Ablation Evidence" in sections.experiments
    assert "w/o bidirectional hyperedge update" in sections.experiments
    assert "signed drop +0.012" in sections.experiments


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
            InnovationPoint(
                name="Innovation 4: Simplify the training objective",
                technical_idea="Simplify the training objective by removing unsupported legacy regularizers.",
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
    assert not any("simplify objective removing" in query for query in queries)
    assert not any("the legacy" in query for query in queries)


def test_bibliography_skips_compound_threads_covered_by_existing_terms():
    state = {
        "request": PaperRequest(
            project_name="citation-demo",
            target_venue="TPAMI",
            keywords=["whole-slide images", "survival prediction", "hypergraph learning"],
        ),
        "baseline": BaselineSummary(
            title="Baseline",
            related_terms=["prototype learning"],
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive hypergraph prototype learning",
                technical_idea="Adaptive hypergraph prototype learning with bidirectional updates.",
                motivation="The baseline uses static prototypes.",
            )
        ],
        "artifacts": {},
    }

    state = BibliographyAgent().run(state)

    queries = [entry.query for entry in state["bibliography"]]
    assert any("hypergraph learning" in query for query in queries)
    assert any("prototype learning" in query for query in queries)
    assert not any("hypergraph learning prototype learning" in query for query in queries)


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
    trace = state["artifacts"]["reference_resolution_trace"]
    assert trace
    assert trace[0]["status"] == "resolved"
    assert trace[0]["source"] == "openalex"
    assert trace[0]["doi"] == "10.1234/example"
    assert trace[0]["retained"]


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
    trace = state["artifacts"]["reference_resolution_trace"]
    assert any(not item["retained"] for item in trace)
    assert all(item["retained_key"] in state["artifacts"]["citation_keys"] for item in trace)


def test_related_work_discovery_adds_categorized_candidates(monkeypatch):
    def work(title, identifier, year, cited_by_count, referenced_works=None, authors=None):
        return {
            "id": f"https://openalex.org/{identifier}",
            "title": title,
            "doi": f"https://doi.org/10.1234/{identifier.lower()}",
            "publication_year": year,
            "authorships": [{"author": {"display_name": author}} for author in (authors or ["Ada Lovelace"])],
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
        if "Predicting cancer outcomes" in str(params.get("search", "")):
            return {
                "results": [
                    work("Unrelated software library", "WBAD", 2020, 999, authors=["Ada Lovelace"]),
                    work(
                        "Predicting cancer outcomes from histology",
                        "WMENTION",
                        2018,
                        850,
                        authors=["Mobadersany Person"],
                    ),
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
            structured_sections={
                "related_work": (
                    "Mobadersany et al. [18] proposed an end-to-end CNN method for "
                    "survival prediction in whole-slide histology images."
                )
            },
            references={
                "18": (
                    "Mobadersany Person. Predicting cancer outcomes from histology. "
                    "IEEE Transactions, 2018."
                )
            },
        ),
        "bibliography": [
            CitationEntry(
                key="baseline",
                title="Baseline Survival Paper",
                query="Baseline Survival Paper",
                authors=["Baseline authors"],
                note="Seed entry extracted from the provided baseline PDF; verify metadata before submission.",
            ),
            CitationEntry(
                key="wholeslideimages",
                title="Representative work on whole-slide images",
                query="whole-slide images survival prediction",
                authors=["Related work authors"],
                note="Seed related-work entry generated from project keywords; replace with real paper metadata.",
            ),
        ],
        "artifacts": {},
    }

    state = RelatedWorkDiscoveryAgent().run(state)

    categories = {item["category"] for item in state["artifacts"]["related_work_candidates"]}
    titles = [item["title"] for item in state["artifacts"]["related_work_candidates"]]
    candidate_by_category = {
        item["category"]: item for item in state["artifacts"]["related_work_candidates"] if isinstance(item, dict)
    }
    assert {"baseline_reference", "baseline_citing", "baseline_mentioned", "influential", "recent"} <= categories
    assert titles.count("New whole-slide survival prediction model") == 1
    assert "Predicting cancer outcomes from histology" in titles
    assert any("Predicting cancer outcomes" in item.get("query", "") for item in state["artifacts"]["related_work_candidates"])
    assert state["artifacts"]["related_work_field_query"] == "whole slide images survival prediction"
    assert state["artifacts"]["related_work_baseline_mentioned_queries"] == [
        "Mobadersany | Predicting cancer outcomes from histology"
    ]
    assert candidate_by_category["baseline_reference"]["discovery_path_label"] == "baseline reference list"
    assert candidate_by_category["baseline_reference"]["source_query"].startswith("baseline references of Baseline Survival Paper")
    assert candidate_by_category["baseline_citing"]["discovery_path_label"] == "papers citing the baseline"
    assert candidate_by_category["baseline_citing"]["source_query"].startswith("papers citing Baseline Survival Paper")
    assert candidate_by_category["influential"]["source_query"] == "whole slide images survival prediction"
    assert candidate_by_category["recent"]["source_query"] == "whole slide images survival prediction"
    assert candidate_by_category["baseline_mentioned"]["discovery_path_label"] == "baseline related-work mention"
    assert any(entry.title == "Classic survival analysis for whole-slide images" for entry in state["bibliography"])
    assert "wholeslideimages" not in [entry.key for entry in state["bibliography"]]
    assert state["artifacts"]["reference_pruned_seed_keys"] == ["wholeslideimages"]
    assert state["artifacts"]["citation_keys"]
    verification = state["artifacts"]["reference_verification"]
    assert verification["resolved_count"] == len(state["artifacts"]["related_work_candidates"]) + 1
    assert verification["unresolved_seed_keys"] == []
    assert "baseline" in verification["resolved_keys"]
    assert verification["needs_manual_check_keys"] == verification["resolved_keys"]


def test_related_work_discovery_extracts_baseline_mentioned_queries():
    baseline = BaselineSummary(
        structured_sections={
            "related_work": (
                "Mobadersany et al. [18] proposed an end-to-end CNN method for processing "
                "manually annotated ROIs in whole-slide histology survival prediction. "
                "Chen et al. [5] employed a graph convolutional network for context-aware WSI modeling."
            )
        },
        references={
            "18": "Mobadersany Person. Predicting cancer outcomes from histology. Journal, 2018.",
            "5": (
                "Chen Person. Whole slide images are 2d point clouds: Context-aware survival "
                "prediction using patch-based graph convolutional networks. MICCAI, 2021."
            ),
        },
    )

    queries = RelatedWorkDiscoveryAgent()._mentioned_work_queries(baseline)

    assert any(query.startswith("Mobadersany |") and "Predicting cancer outcomes" in query for query in queries)
    assert any(query.startswith("Chen |") and "2d point clouds" in query for query in queries)


def test_related_work_discovery_records_error_details_with_query_context(monkeypatch):
    def work(title, identifier, year, cited_by_count, referenced_works=None, authors=None):
        return {
            "id": f"https://openalex.org/{identifier}",
            "title": title,
            "doi": f"https://doi.org/10.1234/{identifier.lower()}",
            "publication_year": year,
            "authorships": [{"author": {"display_name": author}} for author in (authors or ["Ada Lovelace"])],
            "primary_location": {"source": {"display_name": "IEEE Transactions"}},
            "ids": {"openalex": f"https://openalex.org/{identifier}"},
            "referenced_works": referenced_works or [],
            "cited_by_count": cited_by_count,
        }

    def fake_query(self, params):
        if params.get("search") == "Baseline Survival Paper":
            return {"results": [work("Baseline Survival Paper", "WBASE", 2024, 42)]}
        if params.get("sort") == "cited_by_count:desc":
            raise RuntimeError("openalex quota exceeded for cited_by_count")
        return {"results": []}

    monkeypatch.setenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "0")
    monkeypatch.setenv("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY", "0")
    monkeypatch.setattr(RelatedWorkDiscoveryAgent, "_query_openalex", fake_query)
    state = {
        "request": PaperRequest(
            project_name="error-demo",
            target_venue="TPAMI",
            keywords=["whole-slide images", "survival prediction"],
        ),
        "baseline": BaselineSummary(title="Baseline Survival Paper"),
        "bibliography": [],
        "artifacts": {},
    }

    state = RelatedWorkDiscoveryAgent().run(state)

    details = state["artifacts"]["related_work_discovery_error_details"]
    assert details[0]["source"] == "influential_search"
    assert details[0]["query"] == "whole slide images survival prediction"
    assert details[0]["sort"] == "cited_by_count:desc"
    assert "quota exceeded" in details[0]["error"]


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
                {
                    "key": "mentionedpaper",
                    "category": "baseline_mentioned",
                    "title": "Mentioned paper",
                },
            ],
        },
    }

    sections = SectionWriterAgent()._run_fallback(state)

    assert r"\cite{classicpaper}" in sections.related_work
    assert r"\cite{recentpaper}" in sections.related_work
    assert r"\cite{mentionedpaper}" in sections.related_work
    assert "### Baseline Lineage" in sections.related_work
    assert "### Recent Developments" in sections.related_work
    assert "Classic paper" in sections.related_work
    assert "should be discussed" not in sections.related_work
    assert "should be positioned" not in sections.related_work


def test_cli_related_work_doctor_writes_summary_and_report(monkeypatch, tmp_path, capsys):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    output_dir = tmp_path / "related-work-doctor"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    captured = {}
    monkeypatch.delenv("PAPER_AGENT_DISABLE_TEMPLATE_FETCH", raising=False)
    monkeypatch.delenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", raising=False)
    monkeypatch.delenv("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY", raising=False)

    def fake_pipeline(request):
        captured["request"] = request
        captured["disable_reference_resolve"] = os.getenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE")
        captured["disable_related_work_discovery"] = os.getenv("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY")
        return {
            "request": request,
            "bibliography": [
                CitationEntry(
                    key="baseline",
                    title="Baseline Survival Paper",
                    query="Baseline Survival Paper",
                    authors=["Ada Lovelace"],
                    year="2024",
                    doi="10.1234/base",
                    note="Resolved baseline reference.",
                )
            ],
            "artifacts": {
                "reference_resolver_mode": "openalex",
                "reference_verification": {
                    "resolved_count": 2,
                    "unresolved_count": 1,
                    "resolved_keys": ["baseline", "classicpaper"],
                    "unresolved_seed_keys": ["seedref"],
                    "needs_manual_check_keys": ["baseline", "classicpaper"],
                },
                "reference_resolution_trace": [{"seed_key": "baseline"}],
                "reference_resolver_errors": {"seedref": "timeout"},
                "citation_keys": ["baseline", "classicpaper", "recentpaper"],
                "related_work_discovery_mode": "openalex",
                "related_work_field_query": "whole slide images survival prediction",
                "related_work_baseline_mentioned_queries": [
                    "Mobadersany | Predicting cancer outcomes from histology"
                ],
                "related_work_candidates": [
                    {
                        "key": "classicpaper",
                        "category": "baseline_reference",
                        "title": "Classic survival analysis for whole-slide images",
                        "year": "2018",
                        "query": "Classic survival analysis for whole-slide images",
                    },
                    {
                        "key": "recentpaper",
                        "category": "recent",
                        "title": "Recent whole-slide survival model",
                        "year": "2026",
                        "query": "Recent whole-slide survival model",
                    },
                ],
                "related_work_discovery_errors": {"baseline_mentions": "timeout"},
                "related_work_discovery_error_details": [
                    {
                        "source": "baseline_mentions",
                        "query": "Mobadersany | Predicting cancer outcomes from histology",
                        "sort": "relevance_score:desc",
                        "error": "timeout",
                    }
                ],
            },
        }

    monkeypatch.setattr(cli_module, "_run_related_work_doctor_pipeline", fake_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "related-work-doctor",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--target-venue",
            "TPAMI",
            "--keyword",
            "whole-slide images",
            "--keyword",
            "survival prediction",
            "--output-dir",
            str(output_dir),
            "--online",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RELATED_WORK_DOCTOR_SUMMARY.json").read_text(encoding="utf-8"))
    report = (output_dir / "RELATED_WORK_DOCTOR_REPORT.md").read_text(encoding="utf-8")
    assert "Related-work doctor summary written to" in output
    assert "Related-work doctor report written to" in output
    assert "related-work-doctor completed with warnings." in output
    assert captured["disable_reference_resolve"] == "0"
    assert captured["disable_related_work_discovery"] == "0"
    assert captured["request"].baseline_pdf_path == str(baseline_pdf)
    assert captured["request"].code_path == str(code_dir)
    assert captured["request"].target_venue == "TPAMI"
    assert summary["schema_version"] == "related-work-doctor/v1"
    assert summary["status"] == "warn"
    assert summary["reference_resolver_error_count"] == 1
    assert summary["reference_resolver_error_sources"] == ["seedref"]
    assert summary["related_work_discovery_mode"] == "openalex"
    assert summary["related_work_candidates"] == 2
    assert summary["related_work_baseline_lineage_candidates"] == 1
    assert summary["related_work_recent_candidates"] == 1
    assert summary["related_work_field_query"] == "whole slide images survival prediction"
    assert summary["related_work_baseline_mentioned_queries"] == [
        "Mobadersany | Predicting cancer outcomes from histology"
    ]
    assert [action["category"] for action in summary["next_actions"]] == [
        "reference_retry",
        "related_work_retry",
        "reference_seed_review",
    ]
    assert "## Discovery Evidence" in report
    assert "## Candidate Preview" in report
    assert "whole slide images survival prediction" in report
    assert "Mobadersany | Predicting cancer outcomes from histology" in report
    assert "source=baseline_mentions; query=Mobadersany | Predicting cancer outcomes from histology; sort=relevance_score:desc; error=timeout" in report
    assert "reference_retry" in report


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


def test_section_writer_evidence_text_prefers_diverse_implementation_labels():
    evidence_text = SectionWriterAgent()._evidence_text(
        [
            "Scanned 62 files. Likely method-bearing files: train.py.",
            "data_preparation/hypergraph.py:10 (OT/Wasserstein hypergraph construction) X_bar = ...",
            "data_preparation/hypergraph.py:12 (OT/Wasserstein hypergraph construction) C = ...",
            "models/HCoN/model.py:8 (BHE/HCoN module) class HCoN(nn.Module)",
            "models/model.py:185 (cross-attention fusion) self.proto_fusion_to_p = ProtoFusion(...)",
            "models/model.py:192 (survival risk head) self.risk_prediction_layer = nn.Linear(...)",
        ]
    )

    assert "Scanned 62 files" not in evidence_text
    assert "OT/Wasserstein" in evidence_text
    assert "BHE/HCoN" in evidence_text
    assert "cross-attention fusion" in evidence_text
    assert "survival risk head" in evidence_text


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


def test_reviewer_treats_manual_novelty_confirmation_as_warning():
    state = {
        "experiments": ExperimentSummary(result_tables=[ExperimentTableSummary(caption="Main")]),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive hypergraph prototypes",
                motivation="Baseline limitation.",
                technical_idea="Adaptive hypergraph prototypes.",
                evidence=["Repository and experiment evidence."],
                risk="Needs manual confirmation that the contribution is novel and not overclaimed.",
            )
        ],
        "sections": DraftSections(
            method="### Innovation 1: Adaptive hypergraph prototypes\nAdaptive hypergraph prototypes are used."
        ),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)

    finding = next(finding for finding in reviewed["review_findings"] if "novelty confirmation" in finding.issue)
    assert finding.severity == "minor"
    assert not any(finding.severity == "major" for finding in reviewed["review_findings"])


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


def test_reviewer_ignores_tme_and_generic_accuracy_in_background_text():
    experiments = ExperimentSummary(datasets=["BLCA", "BRCA"], metrics=["C-INDEX"])
    text = (
        "The tumor microenvironment (TME) provides rich prognostic context. "
        "Manual diagnosis can vary in accuracy across observers before any model evaluation."
    )

    reviewer = ReviewerAgent()

    assert reviewer._unsupported_datasets(text, experiments) == []
    assert reviewer._unsupported_metrics(text, experiments) == []


def test_reviewer_treats_ibs_as_brier_score_evidence():
    state = {
        "experiments": ExperimentSummary(
            raw_preview="| Method | BLCA IBS |\n|---|---:|\n| Ours | 0.171 |\n",
            datasets=["BLCA"],
            metrics=["IBS"],
        ),
        "innovations": [],
        "sections": DraftSections(
            experiments="We evaluate on BLCA using the integrated Brier score (IBS)."
        ),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)
    consistency = {
        item["check"]: item
        for item in reviewed["artifacts"]["factual_consistency"]
    }

    assert consistency["unsupported_metrics"]["status"] == "ok"


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


def test_reviewer_accepts_method_thread_supported_by_ablation_evidence():
    state = {
        "experiments": ExperimentSummary(
            ablation_evidence=[
                AblationEvidence(
                    table_caption="Ablation Results",
                    dataset="Average",
                    metric="C-INDEX",
                    reference="Full Hyper-ProtoSurv",
                    variant="mean-pool fusion instead of cross-attention",
                    reference_value=0.690,
                    variant_value=0.681,
                    signed_drop=0.009,
                    supports=["cross-attention fusion"],
                )
            ]
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive hypergraph prototype learning",
                motivation="The baseline uses static prototypes.",
                technical_idea="Construct adaptive hyperedges from prototype geometry.",
                evidence=["Code constructs adaptive hyperedges."],
            )
        ],
        "sections": DraftSections(
            method=(
                "### Adaptive hypergraph prototype learning\n"
                "The method constructs adaptive hyperedges.\n\n"
                "### Cross-Attention Prototype Fusion and Survival Prediction\n"
                "The model uses cross-attention fusion before survival prediction."
            )
        ),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)
    consistency = {
        item["check"]: item
        for item in reviewed["artifacts"]["factual_consistency"]
    }

    assert consistency["unsupported_method_threads"]["status"] == "ok"


def test_reviewer_accepts_method_thread_supported_by_code_evidence():
    state = {
        "experiments": ExperimentSummary(),
        "code": CodeSummary(
            likely_method_files=["data_preparation/hypergraph_construction_wb.py"],
            implementation_evidence=[
                "data_preparation/hypergraph_construction_wb.py:146 "
                "(OT/Wasserstein hypergraph construction) X_bar = ot.lp.free_support_barycenter("
            ],
            method_claims=["Construct heterogeneous hypergraphs from prototype hyperedges."],
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype geometry",
                motivation="The baseline uses static prototypes.",
                technical_idea="Construct adaptive prototype geometry with optimal-transport evidence.",
                evidence=["Wasserstein barycenter construction is implemented."],
            )
        ],
        "sections": DraftSections(
            method=(
                "### Adaptive prototype geometry\n"
                "The method constructs adaptive prototypes.\n\n"
                "### Heterogeneous Hypergraph Construction\n"
                "Nodes correspond to tissue patches and hyperedges connect patches with prototypes."
            )
        ),
        "artifacts": {
            "code_baseline_comparison": {
                "code_only_terms": ["hypergraph modeling", "optimal transport geometry"],
                "innovation_seeds": [
                    "Introduce hypergraph structure modeling for higher-order tissue and prototype relations."
                ],
                "likely_method_shifts": [
                    {
                        "technique": "hypergraph modeling",
                        "evidence": ["Repository evidence supports hypergraph construction."],
                    }
                ],
            }
        },
    }

    reviewed = ReviewerAgent().run(state)
    consistency = {
        item["check"]: item
        for item in reviewed["artifacts"]["factual_consistency"]
    }

    assert consistency["unsupported_method_threads"]["status"] == "ok"
    assert not any("unsupported_method_threads" in f.issue for f in reviewed["review_findings"])


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


def test_submission_readiness_scores_reviewable_draft(tmp_path):
    state = {
        "request": PaperRequest(project_name="readiness-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline", problem="Problem", method="Method"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(
            datasets=["BLCA"],
            metrics=["C-INDEX"],
            result_tables=[
                ExperimentTableSummary(
                    baseline="baseline",
                    comparisons=[
                        ExperimentComparison(
                            dataset="BLCA",
                            metric="C-INDEX",
                            method="ours",
                            baseline="baseline",
                            method_value=0.671,
                            baseline_value=0.646,
                            signed_improvement=0.025,
                            improved=True,
                        )
                    ],
                )
            ],
        ),
        "sections": DraftSections(
            abstract="A" * 120,
            introduction="I" * 120,
            related_work="R" * 120,
            method="M" * 120,
            experiments="E" * 120,
            conclusion="C" * 120,
        ),
        "bibliography": [CitationEntry(key="paper", title="Paper", authors=["Ada"], year="2024")],
        "venue_template": VenueTemplate(venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "latex_output_path": tmp_path / "main.tex",
        "latex_zip_path": tmp_path / "paper.zip",
        "review_findings": [],
        "artifacts": {
            "reference_verification": {"resolved_count": 1, "unresolved_count": 0},
            "factual_consistency": [
                {"check": "unsupported_datasets", "status": "ok", "values": []},
            ],
            "related_work_citation_coverage": [
                {
                    "thread": "Classic Thread",
                    "requires_citation": True,
                    "covered_by_real_citation": True,
                }
            ],
        },
    }
    state = SubmissionReadinessAgent().run(state)

    readiness = state["artifacts"]["submission_readiness"]
    assert readiness["overall_score"] >= 85
    assert readiness["status"] == "reviewable"
    assert not readiness["blocking_items"]
    assert "final human pass" in readiness["action_items"][0]


def test_submission_readiness_flags_blocking_evidence_gaps():
    state = {
        "request": PaperRequest(project_name="readiness-gap-demo", target_venue="TPAMI"),
        "experiments": ExperimentSummary(
            missing_details=["Dataset names are not explicit."],
        ),
        "sections": DraftSections(abstract="short"),
        "review_findings": [
            SimpleNamespace(
                severity="major",
                issue="Experiment section lacks required details.",
            )
        ],
        "artifacts": {
            "undefined_citation_keys": ["missing"],
            "reference_resolver_mode": "disabled",
        },
    }

    state = SubmissionReadinessAgent().run(state)

    readiness = state["artifacts"]["submission_readiness"]
    assert readiness["status"] == "needs_evidence"
    assert any("Experiment details are incomplete" in item for item in readiness["blocking_items"])
    assert any("Undefined citation keys" in item for item in readiness["blocking_items"])
    assert readiness["overall_score"] < 70


def test_submission_readiness_blocks_synthetic_experiment_evidence():
    state = {
        "request": PaperRequest(
            project_name="readiness-mock-demo",
            target_venue="TPAMI",
            experiment_results=(
                "This file is synthetic mock data for pipeline testing only.\n\n"
                "| Method | BLCA C-index |\n"
                "|---|---:|\n"
                "| Baseline | 0.640 |\n"
                "| Ours | 0.671 |\n"
            ),
        ),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(
            datasets=["BLCA"],
            metrics=["C-INDEX"],
            result_tables=[
                ExperimentTableSummary(
                    baseline="Baseline",
                    comparisons=[
                        ExperimentComparison(
                            dataset="BLCA",
                            metric="C-INDEX",
                            method="Ours",
                            baseline="Baseline",
                            method_value=0.671,
                            baseline_value=0.640,
                            signed_improvement=0.031,
                            improved=True,
                        )
                    ],
                )
            ],
        ),
        "sections": DraftSections(
            abstract="A" * 120,
            introduction="I" * 120,
            related_work="R" * 120,
            method="M" * 120,
            experiments="E" * 120,
            conclusion="C" * 120,
        ),
        "bibliography": [CitationEntry(key="paper", title="Paper", authors=["Ada"], year="2024")],
        "venue_template": VenueTemplate(venue="TPAMI"),
        "review_findings": [],
        "artifacts": {"reference_verification": {"resolved_count": 1, "unresolved_count": 0}},
    }

    state = SubmissionReadinessAgent().run(state)

    readiness = state["artifacts"]["submission_readiness"]
    assert state["artifacts"]["experiment_evidence_kind"] == "synthetic_mock"
    assert readiness["status"] == "needs_evidence"
    assert any("synthetic/mock" in item for item in readiness["blocking_items"])


def test_draft_report_includes_submission_readiness(tmp_path):
    state = {
        "request": PaperRequest(project_name="readiness-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "submission_readiness": {
                "overall_score": 88,
                "status": "reviewable",
                "scores": {
                    "evidence_grounding": 90,
                    "writing_completeness": 85,
                    "citation_readiness": 88,
                    "venue_package": 100,
                },
                "blocking_items": [],
                "action_items": ["Perform a final human pass."],
            }
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Submission Readiness" in report
    assert "- Status: reviewable" in report
    assert "- Overall score: 88/100" in report
    assert "Evidence Grounding: 90/100" in report
    assert "Perform a final human pass." in report


def test_draft_report_includes_reference_resolution_trace(tmp_path):
    state = {
        "request": PaperRequest(project_name="reference-trace-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "bibliography": [
            CitationEntry(
                key="resolved",
                title="Resolved Paper",
                authors=["Ada Lovelace"],
                year="2024",
                doi="10.1234/resolved",
            )
        ],
        "artifacts": {
            "reference_verification": {
                "resolved_count": 1,
                "unresolved_count": 0,
                "resolved_keys": ["resolved"],
                "unresolved_seed_keys": [],
            },
            "reference_resolution_trace": [
                {
                    "key": "seed",
                    "query": "whole slide image survival prediction",
                    "resolved_title": "Resolved Paper",
                    "status": "resolved",
                    "source": "openalex",
                    "doi": "10.1234/resolved",
                    "retained": False,
                    "retained_key": "resolved",
                }
            ],
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "Reference resolution trace:" in report
    assert "`seed`: resolved via openalex; merged into `resolved`; doi: 10.1234/resolved" in report
    assert "Query: whole slide image survival prediction" in report


def test_draft_report_includes_ablation_evidence(tmp_path):
    state = {
        "request": PaperRequest(project_name="ablation-report-demo", target_venue="TPAMI"),
        "experiments": ExperimentSummary(
            ablation_evidence=[
                AblationEvidence(
                    table_caption="Ablation Results",
                    dataset="Average",
                    metric="C-INDEX",
                    reference="Full Hyper-ProtoSurv",
                    variant="w/o L_rec",
                    reference_value=0.690,
                    variant_value=0.672,
                    signed_drop=0.018,
                    supports=["reconstruction regularization"],
                )
            ]
        ),
        "latex_project_dir": tmp_path,
        "artifacts": {},
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Ablation Evidence" in report
    assert "w/o L_rec" in report
    assert "reconstruction regularization" in report


def test_draft_report_includes_sensitivity_and_statistical_evidence(tmp_path):
    state = {
        "request": PaperRequest(project_name="sensitivity-report-demo", target_venue="TPAMI"),
        "experiments": ExperimentSummary(
            sensitivity_evidence=[
                SensitivityEvidence(
                    table_caption="Sensitivity Analysis",
                    parameter="lambda_rec",
                    dataset="Average",
                    metric="C-INDEX",
                    best_parameter_value="1.0",
                    best_metric_value=0.690,
                    worst_metric_value=0.681,
                    tested_values=["0.1", "0.5", "1.0"],
                )
            ],
            statistical_tests=[
                StatisticalTestEvidence(
                    table_caption="Statistical Testing",
                    comparison="Hyper-ProtoSurv vs ProtoSurv",
                    metric="C-INDEX",
                    test="Wilcoxon signed-rank",
                    p_value=0.018,
                    p_value_text="p=0.018",
                    significant=True,
                )
            ],
        ),
        "latex_project_dir": tmp_path,
        "artifacts": {},
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Sensitivity Evidence" in report
    assert "lambda_rec" in report
    assert "best value 1.0 -> 0.690" in report
    assert "## Statistical Test Evidence" in report
    assert "Hyper-ProtoSurv vs ProtoSurv" in report
    assert "p=0.018" in report


def test_draft_report_includes_llm_section_successes(tmp_path):
    state = {
        "request": PaperRequest(project_name="llm-section-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "section_writer_mode": "partial_llm",
            "section_writer_llm_successes": ["abstract", "method"],
            "section_writer_repaired_sections": ["method"],
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "- LLM-written sections: 2" in report
    assert "## LLM Section Drafting" in report
    assert "Successful sections: abstract, method" in report
    assert "Repaired sections: method" in report


def test_draft_report_includes_baseline_evidence(tmp_path):
    state = {
        "request": PaperRequest(project_name="baseline-report-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(
            title="ProtoSurv",
            problem="Current methods neglect tissue-type contributions.",
            method="ProtoSurv is a heterogeneous graph model.",
            experiments="Evaluated on TCGA cohorts.",
            limitations=["Current methods neglect tissue-type contributions."],
            structured_sections={"abstract": "A", "method": "M", "experiments": "E"},
        ),
        "latex_project_dir": tmp_path,
        "artifacts": {},
        "bibliography": [],
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Baseline Evidence" in report
    assert "- Title: ProtoSurv" in report
    assert "- Method: ProtoSurv is a heterogeneous graph model." in report
    assert "- Structured sections: abstract, method, experiments" in report


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
    assert reviewed["artifacts"]["llm_self_review"]["unsupported_claims"] == []
    assert reviewed["artifacts"]["llm_self_review"]["auto_revisions"][0]["section"] == "experiments"
    assert reviewed["artifacts"]["llm_self_review"]["auto_revised_claims"][0]["section"] == "experiments"
    assert reviewed["sections"].experiments == ""
    assert reviewed.get("review_findings", []) == []
    assert client.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}
    payload = json.loads(client.calls[0]["messages"][1].content)
    assert any("draft datasets" in rule for rule in payload["hard_rules"])


def test_llm_self_review_keeps_finding_when_claim_cannot_be_located(monkeypatch):
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
          "section_quality_notes": []
        }
        """
    )
    state = {
        "request": PaperRequest(project_name="llm-review-unlocated-demo", target_venue="TPAMI"),
        "sections": DraftSections(experiments="The reported TCGA results are summarized in Table 1."),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    reviewed = LLMSelfReviewAgent(llm_client=client).run(state)

    assert reviewed["artifacts"]["llm_self_review"]["auto_revisions"] == []
    assert reviewed["artifacts"]["llm_self_review"]["unsupported_claims"][0]["section"] == "experiments"
    assert any("LLM self-review flagged unsupported claim" in finding.issue for finding in reviewed["review_findings"])


def test_llm_self_review_rewrites_unlocated_claim_with_llm(monkeypatch):
    monkeypatch.delenv("PAPER_AGENT_DISABLE_LLM_SELF_REVIEW", raising=False)
    monkeypatch.delenv("PAPER_AGENT_DISABLE_LLM_SELF_REWRITE", raising=False)
    client = FakeSequenceLLMClient(
        [
            """
            {
              "unsupported_claims": [
                {
                  "section": "experiments",
                  "claim": "The model obtains 0.999 on XYZ.",
                  "reason": "XYZ and 0.999 are absent from the supplied experiment table.",
                  "evidence_needed": "Add an experiment row for XYZ with this value.",
                  "severity": "major"
                }
              ],
              "section_quality_notes": []
            }
            """,
            """
            {
              "section_revisions": [
                {
                  "section": "experiments",
                  "revised_text": "The experiments summarize the supplied BLCA C-index evidence and avoid claims about unreported cohorts.",
                  "rationale": "Removed the unsupported XYZ performance claim while preserving the grounded experiment scope."
                }
              ]
            }
            """,
        ]
    )
    state = {
        "request": PaperRequest(project_name="llm-review-rewrite-demo", target_venue="TPAMI"),
        "sections": DraftSections(
            experiments=(
                "The evaluation suggests unusually broad reliability beyond the reported TCGA cohorts. "
                "The reported TCGA results are summarized in Table 1."
            )
        ),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    reviewed = LLMSelfReviewAgent(llm_client=client).run(state)

    review = reviewed["artifacts"]["llm_self_review"]
    assert review["unsupported_claims"] == []
    assert review["auto_revisions"][0]["action"] == "llm_rewrite_section"
    assert review["auto_revised_claims"][0]["revision_action"] == "llm_rewrite_section"
    assert reviewed["sections"].experiments.startswith("The experiments summarize")
    assert reviewed.get("review_findings", []) == []
    assert len(client.calls) == 2
    rewrite_payload = json.loads(client.calls[1]["messages"][1].content)
    assert rewrite_payload["output_schema"]["section_revisions"]


def test_llm_self_review_filters_claims_it_marks_supported(monkeypatch):
    monkeypatch.delenv("PAPER_AGENT_DISABLE_LLM_SELF_REVIEW", raising=False)
    client = FakeLLMClient(
        """
        {
          "unsupported_claims": [
            {
              "section": "method",
              "claim": "The objective combines Cox loss and reconstruction loss.",
              "reason": "The evidence shows this objective in configs/protosurv.yml. Supported.",
              "evidence_needed": "N/A",
              "severity": "minor"
            },
            {
              "section": "abstract",
              "claim": "Quantitative evaluation is pending and will be reported once full results are available.",
              "reason": "This is a placeholder statement, not a factual claim.",
              "evidence_needed": "Complete experimental results.",
              "severity": "minor"
            },
            {
              "section": "introduction",
              "claim": "The central claim reserves empirical improvement claims for verified result tables.",
              "reason": "This is a cautious reservation, not an unsupported claim.",
              "evidence_needed": "No evidence needed; this is a statement of intent.",
              "severity": "minor"
            },
            {
              "section": "abstract",
              "claim": "The available dataset summary covers five TCGA cohorts totaling 2,586 patients.",
              "reason": "The evidence shows 2,586 patients and the listed cohorts.",
              "evidence_needed": "Clarify these numbers are from the supplied CSV files.",
              "severity": "minor"
            },
            {
              "section": "experiments",
              "claim": "The planned evaluation section is organized around BLCA and BRCA.",
              "reason": "The evidence lists these cohorts but does not include protocol details.",
              "evidence_needed": "Experimental protocol using these cohorts.",
              "severity": "minor"
            },
            {
              "section": "method",
              "claim": "The model builds prototypes with a Wasserstein hypergraph and reconstruction loss.",
              "reason": "No ablation studies are provided to validate that these components function as claimed.",
              "evidence_needed": "Ablation studies or component analysis.",
              "severity": "major"
            },
            {
              "section": "experiments",
              "claim": "The method obtains 0.999 on XYZ.",
              "reason": "XYZ and 0.999 are absent from the supplied experiment table.",
              "evidence_needed": "Add an experiment row for XYZ.",
              "severity": "major"
            }
          ],
          "section_quality_notes": []
        }
        """
    )
    state = {
        "request": PaperRequest(project_name="llm-review-filter-demo", target_venue="TPAMI"),
        "sections": DraftSections(method="Supported method.", experiments="Unsupported result."),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    reviewed = LLMSelfReviewAgent(llm_client=client).run(state)

    claims = reviewed["artifacts"]["llm_self_review"]["unsupported_claims"]
    assert len(claims) == 1
    assert claims[0]["section"] == "experiments"
    assert "Cox loss" not in " ".join(finding.issue for finding in reviewed["review_findings"])


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
        "review_findings": [
            SimpleNamespace(
                severity="minor",
                issue="Reviewer wants more novelty positioning.",
                suggestion="Add one sentence contrasting the method with ProtoSurv.",
            )
        ],
        "latex_project_dir": tmp_path / "latex",
        "latex_output_path": tmp_path / "latex" / "main.tex",
        "latex_zip_path": tmp_path / "paper.zip",
        "artifacts": {
            "section_writer_mode": "fallback",
            "section_writer_llm_attempted_sections": ["abstract", "method"],
            "section_writer_llm_successes": ["abstract"],
            "section_writer_repaired_sections": ["abstract"],
            "section_writer_section_errors": {"method": "blocked"},
            "llm_self_review": {"mode": "disabled", "unsupported_claims": []},
            "reference_verification": {"resolved_count": 1, "unresolved_count": 2},
            "reference_resolution_trace": [{"key": "paper", "status": "resolved"}],
            "reference_pruned_seed_keys": ["survivalprediction"],
            "related_work_discovery_mode": "openalex",
            "related_work_discovery_errors": {"recent_search": "timeout"},
            "related_work_discovery_error_details": [
                {
                    "source": "recent_search",
                    "query": "whole slide images survival prediction",
                    "sort": "publication_date:desc",
                    "error": "timeout",
                }
            ],
            "related_work_field_query": "whole slide images survival prediction",
            "related_work_baseline_mentioned_queries": [
                "Mobadersany | Predicting cancer outcomes from histology"
            ],
            "related_work_candidates": [
                {
                    "title": "A",
                    "category": "baseline_reference",
                    "discovery_path_label": "baseline reference list",
                    "query": "A",
                    "source_query": "baseline references of Baseline Survival Paper",
                    "year": "2018",
                },
                {
                    "title": "B",
                    "category": "influential",
                    "discovery_path_label": "high-citation field search",
                    "query": "B",
                    "source_query": "whole slide images survival prediction",
                    "year": "2020",
                },
                {
                    "title": "C",
                    "category": "recent",
                    "discovery_path_label": "recent field search",
                    "query": "C",
                    "source_query": "whole slide images survival prediction",
                    "year": "2026",
                },
            ],
            "experiment_result_tables": [{"caption": "Main Results"}],
            "experiment_sensitivity_evidence": [{"parameter": "lambda_rec"}],
            "experiment_statistical_tests": [{"comparison": "A vs B"}],
            "submission_readiness": {"overall_score": 82, "status": "needs_author_pass"},
            "submission_package": {
                "status": "needs_attention",
                "errors": ["missing main.tex"],
                "warnings": ["compile unavailable"],
                "checks": {
                    "compile": {
                        "mode": "compile",
                        "status": "failed",
                        "tool": "tectonic.exe",
                    }
                },
            },
            "presentation_plan": {
                "figures": [{"label": "fig:method-overview"}],
                "tables": [{"label": "tab:main-results"}],
                "open_items": ["Create method overview figure."],
            },
            "generated_figures": [{"label": "fig:main-results"}],
            "presentation_plan_path": str(tmp_path / "latex" / "FIGURE_TABLE_PLAN.md"),
            "code_baseline_comparison": {
                "likely_method_shifts": [{"technique": "hypergraph modeling"}],
                "innovation_seeds": ["Introduce hypergraph structure modeling."],
            },
            "latex_table_count": 3,
            "undefined_citation_keys": ["missing"],
            "draft_report_path": str(tmp_path / "latex" / "DRAFT_REPORT.md"),
            "submission_checklist_path": str(tmp_path / "latex" / "SUBMISSION_CHECKLIST.md"),
        },
    }

    summary = cli_module._build_run_summary(state, tmp_path / "draft.md")

    assert summary["project_name"] == "summary-demo"
    assert summary["llm_self_review_mode"] == "disabled"
    assert summary["bibliography_entries"] == 1
    assert summary["review_findings"] == 1
    assert summary["review_findings_minor"] == 1
    assert summary["review_finding_details"][0]["issue"] == "Reviewer wants more novelty positioning."
    assert summary["submission_readiness_score"] == 82
    assert summary["submission_readiness_status"] == "needs_author_pass"
    assert summary["submission_package_status"] == "needs_attention"
    assert summary["submission_package_errors"] == 1
    assert summary["submission_package_warnings"] == 1
    assert summary["submission_compile_status"] == "failed"
    assert summary["submission_compile_tool"] == "tectonic.exe"
    assert summary["presentation_figures"] == 1
    assert summary["generated_figures"] == 1
    assert summary["presentation_tables"] == 1
    assert summary["presentation_open_items"] == 1
    assert summary["code_baseline_method_shifts"] == 1
    assert summary["code_baseline_innovation_seeds"] == 1
    assert summary["reference_unresolved"] == 2
    assert summary["reference_pruned_seed_count"] == 1
    assert summary["reference_pruned_seed_keys"] == ["survivalprediction"]
    assert summary["reference_resolution_trace"] == 1
    assert summary["related_work_discovery_mode"] == "openalex"
    assert summary["related_work_candidates"] == 3
    assert summary["related_work_baseline_lineage_candidates"] == 1
    assert summary["related_work_influential_candidates"] == 1
    assert summary["related_work_recent_candidates"] == 1
    assert summary["related_work_candidate_categories"] == {
        "baseline_reference": 1,
        "influential": 1,
        "recent": 1,
    }
    assert summary["related_work_field_query"] == "whole slide images survival prediction"
    assert summary["related_work_baseline_mentioned_queries"] == [
        "Mobadersany | Predicting cancer outcomes from histology"
    ]
    assert summary["related_work_candidate_preview"][0]["title"] == "A"
    assert summary["related_work_candidate_preview"][0]["discovery_path_label"] == "baseline reference list"
    assert summary["related_work_discovery_error_count"] == 1
    assert summary["related_work_discovery_error_sources"] == ["recent_search"]
    assert summary["related_work_discovery_error_details"][0]["source"] == "recent_search"
    assert summary["related_work_discovery_error_details"][0]["query"] == "whole slide images survival prediction"
    assert summary["experiment_result_tables"] == 1
    assert summary["experiment_contract_status"] == "needs_attention"
    assert summary["experiment_contract_warnings"] == 1
    assert summary["experiment_sensitivity_evidence"] == 1
    assert summary["experiment_statistical_tests"] == 1
    assert summary["inputs"]["experiment_results_source"] == "none"
    assert summary["inputs"]["experiment_evidence_kind"] == "structured_state"
    assert summary["section_writer_llm_successes"] == ["abstract"]
    assert summary["section_writer_repaired_sections"] == ["abstract"]
    assert summary["section_writer_section_errors"] == {"method": "blocked"}
    assert summary["outputs"]["markdown"].endswith("draft.md")
    assert summary["outputs"]["presentation_plan_path"].endswith("FIGURE_TABLE_PLAN.md")
    assert summary["outputs"]["submission_checklist_path"].endswith("SUBMISSION_CHECKLIST.md")


def test_run_summary_records_llm_runtime_metadata_without_api_key(tmp_path):
    state = {
        "request": PaperRequest(project_name="llm-metadata-demo", target_venue="TPAMI"),
        "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
        "bibliography": [],
        "review_findings": [],
        "latex_project_dir": tmp_path / "latex",
        "latex_output_path": tmp_path / "latex" / "main.tex",
        "artifacts": {
            "section_writer_mode": "llm",
            "section_writer_llm_attempted_sections": ["abstract"],
            "section_writer_llm_successes": ["abstract"],
            "section_writer_section_errors": {},
            "section_writer_llm_call_trace": [
                {
                    "section": "abstract",
                    "phase": "draft",
                    "status": "success",
                    "model": "deepseek-v4-pro",
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                    "elapsed_seconds": 0.25,
                    "prompt_chars": 120,
                    "output_chars": 32,
                    "temperature": 0.25,
                    "max_tokens": 450,
                }
            ],
            "llm_self_review": {"mode": "disabled", "unsupported_claims": []},
            "experiment_result_tables": [{"caption": "Main Results"}],
            "submission_readiness": {"overall_score": 95, "status": "reviewable"},
            "submission_package": {"status": "valid", "errors": [], "warnings": [], "checks": {}},
        },
    }
    cli_module._record_runtime_modes(
        state,
        network_mode="offline",
        llm_mode="required",
        compile_latex_requested=False,
        min_llm_sections=1,
        llm_config=LLMConfig(
            api_key="secret-test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            timeout_seconds=30,
            max_retries=2,
        ),
    )

    summary = cli_module._build_run_summary(state, tmp_path / "draft.md")
    report = cli_module._build_acceptance_report(summary, min_llm_sections=1)
    serialized = json.dumps(summary)

    assert summary["inputs"]["llm_provider"] == "deepseek"
    assert summary["inputs"]["llm_model"] == "deepseek-v4-pro"
    assert summary["inputs"]["llm_endpoint_host"] == "api.deepseek.com"
    assert summary["inputs"]["llm_configured"] is True
    assert summary["inputs"]["llm_timeout_seconds"] == 30
    assert summary["inputs"]["llm_max_retries"] == 2
    assert summary["section_writer_llm_call_count"] == 1
    assert summary["section_writer_llm_call_successes"] == 1
    assert summary["section_writer_llm_total_tokens"] == 12
    assert "secret-test-key" not in serialized
    assert "- LLM provider/model: deepseek / deepseek-v4-pro" in report
    assert "- LLM endpoint host: api.deepseek.com" in report
    assert "- LLM section call trace: 1/1 successful; total_tokens=12" in report
    assert "secret-test-key" not in report


def test_acceptance_report_summarizes_passed_real_draft_contract(tmp_path):
    summary = {
        "project_name": "hyper-protosurv",
        "target_venue": "TPAMI",
        "inputs": {
            "code_path": "D:/code/agent/example/code/hyper-protosurv",
            "baseline_pdf_path": "D:/code/agent/example/baseline/baseline.pdf",
            "target_venue": "TPAMI",
            "experiment_results_provided": True,
            "experiment_results_source": "file",
            "experiment_results_path": "examples/tcga_real_results.md",
        },
        "template_source": "built-in",
        "section_writer_llm_attempted_sections": [
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "conclusion",
        ],
        "section_writer_llm_successes": [
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "conclusion",
        ],
        "section_writer_llm_call_count": 6,
        "section_writer_llm_call_successes": 6,
        "section_writer_llm_total_tokens": 4800,
        "section_writer_section_errors": {},
        "evidence_guard_findings": 0,
        "review_findings": 0,
        "submission_readiness_status": "reviewable",
        "submission_readiness_score": 100,
        "submission_package_status": "valid",
        "submission_package_errors": 0,
        "submission_package_warnings": 0,
        "submission_compile_mode": "compile",
        "submission_compile_status": "passed",
        "submission_compile_tool": "tectonic.exe",
        "related_work_discovery_mode": "openalex",
        "related_work_candidates": 7,
        "related_work_baseline_lineage_candidates": 3,
        "related_work_influential_candidates": 2,
        "related_work_recent_candidates": 2,
        "related_work_discovery_error_count": 1,
        "related_work_discovery_error_sources": ["recent_search"],
        "next_actions": [
            {
                "category": "related_work_retry",
                "action": "Review related-work discovery error sources and rerun online.",
                "command": "paper-agent paper-e2e-smoke --online",
            }
        ],
        "experiment_result_tables": 2,
        "experiment_ablation_evidence": 4,
        "experiment_sensitivity_evidence": 1,
        "experiment_statistical_tests": 1,
        "experiment_provenance": {
            "status": "complete",
            "errors": [],
            "warnings": [],
            "entries": [{"path": "logs/tcga_folds.csv", "kind": "local", "exists": True}],
            "checks": {
                "tables": 1,
                "entries": 1,
                "local_paths": 1,
                "remote_references": 0,
                "missing_paths": 0,
            },
        },
        "experiment_artifact_consistency": {
            "status": "complete",
            "errors": [],
            "warnings": [],
            "checks": {
                "paper_values": 4,
                "matched_values": 4,
                "missing_values": 0,
                "mismatched_values": 0,
                "csv_artifacts": 1,
            },
            "matches": [],
            "missing": [],
            "mismatches": [],
        },
        "presentation_figures": 4,
        "generated_figures": 4,
        "outputs": {
            "markdown": "outputs/llm-draft-smoke/draft.md",
            "latex_project_dir": "outputs/hyper-protosurv-llm-smoke",
            "latex_output_path": "outputs/hyper-protosurv-llm-smoke/main.tex",
            "latex_zip_path": "outputs/llm-draft-smoke-overleaf.zip",
            "draft_report_path": "outputs/hyper-protosurv-llm-smoke/DRAFT_REPORT.md",
            "presentation_plan_path": "outputs/hyper-protosurv-llm-smoke/FIGURE_TABLE_PLAN.md",
        },
    }

    report_path = cli_module._write_acceptance_report(
        summary,
        tmp_path / "ACCEPTANCE_REPORT.md",
        min_llm_sections=4,
    )
    report = report_path.read_text(encoding="utf-8")

    assert "- Overall status: PASS" in report
    assert "- Pipeline status: PASS" in report
    assert "- Submission evidence status: PASS" in report
    assert "| Experiment source integrity | PASS | kind=real_result_file" in report
    assert "| Experiment result contract | PASS | complete; main=2;" in report
    assert "| Experiment result provenance | PASS | complete; entries=1;" in report
    assert "| Experiment artifact consistency | PASS | complete; matched=4/4;" in report
    assert "- Main result tables: 2" in report
    assert "## Reference Readiness" in report
    assert "- Unresolved seed references: 0" in report
    assert "- Related-work discovery mode: openalex" in report
    assert "- Related-work candidate buckets: baseline_lineage=3; influential=2; recent=2; total=7" in report
    assert "- Related-work discovery errors: 1; sources=recent_search" in report
    assert "## Recommended Next Actions" in report
    assert "related_work_retry" in report
    assert "paper-agent paper-e2e-smoke --online" in report
    assert "| Experiment evidence coverage | PASS | main=2; ablation=4; sensitivity=1; statistical=1 |" in report
    assert "| LLM section drafting | PASS | 6/6 sections succeeded" in report
    assert "| LLM call trace | PASS | 6/6 calls succeeded; total_tokens=4800; required >= 4 |" in report
    assert "| LaTeX compile | PASS | status=passed; tool=tectonic.exe; mode=compile |" in report
    assert "outputs/hyper-protosurv-llm-smoke/main.tex" in report


def test_acceptance_report_fails_submission_evidence_for_mock_experiment_source():
    summary = {
        "inputs": {
            "code_path": "code",
            "baseline_pdf_path": "baseline.pdf",
            "target_venue": "TPAMI",
            "experiment_results_provided": True,
            "experiment_results_source": "file",
            "experiment_results_path": "examples/hyper_protosurv_mock_experiments.md",
        },
        "section_writer_llm_attempted_sections": [],
        "section_writer_llm_successes": [],
        "section_writer_section_errors": {},
        "evidence_guard_findings": 0,
        "review_findings": 0,
        "submission_readiness_status": "reviewable",
        "submission_readiness_score": 95,
        "submission_package_status": "valid",
        "submission_package_errors": 0,
        "submission_package_warnings": 0,
        "submission_compile_mode": "compile",
        "submission_compile_status": "passed",
        "submission_compile_tool": "tectonic.exe",
        "experiment_result_tables": 1,
        "presentation_figures": 0,
        "generated_figures": 0,
        "outputs": {
            "markdown": "draft.md",
            "latex_output_path": "main.tex",
            "draft_report_path": "DRAFT_REPORT.md",
        },
    }

    report = cli_module._build_acceptance_report(summary, min_llm_sections=0)

    assert "- Overall status: FAIL" in report
    assert "- Pipeline status: PASS" in report
    assert "- Submission evidence status: FAIL" in report
    assert "| Experiment source integrity | FAIL | kind=synthetic_mock" in report


def test_acceptance_report_fails_submission_evidence_for_invalid_artifact_consistency():
    summary = {
        "inputs": {
            "code_path": "code",
            "baseline_pdf_path": "baseline.pdf",
            "target_venue": "TPAMI",
            "experiment_results_provided": True,
            "experiment_results_source": "file",
            "experiment_results_path": "results.md",
        },
        "section_writer_llm_attempted_sections": ["abstract"],
        "section_writer_llm_successes": ["abstract"],
        "section_writer_section_errors": {},
        "evidence_guard_findings": 0,
        "review_findings": 0,
        "submission_readiness_status": "reviewable",
        "submission_readiness_score": 95,
        "submission_package_status": "valid",
        "submission_package_errors": 0,
        "submission_package_warnings": 0,
        "submission_compile_mode": "compile",
        "submission_compile_status": "passed",
        "submission_compile_tool": "tectonic.exe",
        "experiment_result_tables": 1,
        "experiment_ablation_evidence": 1,
        "experiment_sensitivity_evidence": 1,
        "experiment_statistical_tests": 1,
        "experiment_contract": {
            "status": "complete",
            "errors": [],
            "warnings": [],
            "checks": {
                "result_tables": 1,
                "numeric_comparisons": 2,
                "ablation_items": 1,
                "sensitivity_items": 1,
                "statistical_tests": 1,
            },
        },
        "experiment_provenance": {
            "status": "complete",
            "errors": [],
            "warnings": [],
            "checks": {
                "entries": 1,
                "local_paths": 1,
                "fingerprinted_local_paths": 1,
                "verified_checksums": 1,
            },
        },
        "experiment_artifact_consistency": {
            "status": "invalid",
            "errors": ["Artifact value mismatch for main_method ours BLCA C-index."],
            "warnings": [],
            "checks": {
                "paper_values": 2,
                "matched_values": 1,
                "missing_values": 0,
                "mismatched_values": 1,
                "csv_artifacts": 1,
            },
        },
        "presentation_figures": 0,
        "generated_figures": 0,
        "outputs": {
            "markdown": "draft.md",
            "latex_output_path": "main.tex",
            "draft_report_path": "DRAFT_REPORT.md",
        },
    }

    report = cli_module._build_acceptance_report(summary, min_llm_sections=1)

    assert "- Overall status: FAIL" in report
    assert "- Submission evidence status: FAIL" in report
    assert "| Experiment source integrity | PASS | kind=real_result_file" in report
    assert "| Experiment artifact consistency | FAIL | invalid; matched=1/2;" in report


def test_acceptance_report_marks_disabled_compile_as_warning():
    summary = {
        "inputs": {
            "code_path": "code",
            "baseline_pdf_path": "baseline.pdf",
            "target_venue": "TPAMI",
            "experiment_results_provided": True,
            "experiment_results_source": "file",
            "experiment_results_path": "results.md",
        },
        "section_writer_llm_attempted_sections": ["abstract", "method"],
        "section_writer_llm_successes": ["abstract", "method"],
        "section_writer_section_errors": {},
        "evidence_guard_findings": 0,
        "review_findings": 0,
        "submission_readiness_status": "reviewable",
        "submission_readiness_score": 95,
        "submission_package_status": "valid",
        "submission_package_errors": 0,
        "submission_package_warnings": 0,
        "submission_compile_mode": "not_run",
        "submission_compile_status": "disabled",
        "submission_compile_tool": "tectonic.exe",
        "experiment_result_tables": 1,
        "presentation_figures": 0,
        "generated_figures": 0,
        "outputs": {"markdown": "draft.md", "latex_output_path": "main.tex", "draft_report_path": "DRAFT_REPORT.md"},
    }

    report = cli_module._build_acceptance_report(summary, min_llm_sections=2)

    assert "- Overall status: PASS_WITH_WARNINGS" in report
    assert "- Submission evidence status: WARN" in report
    assert "| LaTeX compile | WARN | status=disabled; tool=tectonic.exe; mode=not_run |" in report


def test_acceptance_report_includes_latex_install_hint_when_tool_missing():
    summary = {
        "inputs": {
            "code_path": "code",
            "baseline_pdf_path": "baseline.pdf",
            "target_venue": "TPAMI",
            "experiment_results_provided": True,
            "experiment_results_source": "file",
            "experiment_results_path": "results.md",
        },
        "section_writer_llm_attempted_sections": [],
        "section_writer_llm_successes": [],
        "section_writer_section_errors": {},
        "evidence_guard_findings": 0,
        "review_findings": 0,
        "submission_readiness_status": "reviewable",
        "submission_readiness_score": 95,
        "submission_package_status": "needs_attention",
        "submission_package_errors": 0,
        "submission_package_warnings": 1,
        "submission_compile_mode": "not_run",
        "submission_compile_status": "tool_unavailable",
        "submission_compile_tool": "",
        "submission_compile_install_hint": "conda install -n agent -c conda-forge tectonic",
        "experiment_result_tables": 1,
        "presentation_figures": 0,
        "generated_figures": 0,
        "outputs": {"markdown": "draft.md", "latex_output_path": "main.tex", "draft_report_path": "DRAFT_REPORT.md"},
    }

    report = cli_module._build_acceptance_report(summary, min_llm_sections=0)

    assert "| LaTeX compile | WARN | status=tool_unavailable; tool=none; mode=not_run;" in report
    assert "install=conda install -n agent -c conda-forge tectonic" in report


def test_acceptance_report_treats_minor_reviewer_and_package_warnings_as_warnings():
    summary = {
        "inputs": {
            "code_path": "code",
            "baseline_pdf_path": "baseline.pdf",
            "target_venue": "TPAMI",
            "experiment_results_provided": True,
            "experiment_results_source": "file",
        },
        "section_writer_llm_attempted_sections": [],
        "section_writer_llm_successes": [],
        "section_writer_section_errors": {},
        "evidence_guard_findings": 0,
        "review_findings": 1,
        "review_findings_major": 0,
        "review_findings_minor": 1,
        "review_finding_details": [
            {
                "severity": "minor",
                "issue": "Reviewer wants stronger novelty positioning.",
                "suggestion": "Add a concise comparison to ProtoSurv in the introduction.",
            }
        ],
        "submission_readiness_status": "reviewable",
        "submission_readiness_score": 91,
        "submission_package_status": "needs_attention",
        "submission_package_errors": 0,
        "submission_package_warnings": 2,
        "submission_compile_mode": "compile",
        "submission_compile_status": "passed",
        "submission_compile_tool": "tectonic.exe",
        "experiment_result_tables": 1,
        "experiment_ablation_evidence": 0,
        "experiment_sensitivity_evidence": 0,
        "experiment_statistical_tests": 0,
        "presentation_figures": 0,
        "generated_figures": 0,
        "outputs": {"markdown": "draft.md", "latex_output_path": "main.tex", "draft_report_path": "DRAFT_REPORT.md"},
    }

    report = cli_module._build_acceptance_report(summary, min_llm_sections=0)

    assert "- Overall status: PASS_WITH_WARNINGS" in report
    assert "- Submission evidence status: WARN" in report
    assert "| Reviewer | WARN | 0 major; 1 minor |" in report
    assert "| Submission package | WARN | needs_attention; errors=0; warnings=2 |" in report
    assert "## Reviewer Findings" in report
    assert "- [minor] Reviewer wants stronger novelty positioning." in report
    assert "Suggestion: Add a concise comparison to ProtoSurv in the introduction." in report


def test_tcga_cohort_summary_uses_dataset_csv_without_performance_claims(tmp_path):
    dataset_dir = tmp_path / "dataset_csv"
    dataset_dir.mkdir()
    (dataset_dir / "BLCA.csv").write_text(
        ",case_id,slide_id,survival_months,censorship\n"
        "0,TCGA-AA-0001-01Z-00-DX1.A,TCGA-AA-0001-01Z-00-DX1.A.svs,12.0,0\n"
        "1,TCGA-AA-0001-01Z-00-DX2.B,TCGA-AA-0001-01Z-00-DX2.B.svs,12.0,0\n"
        "2,TCGA-AA-0002-01Z-00-DX1.C,TCGA-AA-0002-01Z-00-DX1.C.svs,24.0,1\n",
        encoding="utf-8",
    )

    summary = cli_module._build_tcga_cohort_summary(dataset_dir)

    assert "TCGA Cohort Data Summary" in summary
    assert "| BLCA | 2 | 3 | 2 | 1 | 12.00 | 12.00 | 24.00 |" in summary
    assert "not a model-performance result file" in summary
    assert "Add real trained-model performance tables" in summary
    assert "improvement" not in summary.lower()
    assert "ablation" not in summary.lower()
    assert "baseline" not in summary.lower()
    assert "C-index" not in summary
    assert "IBS" not in summary


def test_planner_reserves_improvement_claim_when_experiments_incomplete():
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive hypergraph prototype learning",
                motivation="Robustness",
                technical_idea="Use adaptive hyperedges.",
            )
        ],
        "experiments": ExperimentSummary(
            datasets=["BLCA", "BRCA"],
            missing_details=["Evaluation metrics are not explicit."],
        ),
    }

    planned = PaperPlannerAgent().run(state)

    assert "reserving empirical improvement claims" in planned["outline"].central_claim
    assert "This paper improves" not in planned["outline"].central_claim


def test_fallback_conclusion_avoids_guard_trigger_when_results_missing():
    sections = SectionWriterAgent()._run_fallback(
        {
            "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
            "experiments": ExperimentSummary(
                datasets=["BLCA"],
                missing_details=["Evaluation metrics are not explicit."],
            ),
            "innovations": [
                InnovationPoint(
                    name="Innovation 1: Adaptive hypergraph prototype learning",
                    motivation="Robustness",
                    technical_idea="Use adaptive hyperedges.",
                )
            ],
            "outline": PaperOutline(
                central_claim=(
                    "This paper addresses the baseline setting through adaptive hyperedges, "
                    "while reserving empirical improvement claims for verified result tables."
                )
            ),
            "artifacts": {},
        }
    )

    assert "empirical section remains incomplete" in sections.conclusion
    assert "empirical validation" not in sections.conclusion.lower()
    assert "improving the baseline setting" not in sections.conclusion


def test_llm_section_prompt_blocks_performance_claims_when_results_missing(monkeypatch):
    client = FakeLLMClient("A cautious section.")
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(
            datasets=["BLCA"],
            missing_details=["Evaluation metrics are not explicit."],
        ),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    SectionWriterAgent(llm_client=client)._run_llm_section(state, "abstract")

    payload = json.loads(client.calls[0]["messages"][1].content)
    joined_rules = " ".join(payload["hard_rules"])
    assert "experiment evidence is incomplete" in joined_rules
    assert "Do not mention C-index" in joined_rules
    assert "Do not invent preprocessing accuracies" in joined_rules
    assert "Do not include writer instructions" in joined_rules
    assert "Do not copy numeric citations" in joined_rules
    assert "do not describe the design as baseline modifications" in joined_rules
    assert payload["missing_experiment_details"] == ["Evaluation metrics are not explicit."]


def test_llm_section_writer_records_successful_sections():
    client = FakeLLMClient("A cautious section.")
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    SectionWriterAgent(llm_client=client).run(state)

    assert state["artifacts"]["section_writer_mode"] == "llm"
    assert state["artifacts"]["section_writer_llm_attempted_sections"] == [
        "abstract",
        "introduction",
        "related_work",
        "method",
        "experiments",
        "conclusion",
    ]
    assert state["artifacts"]["section_writer_llm_successes"] == [
        "abstract",
        "introduction",
        "related_work",
        "method",
        "experiments",
        "conclusion",
    ]


def test_llm_section_writer_records_sanitized_call_trace():
    client = FakeLLMClient(
        "A cautious section.",
        model="deepseek-v4-pro",
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    SectionWriterAgent(llm_client=client).run(state)

    trace = state["artifacts"]["section_writer_llm_call_trace"]
    assert len(trace) == 6
    assert trace[0]["section"] == "abstract"
    assert trace[0]["phase"] == "draft"
    assert trace[0]["status"] == "success"
    assert trace[0]["model"] == "deepseek-v4-pro"
    assert trace[0]["usage"]["total_tokens"] == 18
    assert trace[0]["prompt_chars"] > 0
    assert trace[0]["output_chars"] == len("A cautious section.")
    assert "messages" not in trace[0]
    assert "content" not in trace[0]


def test_llm_related_work_gets_citation_backstop_when_model_omits_cites():
    client = FakeLLMClient("Prior survival-prediction work motivates this setting.")
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [],
        "bibliography": [
            CitationEntry(key="baseline", title="Baseline Paper"),
            CitationEntry(key="recent", title="Recent Paper"),
        ],
        "artifacts": {},
    }

    section = SectionWriterAgent(llm_client=client)._run_llm_section(state, "related_work")

    assert r"\cite{baseline,recent}" in section


def test_llm_section_writer_repairs_rejected_method_once():
    client = FakeSequenceLLMClient(
        [
            "A cautious abstract.",
            "A cautious introduction.",
            "A cautious related work section.",
            (
                "### Adaptive Prototype Geometry\n"
                "The baseline uses online prototypes, and the proposed method replaces "
                "the baseline prototype bank with offline OT prototypes."
            ),
            (
                "### Adaptive Prototype Geometry\n"
                "The proposed method constructs an offline optimal-transport prototype geometry "
                "from patch features and uses it as the scaffold for survival representation learning."
            ),
            "A cautious experiments section.",
            "A cautious conclusion.",
        ]
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype geometry",
                motivation="Prototype geometry should be explicit.",
                technical_idea="Construct adaptive prototype geometry with optimal transport.",
                evidence=["data_preparation/hypergraph.py:20"],
            )
        ],
        "bibliography": [],
        "artifacts": {},
    }

    SectionWriterAgent(llm_client=client).run(state)

    assert state["artifacts"]["section_writer_mode"] == "llm"
    assert state["artifacts"]["section_writer_repaired_sections"] == ["method"]
    assert "method" not in state["artifacts"].get("section_writer_section_errors", {})
    assert "replaces" not in state["sections"].method
    assert "offline optimal-transport prototype geometry" in state["sections"].method
    trace = state["artifacts"]["section_writer_llm_call_trace"]
    assert trace[3]["section"] == "method"
    assert trace[3]["phase"] == "draft"
    assert trace[4]["section"] == "method"
    assert trace[4]["phase"] == "repair"
    assert len(client.calls) == 7
    repair_payload = json.loads(client.calls[4]["messages"][1].content)
    assert repair_payload["task"] == "Repair the rejected method section."
    assert "code/baseline differences" in repair_payload["validation_error"]
    assert "Present the proposed computation as a standalone method." in repair_payload["repair_rules"]


def test_llm_section_writer_repairs_method_omitted_innovation():
    client = FakeSequenceLLMClient(
        [
            (
                "### Adaptive Prototype Geometry\n"
                "The proposed method constructs optimal-transport prototypes from patch features."
            ),
            (
                "### Adaptive Prototype Geometry\n"
                "The proposed method constructs optimal-transport prototypes from patch features.\n\n"
                "### Minimal Survival-Reconstruction Objective\n"
                "The training objective combines Cox survival prediction with reconstruction evidence "
                "and leaves legacy regularizers outside the proposed objective."
            ),
        ]
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype geometry",
                motivation="Prototype geometry should be explicit.",
                technical_idea="Construct adaptive prototype geometry with optimal transport.",
                evidence=["data_preparation/hypergraph.py:20"],
            ),
            InnovationPoint(
                name="Innovation 2: Minimal survival-reconstruction objective",
                motivation="Training should remain compact.",
                technical_idea="Simplify the training objective by removing unsupported legacy regularizers.",
                evidence=["utils/loss.py:12"],
            ),
        ],
        "bibliography": [],
        "artifacts": {},
    }

    section = SectionWriterAgent(llm_client=client)._run_llm_section(state, "method")

    assert "Minimal Survival-Reconstruction Objective" in section
    assert state["artifacts"]["section_writer_repaired_sections"] == ["method"]
    assert "omitted innovation points" in state["artifacts"]["section_writer_repair_attempts"]["method"]
    repair_payload = json.loads(client.calls[1]["messages"][1].content)
    assert "omitted innovation points" in repair_payload["validation_error"]


def test_llm_method_repair_auto_augments_still_omitted_innovation():
    client = FakeSequenceLLMClient(
        [
            (
                "### Adaptive Prototype Geometry\n"
                "The proposed method constructs optimal-transport prototypes from patch features."
            ),
            (
                "### Adaptive Prototype Geometry\n"
                "The proposed method constructs optimal-transport prototypes from patch features."
            ),
        ]
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype geometry",
                motivation="Prototype geometry should be explicit.",
                technical_idea="Construct adaptive prototype geometry with optimal transport.",
                evidence=["data_preparation/hypergraph.py:20"],
            ),
            InnovationPoint(
                name="Innovation 2: Minimal survival-reconstruction objective",
                motivation="Training should remain compact.",
                technical_idea="Simplify the training objective with survival and reconstruction terms.",
                evidence=["utils/loss.py:12"],
            ),
        ],
        "bibliography": [],
        "artifacts": {},
    }

    section = SectionWriterAgent(llm_client=client)._run_llm_section(state, "method")

    assert "Minimal survival-reconstruction objective" in section
    assert "survival and reconstruction terms" in section
    assert state["artifacts"]["section_writer_repaired_sections"] == ["method"]


def test_llm_section_rejects_placeholders_and_writer_instructions():
    client = FakeLLMClient(
        "[Placeholder: Table I should include final experimental results once final results are available.]"
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    try:
        SectionWriterAgent(llm_client=client)._run_llm_section(state, "experiments")
    except ValueError as exc:
        assert "draft instructions or placeholders" in str(exc)
    else:
        raise AssertionError("Expected placeholder-heavy LLM section to be rejected.")


def test_llm_section_rejects_unsupported_experiment_claims_with_results():
    client = FakeLLMClient(
        "Patch-level tissue classification achieves 92.5% accuracy before survival training."
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(
            raw_preview=(
                "| Method | BLCA C-index |\n"
                "|---|---:|\n"
                "| baseline | 0.646 |\n"
                "| ours | 0.671 |\n"
            ),
            datasets=["BLCA"],
            metrics=["C-INDEX"],
        ),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    try:
        SectionWriterAgent(llm_client=client)._run_llm_section(state, "experiments")
    except ValueError as exc:
        message = str(exc)
        assert "unsupported experiment claims" in message
        assert "92.5%" in message
        assert "ACCURACY" in message
    else:
        raise AssertionError("Expected unsupported LLM experiment claim to be rejected.")


def test_llm_experiments_prompt_includes_evidence_contract():
    client = FakeLLMClient(
        "### Main Results\n"
        "On BLCA C-index, Hyper-ProtoSurv obtains 0.671 compared with 0.646 for the baseline."
    )
    experiments = ExperimentSummary(
        raw_preview=(
            "| Method | BLCA C-index |\n"
            "|---|---:|\n"
            "| ProtoSurv baseline | 0.646 |\n"
            "| Hyper-ProtoSurv ours | 0.671 |\n"
        ),
        datasets=["BLCA"],
        metrics=["C-INDEX"],
        result_tables=[
            ExperimentTableSummary(
                caption="Main Results",
                metric="C-INDEX",
                method="Hyper-ProtoSurv ours",
                baseline="ProtoSurv baseline",
                comparisons=[
                    ExperimentComparison(
                        table_caption="Main Results",
                        dataset="BLCA",
                        metric="C-INDEX",
                        method="Hyper-ProtoSurv ours",
                        baseline="ProtoSurv baseline",
                        method_value=0.671,
                        baseline_value=0.646,
                        signed_improvement=0.025,
                        improved=True,
                    )
                ],
            )
        ],
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": experiments,
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }

    SectionWriterAgent(llm_client=client)._run_llm_section(state, "experiments")

    payload = json.loads(client.calls[0]["messages"][1].content)
    contract = payload["experiment_evidence_contract"]
    assert contract["status"] == "structured"
    assert contract["allowed_datasets"] == ["BLCA"]
    assert contract["allowed_metrics"] == ["C-INDEX"]
    assert "0.671" in contract["allowed_numbers"]
    assert any("signed improvement +0.025" in claim for claim in contract["allowed_result_claims"])
    assert any("accuracy" in rule for rule in contract["rules"])
    assert "structured result tables are present" in payload["section_instruction"].lower()


def test_llm_experiment_claim_validator_ignores_markdown_heading_numbers():
    experiments = ExperimentSummary(
        raw_preview=(
            "| Method | BLCA C-index |\n"
            "|---|---:|\n"
            "| ProtoSurv baseline | 0.646 |\n"
            "| Hyper-ProtoSurv ours | 0.671 |\n"
        ),
        datasets=["BLCA"],
        metrics=["C-INDEX"],
    )

    SectionWriterAgent()._validate_llm_section(
        "### 4.1 Main Results\n"
        "On BLCA C-index, Hyper-ProtoSurv obtains 0.671 compared with 0.646 for the baseline.",
        "experiments",
        experiments,
        [],
        [],
    )


def test_llm_section_cleaner_removes_numeric_citations_only():
    text = SectionWriterAgent()._clean_section_text(
        "related_work",
        "Related Work\nGraph MIL methods [5, 18, 20] motivate this line [baseline].",
    )

    assert "[5" not in text
    assert "[baseline]" in text
    assert text.startswith("Graph MIL")


def test_llm_sections_fall_back_on_empirical_overclaim_when_results_missing():
    client = FakeLLMClient("We report C-index after five-fold cross-validation and ablation studies.")
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(
            datasets=["BLCA"],
            missing_details=["Evaluation metrics are not explicit."],
        ),
        "innovations": [],
        "outline": PaperOutline(
            central_claim=(
                "This paper addresses the baseline setting while reserving empirical improvement claims "
                "for verified result tables."
            )
        ),
        "bibliography": [],
        "artifacts": {},
    }

    SectionWriterAgent(llm_client=client).run(state)

    assert state["artifacts"]["section_writer_mode"] == "partial_llm"
    assert "unsupported empirical language" in state["artifacts"]["section_writer_section_errors"]["experiments"]
    assert "C-index" not in state["sections"].experiments
    assert "five-fold" not in state["sections"].experiments
    assert "structured numeric result table" in state["sections"].experiments


def test_llm_method_rejects_unsupported_mechanistic_outcome_when_results_missing():
    client = FakeLLMClient(
        "### Bidirectional Hyperedge Convolution\n"
        "This bidirectional scheme allows the model to capture both local tissue composition "
        "and global context shared across hyperedges."
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(
            datasets=["BLCA"],
            missing_details=["Ablation rows are not explicit."],
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive hypergraph prototype learning",
                motivation="Prototype learning needs evidence.",
                technical_idea="Use bidirectional hyperedge convolution.",
                evidence=["models/HCoN/model.py:42 (BHE/HCoN)"],
            )
        ],
        "outline": PaperOutline(),
        "bibliography": [],
        "artifacts": {},
    }

    try:
        SectionWriterAgent(llm_client=client)._run_llm_section(state, "method")
    except ValueError as exc:
        assert "unsupported method outcome" in str(exc)
    else:
        raise AssertionError("Expected unsupported method outcome to be rejected")


def test_llm_method_rejects_baseline_diff_framing():
    client = FakeLLMClient(
        "### Adaptive Prototype Geometry\n"
        "In the baseline ProtoSurv, an online prototype bank is used. "
        "Hyper-ProtoSurv replaces the baseline prototype bank with offline OT prototypes."
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(datasets=["BLCA"], metrics=["C-INDEX"]),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype geometry",
                motivation="Prototype geometry should be explicit.",
                technical_idea="Construct adaptive prototype geometry with optimal transport.",
                evidence=["data_preparation/hypergraph.py:20"],
            )
        ],
        "outline": PaperOutline(),
        "bibliography": [],
        "artifacts": {},
    }

    try:
        SectionWriterAgent(llm_client=client)._run_llm_section(state, "method")
    except ValueError as exc:
        assert "code/baseline differences" in str(exc)
    else:
        raise AssertionError("Expected baseline-diff Method framing to be rejected")


def test_llm_related_work_rejects_method_effect_claims_when_results_missing():
    client = FakeLLMClient(
        "The proposed method preserves class-specific geometric structure, removes the need "
        "for online losses, and learns soft hyperedge weights unlike prior hypergraph models."
    )
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(
            datasets=["BLCA"],
            missing_details=["Baseline comparison rows are not explicit."],
        ),
        "innovations": [],
        "outline": PaperOutline(),
        "bibliography": [],
        "artifacts": {},
    }

    try:
        SectionWriterAgent(llm_client=client)._run_llm_section(state, "related_work")
    except ValueError as exc:
        assert "unsupported related-work method effect" in str(exc)
    else:
        raise AssertionError("Expected unsupported related-work method effect to be rejected")


def test_llm_abstract_rejects_final_dataset_claim_when_results_missing():
    client = FakeLLMClient("The dataset used in this study comprises five TCGA cohorts.")
    state = {
        "request": PaperRequest(project_name="tcga-demo", target_venue="TPAMI"),
        "baseline": BaselineSummary(title="Baseline"),
        "code": CodeSummary(summary="Code summary"),
        "experiments": ExperimentSummary(
            datasets=["BLCA"],
            missing_details=["Evaluation metrics are not explicit."],
        ),
        "innovations": [],
        "outline": PaperOutline(),
        "bibliography": [],
        "artifacts": {},
    }

    try:
        SectionWriterAgent(llm_client=client)._run_llm_section(state, "abstract")
    except ValueError as exc:
        assert "final dataset claim" in str(exc)
    else:
        raise AssertionError("Expected final dataset claim to be rejected")


def test_reviewer_accepts_dataset_tokens_present_in_experiment_evidence():
    experiments = ExperimentSummary(
        raw_preview="TCGA cohort summary built from BLCA and BRCA CSV files.",
        datasets=["BLCA", "BRCA"],
    )
    text = (
        "The available cohort summary is organized around TCGA, BLCA, and BRCA cases "
        "using UNI feature embeddings, OTSU thresholding, and L_{rec} regularization. "
        "The average C-index across the five cohorts is reported in Table III."
    )

    unsupported = ReviewerAgent()._unsupported_datasets(text, experiments)

    assert unsupported == []


def test_cli_draft_writes_acceptance_report_next_to_summary(monkeypatch, tmp_path, capsys):
    baseline_dir = tmp_path / "baseline"
    code_dir = tmp_path / "code"
    output_dir = tmp_path / "out"
    latex_dir = tmp_path / "latex"
    baseline_dir.mkdir()
    code_dir.mkdir()
    latex_dir.mkdir()
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (code_dir / "train.py").write_text("class HyperProtoSurv: pass\n", encoding="utf-8")
    experiment_path = tmp_path / "tcga_results.md"
    experiment_path.write_text(
        "| Method | BLCA C-index |\n"
        "|---|---:|\n"
        "| baseline | 0.646 |\n"
        "| ours | 0.671 |\n",
        encoding="utf-8",
    )
    (latex_dir / "main.tex").write_text("\\documentclass{IEEEtran}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def run(self, request):
            captured["request"] = request
            captured["disable_template_fetch"] = os.getenv("PAPER_AGENT_DISABLE_TEMPLATE_FETCH")
            captured["disable_reference_resolve"] = os.getenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE")
            captured["disable_related_work_discovery"] = os.getenv("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY")
            captured["disable_llm"] = os.getenv("PAPER_AGENT_DISABLE_LLM")
            captured["compile_latex"] = os.getenv("PAPER_AGENT_RUN_LATEX_COMPILE")
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                    "submission_readiness": {
                        "overall_score": 94,
                        "status": "reviewable",
                    },
                    "submission_package": {
                        "status": "valid",
                        "errors": [],
                        "warnings": [],
                        "checks": {
                            "compile": {
                                "mode": "compile",
                                "status": "passed",
                                "tool": "tectonic",
                            }
                        },
                    },
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    markdown_path = output_dir / "draft.md"
    summary_path = output_dir / "RUN_SUMMARY.json"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "draft",
            "--project-name",
            "hyper-protosurv-tcga",
            "--target-venue",
            "TPAMI",
            "--baseline",
            str(baseline_dir),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--output",
            str(markdown_path),
            "--summary",
            str(summary_path),
            "--offline",
            "--disable-llm",
            "--compile-latex",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "Acceptance report written to" in output
    assert "Experiment result contract: needs_attention" in output
    assert markdown_path.read_text(encoding="utf-8") == "# Draft"
    assert captured["request"].project_name == "hyper-protosurv-tcga"
    assert captured["disable_template_fetch"] == "1"
    assert captured["disable_reference_resolve"] == "1"
    assert captured["disable_related_work_discovery"] == "1"
    assert captured["disable_llm"] == "1"
    assert captured["compile_latex"] == "1"
    assert summary["inputs"]["network_mode"] == "offline"
    assert summary["inputs"]["llm_mode"] == "disabled"
    assert summary["inputs"]["latex_compile_requested"]
    assert summary["outputs"]["acceptance_report_path"].endswith("ACCEPTANCE_REPORT.md")
    assert "- Overall status: PASS_WITH_WARNINGS" in acceptance_report
    assert "- Submission evidence status: WARN" in acceptance_report


def test_cli_experiment_template_writes_contract_template(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "tcga_results_template.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "experiment-template",
            "--output",
            str(output_path),
            "--method",
            "Hyper-ProtoSurv ours",
            "--baseline",
            "ProtoSurv baseline",
            "--dataset",
            "BLCA",
            "--dataset",
            "BRCA",
        ],
    )

    cli_module.main()

    text = output_path.read_text(encoding="utf-8")
    assert "Experiment result template written to" in capsys.readouterr().out
    assert "| Method | BLCA C-index | BRCA C-index |" in text
    assert "## Ablation Study" in text
    assert "## Statistical Testing" in text
    assert "## Result Provenance" in text
    assert "TODO" in text


def test_cli_validate_results_writes_summary_for_complete_file(monkeypatch, tmp_path, capsys):
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index | BRCA C-index | LGG C-index | LUAD C-index | UCEC C-index |",
                "|---|---:|---:|---:|---:|---:|",
                "| ProtoSurv baseline | 0.646 | 0.669 | 0.724 | 0.636 | 0.658 |",
                "| Hyper-ProtoSurv ours | 0.671 | 0.691 | 0.746 | 0.661 | 0.681 |",
                "",
                "## Ablation Study",
                "",
                "Metric: Average C-index. Higher is better.",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.681 |",
                "| w/o L_rec | 0.665 |",
                "",
                "## Sensitivity Analysis",
                "",
                "Metric: Average C-index. Higher is better.",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.1 | 0.671 |",
                "| 1.0 | 0.681 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv vs ProtoSurv | C-index | Wilcoxon signed-rank | 0.018 |",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "validate-summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--summary",
            str(summary_path),
            "--strict",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "Experiment result contract: complete" in output
    assert summary["experiment_evidence"]["kind"] == "real_result_file"
    assert summary["experiment_contract"]["status"] == "complete"
    assert summary["experiment_contract"]["checks"]["numeric_comparisons"] == 5


def test_cli_validate_results_reports_complete_result_provenance(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    fold_log = logs_dir / "tcga_folds.csv"
    values_csv = logs_dir / "tcga_values.csv"
    eval_log = logs_dir / "tcga_eval.log"
    fold_log.write_text("fold,seed,cindex\n0,2026,0.671\n", encoding="utf-8")
    values_csv.write_text(
        "\n".join(
            [
                "method,parameter,parameter_value,comparison,dataset,metric,test,p_value,value",
                "ProtoSurv baseline,,,,BLCA,C-index,,,0.646",
                "Hyper-ProtoSurv ours,,,,BLCA,C-index,,,0.671",
                "ProtoSurv baseline,,,,BRCA,C-index,,,0.669",
                "Hyper-ProtoSurv ours,,,,BRCA,C-index,,,0.691",
                "Hyper-ProtoSurv ours,,,,Average,C-index,,,0.681",
                "w/o reconstruction loss,,,,Average,C-index,,,0.665",
                ",lambda_rec,0.5,,Average,C-index,,,0.676",
                ",lambda_rec,1.0,,Average,C-index,,,0.681",
                ",,,Hyper-ProtoSurv vs ProtoSurv,,C-index,Wilcoxon signed-rank,0.018,",
            ]
        ),
        encoding="utf-8",
    )
    eval_log.write_text("seed=2026 fold=0..4\n", encoding="utf-8")
    fold_hash = hashlib.sha256(fold_log.read_bytes()).hexdigest()
    values_hash = hashlib.sha256(values_csv.read_bytes()).hexdigest()
    eval_hash = hashlib.sha256(eval_log.read_bytes()).hexdigest()
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index | BRCA C-index |",
                "|---|---:|---:|",
                "| ProtoSurv baseline | 0.646 | 0.669 |",
                "| Hyper-ProtoSurv ours | 0.671 | 0.691 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.681 |",
                "| w/o reconstruction loss | 0.665 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.676 |",
                "| 1.0 | 0.681 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv vs ProtoSurv | C-index | Wilcoxon signed-rank | 0.018 |",
                "",
                "## Result Provenance",
                "",
                "| Artifact | Path | SHA256 | Description |",
                "|---|---|---|---|",
                f"| Fold-level CSV | logs/tcga_folds.csv | {fold_hash} | seed=2026; fold=0..4 |",
                f"| Result values CSV | logs/tcga_values.csv | {values_hash} | source values for paper table |",
                f"| Evaluation log | logs/tcga_eval.log | {eval_hash} | seed=2026; fold=0..4 |",
                "| Tracker export | wandb://entity/project/run-1 | - | final metrics snapshot |",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--summary",
            str(summary_path),
            "--strict",
            "--require-provenance",
            "--require-artifact-consistency",
            "--expected-dataset",
            "BLCA",
            "--expected-dataset",
            "BRCA",
            "--expected-metric",
            "C-INDEX",
            "--expected-method",
            "Hyper-ProtoSurv",
            "--expected-baseline",
            "ProtoSurv",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "Experiment result provenance: complete" in output
    assert "Provenance fingerprints: 3/3 local files; verified_checksums=3; checksum_mismatches=0" in output
    assert "Experiment artifact consistency: complete" in output
    assert "Artifact consistency coverage: matched=9/9; missing=0; mismatched=0; aggregated=0; csv_artifacts=2" in output
    assert "Experiment result quality: complete" in output
    assert summary["experiment_quality"]["status"] == "complete"
    assert summary["experiment_provenance"]["status"] == "complete"
    assert summary["experiment_provenance"]["checks"]["entries"] == 4
    assert summary["experiment_provenance"]["checks"]["local_paths"] == 3
    assert summary["experiment_provenance"]["checks"]["remote_references"] == 1
    assert summary["experiment_provenance"]["checks"]["fingerprinted_local_paths"] == 3
    assert summary["experiment_provenance"]["checks"]["verified_checksums"] == 3
    assert summary["experiment_provenance"]["entries"][0]["sha256"] == fold_hash
    assert summary["experiment_provenance"]["entries"][0]["hash_verified"] is True
    assert summary["experiment_artifact_consistency"]["status"] == "complete"
    assert summary["experiment_artifact_consistency"]["checks"]["matched_values"] == 9
    assert summary["experiment_artifact_consistency"]["checks"]["ablation_values"] == 2
    assert summary["experiment_artifact_consistency"]["checks"]["sensitivity_values"] == 2
    assert summary["experiment_artifact_consistency"]["checks"]["statistical_values"] == 1


def test_cli_validate_results_matches_wide_csv_artifacts(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    main_csv = logs_dir / "tcga_main_wide.csv"
    ablation_csv = logs_dir / "tcga_ablation_wide.csv"
    sensitivity_csv = logs_dir / "tcga_sensitivity_wide.csv"
    stats_csv = logs_dir / "tcga_stats.csv"
    main_csv.write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index",
                "ProtoSurv baseline,0.646,0.669",
                "Hyper-ProtoSurv ours,0.671,0.691",
            ]
        ),
        encoding="utf-8",
    )
    ablation_csv.write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.681",
                "w/o reconstruction loss,0.665",
            ]
        ),
        encoding="utf-8",
    )
    sensitivity_csv.write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.676",
                "1.0,0.681",
            ]
        ),
        encoding="utf-8",
    )
    stats_csv.write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv vs ProtoSurv,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    main_hash = hashlib.sha256(main_csv.read_bytes()).hexdigest()
    ablation_hash = hashlib.sha256(ablation_csv.read_bytes()).hexdigest()
    sensitivity_hash = hashlib.sha256(sensitivity_csv.read_bytes()).hexdigest()
    stats_hash = hashlib.sha256(stats_csv.read_bytes()).hexdigest()
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index | BRCA C-index |",
                "|---|---:|---:|",
                "| ProtoSurv baseline | 0.646 | 0.669 |",
                "| Hyper-ProtoSurv ours | 0.671 | 0.691 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.681 |",
                "| w/o reconstruction loss | 0.665 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.676 |",
                "| 1.0 | 0.681 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv vs ProtoSurv | C-index | Wilcoxon signed-rank | 0.018 |",
                "",
                "## Result Provenance",
                "",
                "| Artifact | Path | SHA256 | Description |",
                "|---|---|---|---|",
                f"| Main wide CSV | logs/tcga_main_wide.csv | {main_hash} | seed=2026; fold=0..4 |",
                f"| Ablation wide CSV | logs/tcga_ablation_wide.csv | {ablation_hash} | seed=2026; fold=0..4 |",
                f"| Sensitivity wide CSV | logs/tcga_sensitivity_wide.csv | {sensitivity_hash} | seed=2026; fold=0..4 |",
                f"| Statistical CSV | logs/tcga_stats.csv | {stats_hash} | seed=2026; fold=0..4 |",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--summary",
            str(summary_path),
            "--strict",
            "--require-provenance",
            "--require-artifact-consistency",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "Experiment artifact consistency: complete" in output
    assert "Artifact consistency coverage: matched=9/9; missing=0; mismatched=0; aggregated=0; csv_artifacts=4" in output
    consistency = summary["experiment_artifact_consistency"]
    assert consistency["status"] == "complete"
    assert consistency["checks"]["matched_values"] == 9
    assert consistency["checks"]["wide_values"] == 8
    assert consistency["checks"]["statistical_values"] == 1
    assert all(match["wide"] for match in consistency["matches"] if match["role"] != "statistical_test")


def test_cli_tcga_results_from_artifacts_writes_strict_file(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    main_csv = logs_dir / "tcga_main_wide.csv"
    ablation_csv = logs_dir / "tcga_ablation_wide.csv"
    sensitivity_csv = logs_dir / "tcga_sensitivity_wide.csv"
    stats_csv = logs_dir / "tcga_stats.csv"
    main_csv.write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index",
                "ProtoSurv baseline,0.646,0.669",
                "Hyper-ProtoSurv ours,0.671,0.691",
            ]
        ),
        encoding="utf-8",
    )
    ablation_csv.write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.681",
                "w/o reconstruction loss,0.665",
            ]
        ),
        encoding="utf-8",
    )
    sensitivity_csv.write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.676",
                "1.0,0.681",
            ]
        ),
        encoding="utf-8",
    )
    stats_csv.write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "tcga_results.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-results-from-artifacts",
            "--main-csv",
            str(main_csv),
            "--ablation-csv",
            str(ablation_csv),
            "--sensitivity-csv",
            str(sensitivity_csv),
            "--stats-csv",
            str(stats_csv),
            "--output",
            str(output_path),
            "--dataset",
            "BLCA",
            "--dataset",
            "BRCA",
            "--strict",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    result_text = output_path.read_text(encoding="utf-8")
    assert "TCGA result file written to" in output
    assert "Experiment artifact consistency: complete" in output
    assert "Artifact consistency coverage: matched=9/9; missing=0; mismatched=0; aggregated=0; csv_artifacts=4" in output
    assert "## Main Results" in result_text
    assert "| Hyper-ProtoSurv ours | 0.671 | 0.691 |" in result_text
    assert "## Ablation Study" in result_text
    assert "## Sensitivity Analysis" in result_text
    assert "## Statistical Testing" in result_text
    assert "## Result Provenance" in result_text
    assert "logs/tcga_main_wide.csv" in result_text
    assert hashlib.sha256(main_csv.read_bytes()).hexdigest() in result_text
    assert "TODO" not in result_text


def test_cli_tcga_results_from_artifacts_averages_long_fold_csv(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    main_csv = logs_dir / "tcga_main_folds.csv"
    ablation_csv = logs_dir / "tcga_ablation_folds.csv"
    sensitivity_csv = logs_dir / "tcga_sensitivity_folds.csv"
    stats_csv = logs_dir / "tcga_stats.csv"
    main_csv.write_text(
        "\n".join(
            [
                "method,dataset,metric,fold,seed,value",
                "ProtoSurv baseline,BLCA,C-index,0,2026,0.640",
                "ProtoSurv baseline,BLCA,C-index,1,2026,0.652",
                "Hyper-ProtoSurv ours,BLCA,C-index,0,2026,0.660",
                "Hyper-ProtoSurv ours,BLCA,C-index,1,2026,0.682",
            ]
        ),
        encoding="utf-8",
    )
    ablation_csv.write_text(
        "\n".join(
            [
                "method,dataset,metric,fold,seed,value",
                "Hyper-ProtoSurv ours,Average,C-index,0,2026,0.680",
                "Hyper-ProtoSurv ours,Average,C-index,1,2026,0.682",
                "w/o reconstruction loss,Average,C-index,0,2026,0.660",
                "w/o reconstruction loss,Average,C-index,1,2026,0.670",
            ]
        ),
        encoding="utf-8",
    )
    sensitivity_csv.write_text(
        "\n".join(
            [
                "parameter,parameter_value,dataset,metric,fold,seed,value",
                "lambda_rec,0.5,Average,C-index,0,2026,0.674",
                "lambda_rec,0.5,Average,C-index,1,2026,0.678",
                "lambda_rec,1.0,Average,C-index,0,2026,0.680",
                "lambda_rec,1.0,Average,C-index,1,2026,0.682",
            ]
        ),
        encoding="utf-8",
    )
    stats_csv.write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "tcga_results.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-results-from-artifacts",
            "--main-csv",
            str(main_csv),
            "--ablation-csv",
            str(ablation_csv),
            "--sensitivity-csv",
            str(sensitivity_csv),
            "--stats-csv",
            str(stats_csv),
            "--output",
            str(output_path),
            "--strict",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    result_text = output_path.read_text(encoding="utf-8")
    assert "Experiment artifact consistency: complete" in output
    assert "Artifact consistency coverage: matched=7/7; missing=0; mismatched=0; aggregated=6; csv_artifacts=4" in output
    assert "| ProtoSurv baseline | 0.646 |" in result_text
    assert "| Hyper-ProtoSurv ours | 0.671 |" in result_text
    assert "| w/o reconstruction loss | 0.665 |" in result_text
    assert "| 0.5 | 0.676 |" in result_text


def test_cli_tcga_results_from_artifacts_auto_detects_artifact_dir(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "tcga_main_results.csv").write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index",
                "ProtoSurv baseline,0.646,0.669",
                "Hyper-ProtoSurv ours,0.671,0.691",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "component_ablation.csv").write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.681",
                "w/o reconstruction loss,0.665",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "lambda_sensitivity.csv").write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.676",
                "1.0,0.681",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "statistical_tests.csv").write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "tcga_results.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-results-from-artifacts",
            "--artifacts-dir",
            str(logs_dir),
            "--output",
            str(output_path),
            "--dataset",
            "BLCA",
            "--dataset",
            "BRCA",
            "--strict",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    result_text = output_path.read_text(encoding="utf-8")
    assert "Auto-detected artifacts:" in output
    assert "- main:" in output
    assert "- ablation:" in output
    assert "- sensitivity:" in output
    assert "- stats:" in output
    assert "Experiment artifact consistency: complete" in output
    assert "Artifact consistency coverage: matched=9/9; missing=0; mismatched=0; aggregated=0; csv_artifacts=4" in output
    assert "tcga_main_results.csv" in result_text
    assert "component_ablation.csv" in result_text
    assert "lambda_sensitivity.csv" in result_text
    assert "statistical_tests.csv" in result_text


def test_cli_tcga_results_from_artifacts_auto_detects_combined_fold_csv(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    combined_csv = logs_dir / "tcga_combined_fold_results.csv"
    combined_csv.write_text(
        "\n".join(
            [
                "method,parameter,parameter_value,comparison,dataset,metric,test,p_value,fold,seed,value",
                "ProtoSurv baseline,,,,BLCA,C-index,,,0,2026,0.640",
                "ProtoSurv baseline,,,,BLCA,C-index,,,1,2026,0.652",
                "Hyper-ProtoSurv ours,,,,BLCA,C-index,,,0,2026,0.660",
                "Hyper-ProtoSurv ours,,,,BLCA,C-index,,,1,2026,0.682",
                "Hyper-ProtoSurv ours,,,,Average,C-index,,,0,2026,0.680",
                "Hyper-ProtoSurv ours,,,,Average,C-index,,,1,2026,0.682",
                "w/o reconstruction loss,,,,Average,C-index,,,0,2026,0.660",
                "w/o reconstruction loss,,,,Average,C-index,,,1,2026,0.670",
                ",lambda_rec,0.5,,Average,C-index,,,0,2026,0.674",
                ",lambda_rec,0.5,,Average,C-index,,,1,2026,0.678",
                ",lambda_rec,1.0,,Average,C-index,,,0,2026,0.680",
                ",lambda_rec,1.0,,Average,C-index,,,1,2026,0.682",
                ",,,Hyper-ProtoSurv ours vs ProtoSurv baseline,,C-index,Wilcoxon signed-rank,0.018,,,",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "tcga_results.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-results-from-artifacts",
            "--artifacts-dir",
            str(logs_dir),
            "--output",
            str(output_path),
            "--strict",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    result_text = output_path.read_text(encoding="utf-8")
    assert "Provenance artifacts: 1" in output
    assert "Experiment artifact consistency: complete" in output
    assert "Artifact consistency coverage: matched=7/7; missing=0; mismatched=0; aggregated=6; csv_artifacts=1" in output
    assert "| ProtoSurv baseline | 0.646 |" in result_text
    assert "| Hyper-ProtoSurv ours | 0.671 |" in result_text
    assert "| Combined result CSV | logs/tcga_combined_fold_results.csv |" in result_text
    assert "TODO" not in result_text


def test_cli_tcga_artifacts_doctor_passes_complete_dir(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "tcga_main_results.csv").write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index",
                "ProtoSurv baseline,0.646,0.669",
                "Hyper-ProtoSurv ours,0.671,0.691",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "component_ablation.csv").write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.681",
                "w/o reconstruction loss,0.665",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "lambda_sensitivity.csv").write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.676",
                "1.0,0.681",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "statistical_tests.csv").write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "artifact-doctor-pass.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-artifacts-doctor",
            "--artifacts-dir",
            str(logs_dir),
            "--summary",
            str(summary_path),
            "--dataset",
            "BLCA",
            "--dataset",
            "BRCA",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "TCGA artifact doctor:" in output
    assert "- Artifact directory: PASS" in output
    assert "- Main result CSV: PASS" in output
    assert "- Ablation CSV: PASS" in output
    assert "- Sensitivity CSV: PASS" in output
    assert "- Statistical-test CSV: PASS" in output
    assert "parsed_values=4" in output
    assert "Overall: PASS" in output
    assert "Ready command: paper-agent tcga-results-from-artifacts" in output
    assert "TCGA artifact doctor summary written to" in output
    assert summary["status"] == "pass"
    assert summary["roles"]["main"]["status"] == "pass"
    assert summary["roles"]["main"]["required"] is True
    assert summary["roles"]["main"]["datasets"] == ["BLCA", "BRCA"]
    assert summary["detected_artifacts"]["main"].endswith("tcga_main_results.csv")
    assert summary["next_command"].startswith("paper-agent tcga-results-from-artifacts")


def test_cli_tcga_artifacts_doctor_reports_missing_required_roles(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "tcga_main_results.csv").write_text(
        "\n".join(
            [
                "method,BLCA C-index",
                "ProtoSurv baseline,0.646",
                "Hyper-ProtoSurv ours,0.671",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "artifact-doctor-fail.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-artifacts-doctor",
            "--artifacts-dir",
            str(logs_dir),
            "--summary",
            str(summary_path),
            "--dataset",
            "BLCA",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA artifact doctor failed" in str(exc)
    else:
        raise AssertionError("Expected artifact doctor to fail when required roles are missing.")

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "- Main result CSV: PASS" in output
    assert "- Ablation CSV: FAIL" in output
    assert "- Sensitivity CSV: FAIL" in output
    assert "- Statistical-test CSV: FAIL" in output
    assert "Expected schema:" in output or "expected" in output
    assert "Missing required Ablation CSV" in output
    assert "Overall: FAIL" in output
    assert "TCGA artifact doctor summary written to" in output
    assert summary["status"] == "fail"
    assert summary["roles"]["ablation"]["status"] == "fail"
    assert summary["roles"]["ablation"]["required"] is True
    assert summary["roles"]["ablation"]["path"] == ""
    assert summary["next_command"].startswith("paper-agent tcga-artifact-template")
    assert summary["results_from_artifacts_command"].startswith("paper-agent tcga-results-from-artifacts")


def test_cli_tcga_artifact_template_writes_long_contract(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-artifact-template",
            "--output-dir",
            str(logs_dir),
            "--dataset",
            "BLCA",
            "--dataset",
            "BRCA",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert "TCGA artifact export templates written" in output
    expected_files = {
        "tcga_main_results.csv",
        "tcga_ablation.csv",
        "tcga_sensitivity.csv",
        "tcga_stats.csv",
        "EXPORT_CONTRACT.md",
        "ARTIFACT_SCHEMA.json",
    }
    assert {path.name for path in logs_dir.iterdir()} == expected_files
    main_csv = (logs_dir / "tcga_main_results.csv").read_text(encoding="utf-8")
    assert "method,dataset,metric,fold,seed,value" in main_csv
    assert "ProtoSurv baseline,BLCA,C-index,0,2026,TODO" in main_csv
    assert "Hyper-ProtoSurv ours,BRCA,C-index,0,2026,TODO" in main_csv
    contract = (logs_dir / "EXPORT_CONTRACT.md").read_text(encoding="utf-8")
    assert "# TCGA Artifact Export Contract" in contract
    assert "paper-agent tcga-artifacts-doctor --artifacts-dir ." in contract
    assert "ARTIFACT_SCHEMA.json" in contract
    assert "Do not use synthetic" in contract
    manifest = json.loads((logs_dir / "ARTIFACT_SCHEMA.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["style"] == "long"
    assert manifest["datasets"] == ["BLCA", "BRCA"]
    assert manifest["roles"]["main"]["columns"] == ["method", "dataset", "metric", "fold", "seed", "value"]
    assert manifest["roles"]["main"]["required"] is True
    assert manifest["roles"]["stats"]["file"] == "tcga_stats.csv"
    assert any("tcga-results-from-artifacts" in command for command in manifest["validation_commands"])


def test_cli_tcga_artifact_template_refuses_overwrite_without_force(monkeypatch, tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    existing_path = logs_dir / "tcga_main_results.csv"
    existing_path.write_text("keep this file\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-artifact-template",
            "--output-dir",
            str(logs_dir),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "Refusing to overwrite existing TCGA artifact template files" in str(exc)
    else:
        raise AssertionError("Expected template command to refuse overwriting existing artifacts.")

    assert existing_path.read_text(encoding="utf-8") == "keep this file\n"


def _write_complete_tcga_artifacts(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "tcga_main_wide.csv").write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index,LGG C-index,LUAD C-index,UCEC C-index",
                "ProtoSurv baseline,0.646,0.669,0.724,0.636,0.658",
                "Hyper-ProtoSurv ours,0.671,0.691,0.746,0.661,0.681",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_ablation_wide.csv").write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.690",
                "w/o reconstruction loss,0.672",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_sensitivity_wide.csv").write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.687",
                "1.0,0.690",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_stats.csv").write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )


def test_cli_tcga_results_guide_writes_templates_when_artifacts_missing(
    monkeypatch,
    tmp_path,
    capsys,
):
    example_root = tmp_path / "example"
    logs_dir = example_root / "results" / "logs"
    result_path = example_root / "results" / "tcga_results.md"
    summary_path = tmp_path / "guide-summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-results-guide",
            "--example-root",
            str(example_root),
            "--artifacts-dir",
            str(logs_dir),
            "--output",
            str(result_path),
            "--summary",
            str(summary_path),
            "--dataset",
            "BLCA",
            "--dataset",
            "BRCA",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "fill generated CSV templates" in str(exc)
    else:
        raise AssertionError("Expected guide to stop after writing TODO artifact templates.")

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "TCGA result CSV templates written" in output
    assert (logs_dir / "tcga_main_results.csv").is_file()
    assert (logs_dir / "tcga_ablation.csv").is_file()
    assert (logs_dir / "tcga_sensitivity.csv").is_file()
    assert (logs_dir / "tcga_stats.csv").is_file()
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "artifact_template_written"
    assert summary["experiment_results"] == str(result_path)
    assert len(summary["artifact_csv_todo_files"]) == 4
    assert summary["next_command"].startswith("paper-agent tcga-results-guide")


def test_cli_tcga_results_guide_generates_strict_result_markdown(
    monkeypatch,
    tmp_path,
    capsys,
):
    example_root = tmp_path / "example"
    logs_dir = example_root / "results" / "logs"
    result_path = example_root / "results" / "tcga_results.md"
    summary_path = tmp_path / "guide-summary.json"
    _write_complete_tcga_artifacts(logs_dir)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-results-guide",
            "--example-root",
            str(example_root),
            "--artifacts-dir",
            str(logs_dir),
            "--output",
            str(result_path),
            "--summary",
            str(summary_path),
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result_text = result_path.read_text(encoding="utf-8")
    assert "TCGA results guide completed." in output
    assert "TCGA result file written to" in output
    assert (tmp_path / "ARTIFACT_DOCTOR_SUMMARY.json").is_file()
    assert summary["status"] == "pass"
    assert summary["pipeline_phase"] == "result_markdown_generated"
    assert summary["experiment_contract_status"] == "complete"
    assert summary["experiment_provenance_status"] == "complete"
    assert summary["experiment_artifact_consistency_status"] == "complete"
    assert "## Main Results" in result_text
    assert "Hyper-ProtoSurv ours" in result_text
    assert summary["paper_e2e_acceptance_command"].startswith("paper-agent paper-e2e-acceptance")


def test_cli_tcga_readiness_schema_writes_schema_and_example(monkeypatch, tmp_path, capsys):
    schema_path = tmp_path / "tcga-readiness-schema.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-readiness-schema",
            "--output",
            str(schema_path),
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "TCGA readiness schema written" in output
    assert schema["schema_version"] == "tcga-readiness-contract/v1"
    assert schema["properties"]["schema_version"]["const"] == "tcga-readiness-contract/v1"
    assert "baseline_pdf" in schema["properties"]["requirements"]["properties"]
    assert "ready_to_generate" in schema["properties"]["requirements"]["additionalProperties"]["properties"]["status"]["enum"]

    example_path = tmp_path / "tcga-readiness-example.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-readiness-schema",
            "--example",
            "--output",
            str(example_path),
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    example = json.loads(example_path.read_text(encoding="utf-8"))
    assert "TCGA readiness example written" in output
    assert example["schema_version"] == "tcga-readiness-contract/v1"
    assert example["requirements"]["llm"]["status"] == "fail"
    assert example["next_actions"][0]["command"] == "paper-agent llm-doctor"


def test_cli_tcga_readiness_contract_is_validated_before_summary_write(tmp_path):
    valid_summary_path = tmp_path / "valid-summary.json"
    valid_summary = {"readiness_contract": cli_module._tcga_readiness_contract_example()}

    cli_module._write_run_summary_data(valid_summary, valid_summary_path)

    written = json.loads(valid_summary_path.read_text(encoding="utf-8"))
    assert written["readiness_contract"]["schema_version"] == "tcga-readiness-contract/v1"

    invalid_summary_path = tmp_path / "invalid-summary.json"
    invalid_summary = json.loads(json.dumps(valid_summary))
    del invalid_summary["readiness_contract"]["requirements"]["llm"]["command"]

    try:
        cli_module._write_run_summary_data(invalid_summary, invalid_summary_path)
    except SystemExit as exc:
        assert "requirements.llm is missing fields: command" in str(exc)
    else:
        raise AssertionError("Expected invalid readiness_contract to be rejected.")
    assert not invalid_summary_path.exists()


def test_cli_tcga_preflight_passes_with_complete_artifacts_without_result_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    logs_dir = example_root / "results" / "logs"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (logs_dir / "tcga_main_wide.csv").write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index,LGG C-index,LUAD C-index,UCEC C-index",
                "ProtoSurv baseline,0.646,0.669,0.724,0.636,0.658",
                "Hyper-ProtoSurv ours,0.671,0.691,0.746,0.661,0.681",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_ablation_wide.csv").write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.690",
                "w/o reconstruction loss,0.672",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_sensitivity_wide.csv").write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.687",
                "1.0,0.690",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_stats.csv").write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "preflight.json"
    monkeypatch.setattr(
        cli_module,
        "_latex_toolchain_status",
        lambda: {"available": True, "preferred_tool": "tectonic.exe", "install_hint": ""},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-preflight",
            "--example-root",
            str(example_root),
            "--summary",
            str(summary_path),
            "--submission-grade",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "TCGA preflight:" in output
    assert "- Artifact main: PASS" in output
    assert "- Artifact ablation: PASS" in output
    assert "- Artifact sensitivity: PASS" in output
    assert "- Artifact stats: PASS" in output
    assert "- Experiment results: WARN" in output
    assert "tcga-pipeline can generate it" in output
    assert "- LLM static config: PASS" in output
    assert "- LaTeX compiler: PASS" in output
    assert "Overall: PASS" in output
    assert "Readiness contract:" in output
    assert "Preflight summary written" in output
    assert summary["status"] == "pass"
    assert summary["submission_grade"] is True
    assert not summary["blocking_items"]
    assert summary["readiness_contract"]["schema_version"] == "tcga-readiness-contract/v1"
    assert summary["readiness_contract"]["status"] == "ready"
    assert summary["readiness_contract"]["ready_for_submission_grade"] is True
    assert summary["readiness_contract"]["requirements"]["experiment_results"]["status"] == "ready_to_generate"
    assert summary["readiness_contract"]["requirements"]["result_artifacts"]["status"] == "pass"
    assert summary["readiness_contract"]["requirements"]["llm"]["status"] == "pass"
    assert summary["readiness_contract"]["requirements"]["latex"]["status"] == "pass"
    assert summary["next_actions"][0]["category"] == "pipeline"
    assert "tcga-pipeline" in summary["next_actions"][0]["command"]
    assert summary["llm"]["provider"] == "deepseek"
    assert summary["llm_live_preflight"]["status"] == "skipped"


def test_cli_tcga_preflight_summary_records_live_llm_pass(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    logs_dir = example_root / "results" / "logs"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    _write_complete_tcga_artifacts(logs_dir)
    summary_path = tmp_path / "preflight-live-pass.json"
    monkeypatch.setattr(
        cli_module,
        "_llm_preflight_check",
        lambda *args, **kwargs: {
            "elapsed_seconds": 0.25,
            "response_model": "deepseek-v4-pro",
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-preflight",
            "--example-root",
            str(example_root),
            "--summary",
            str(summary_path),
            "--live-llm",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "- LLM live preflight: PASS" in output
    assert summary["status"] == "pass"
    assert summary["llm_live_preflight"]["status"] == "pass"
    assert summary["llm_live_preflight"]["elapsed_seconds"] == 0.25
    assert summary["llm_live_preflight"]["usage"]["total_tokens"] == 12
    assert "test-key" not in json.dumps(summary)


def test_cli_tcga_preflight_summary_records_live_llm_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    logs_dir = example_root / "results" / "logs"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    _write_complete_tcga_artifacts(logs_dir)
    summary_path = tmp_path / "preflight-live-fail.json"

    def fail_preflight(*args, **kwargs):
        raise SystemExit("TCGA preflight LLM preflight failed for deepseek/deepseek-v4-pro: quota blocked.")

    monkeypatch.setattr(cli_module, "_llm_preflight_check", fail_preflight)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-preflight",
            "--example-root",
            str(example_root),
            "--summary",
            str(summary_path),
            "--live-llm",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA preflight failed" in str(exc)
    else:
        raise AssertionError("Expected TCGA preflight to fail when live LLM preflight fails.")

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "- LLM live preflight: FAIL" in output
    assert summary["status"] == "fail"
    assert summary["llm_live_preflight"]["status"] == "fail"
    assert summary["llm_live_preflight"]["diagnostics"]["failure_kind"] == "quota"
    assert summary["llm_live_preflight"]["diagnostics"]["provider"] == "deepseek"
    assert "test-key" not in json.dumps(summary)


def test_cli_tcga_preflight_fails_without_results_or_artifacts(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    summary_path = tmp_path / "preflight-fail.json"
    monkeypatch.setattr(
        cli_module,
        "_latex_toolchain_status",
        lambda: {"available": True, "preferred_tool": "tectonic.exe", "install_hint": ""},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-preflight",
            "--example-root",
            str(example_root),
            "--disable-llm",
            "--summary",
            str(summary_path),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA preflight failed" in str(exc)
    else:
        raise AssertionError("Expected TCGA preflight to fail without results or artifacts.")

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "- Artifact directory: WARN" in output
    assert "- Experiment results: FAIL" in output
    assert "missing and no complete artifact set is available" in output
    assert "paper-agent tcga-artifact-template --output-dir" in output
    assert "Overall: FAIL" in output
    assert summary["readiness_contract"]["schema_version"] == "tcga-readiness-contract/v1"
    assert summary["readiness_contract"]["status"] == "blocked"
    assert summary["readiness_contract"]["requirements"]["experiment_results"]["status"] == "fail"
    assert summary["readiness_contract"]["requirements"]["llm"]["status"] == "disabled"
    assert summary["readiness_contract"]["requirements"]["latex"]["status"] == "pass"
    assert summary["next_actions"][0]["category"] == "experiment_results"
    assert "tcga-artifact-template" in summary["next_actions"][0]["command"]


def test_cli_validate_results_matches_fold_level_csv_mean(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    values_csv = logs_dir / "tcga_folds.csv"
    values_csv.write_text(
        "\n".join(
            [
                "method,parameter,parameter_value,comparison,dataset,metric,test,p_value,fold,seed,value",
                "ProtoSurv baseline,,,,BLCA,C-index,,,0,2026,0.640",
                "ProtoSurv baseline,,,,BLCA,C-index,,,1,2026,0.652",
                "Hyper-ProtoSurv ours,,,,BLCA,C-index,,,0,2026,0.660",
                "Hyper-ProtoSurv ours,,,,BLCA,C-index,,,1,2026,0.682",
                "Hyper-ProtoSurv ours,,,,Average,C-index,,,,,0.671",
                "w/o reconstruction loss,,,,Average,C-index,,,,,0.659",
                ",lambda_rec,0.5,,Average,C-index,,,,,0.667",
                ",lambda_rec,1.0,,Average,C-index,,,,,0.671",
                ",,,Hyper-ProtoSurv ours vs ProtoSurv baseline,,C-index,Wilcoxon signed-rank,0.018,,,",
            ]
        ),
        encoding="utf-8",
    )
    values_hash = hashlib.sha256(values_csv.read_bytes()).hexdigest()
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index |",
                "|---|---:|",
                "| ProtoSurv baseline | 0.646 |",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "| w/o reconstruction loss | 0.659 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.667 |",
                "| 1.0 | 0.671 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
                "",
                "## Result Provenance",
                "",
                "| Artifact | Path | SHA256 | Description |",
                "|---|---|---|---|",
                f"| Fold-level CSV | logs/tcga_folds.csv | {values_hash} | seed=2026; fold=0..1 |",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--summary",
            str(summary_path),
            "--strict",
            "--require-provenance",
            "--require-artifact-consistency",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "Experiment artifact consistency: complete" in output
    assert "Artifact consistency coverage: matched=7/7; missing=0; mismatched=0; aggregated=2; csv_artifacts=1" in output
    consistency = summary["experiment_artifact_consistency"]
    assert consistency["checks"]["aggregated_values"] == 2
    assert consistency["checks"]["ablation_values"] == 2
    assert consistency["checks"]["sensitivity_values"] == 2
    assert consistency["checks"]["statistical_values"] == 1
    assert consistency["matches"][0]["aggregation"] == "mean"
    assert consistency["matches"][0]["fold_count"] == 2


def test_cli_validate_results_requires_result_provenance_when_requested(monkeypatch, tmp_path, capsys):
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index |",
                "|---|---:|",
                "| ProtoSurv baseline | 0.646 |",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "| w/o reconstruction loss | 0.659 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.667 |",
                "| 1.0 | 0.671 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--strict",
            "--require-provenance",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected strict validation to fail without provenance.")
    output = capsys.readouterr().out
    assert "Experiment result provenance: invalid" in output
    assert "PROVENANCE ERROR: Missing result provenance table." in output


def test_cli_validate_results_fails_on_provenance_checksum_mismatch(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "tcga_folds.csv").write_text("fold,seed,cindex\n0,2026,0.671\n", encoding="utf-8")
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index |",
                "|---|---:|",
                "| ProtoSurv baseline | 0.646 |",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "| w/o reconstruction loss | 0.659 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.667 |",
                "| 1.0 | 0.671 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
                "",
                "## Result Provenance",
                "",
                "| Artifact | Path | SHA256 | Description |",
                "|---|---|---|---|",
                f"| Fold-level CSV | logs/tcga_folds.csv | {'0' * 64} | seed=2026; fold=0..4 |",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--strict",
            "--require-provenance",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected strict validation to fail on checksum mismatch.")
    output = capsys.readouterr().out
    assert "Experiment result provenance: invalid" in output
    assert "checksum_mismatches=1" in output
    assert "PROVENANCE ERROR: Checksum mismatch for provenance artifact: logs/tcga_folds.csv." in output


def test_cli_validate_results_fails_on_csv_artifact_value_mismatch(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    values_csv = logs_dir / "tcga_values.csv"
    values_csv.write_text(
        "\n".join(
            [
                "method,parameter,parameter_value,comparison,dataset,metric,test,p_value,value",
                "ProtoSurv baseline,,,,BLCA,C-index,,,0.646",
                "Hyper-ProtoSurv ours,,,,BLCA,C-index,,,0.700",
                "Hyper-ProtoSurv ours,,,,Average,C-index,,,0.671",
                "w/o reconstruction loss,,,,Average,C-index,,,0.659",
                ",lambda_rec,0.5,,Average,C-index,,,0.667",
                ",lambda_rec,1.0,,Average,C-index,,,0.671",
                ",,,Hyper-ProtoSurv ours vs ProtoSurv baseline,,C-index,Wilcoxon signed-rank,0.018,",
            ]
        ),
        encoding="utf-8",
    )
    values_hash = hashlib.sha256(values_csv.read_bytes()).hexdigest()
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index |",
                "|---|---:|",
                "| ProtoSurv baseline | 0.646 |",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "| w/o reconstruction loss | 0.659 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.667 |",
                "| 1.0 | 0.671 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
                "",
                "## Result Provenance",
                "",
                "| Artifact | Path | SHA256 | Description |",
                "|---|---|---|---|",
                f"| Result values CSV | logs/tcga_values.csv | {values_hash} | source values for paper table |",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--strict",
            "--require-provenance",
            "--require-artifact-consistency",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected strict validation to fail on artifact value mismatch.")
    output = capsys.readouterr().out
    assert "Experiment artifact consistency: invalid" in output
    assert "Artifact consistency coverage: matched=6/7; missing=0; mismatched=1; aggregated=0; csv_artifacts=1" in output
    assert "ARTIFACT CONSISTENCY ERROR: Artifact value mismatch for main_method Hyper-ProtoSurv ours BLCA C-INDEX" in output


def test_cli_validate_results_strict_fails_for_template_todos(monkeypatch, tmp_path, capsys):
    results_path = tmp_path / "tcga_results_template.md"
    results_path.write_text(
        cli_module.experiment_results_template(datasets=["BLCA", "BRCA"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--strict",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected strict validation to fail for TODO template.")
    output = capsys.readouterr().out
    assert "Experiment evidence kind: unstructured" in output
    assert "Experiment result contract: invalid" in output


def test_cli_validate_results_can_disable_optional_contract_requirements(monkeypatch, tmp_path, capsys):
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index |",
                "|---|---:|",
                "| ProtoSurv baseline | 0.646 |",
                "| Hyper-ProtoSurv ours | 0.671 |",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--strict",
            "--no-require-ablation",
            "--no-require-sensitivity",
            "--no-require-statistical-tests",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert "Experiment result contract: complete" in output
    assert "Requirements: ablation=False; sensitivity=False; statistical_tests=False" in output


def test_cli_validate_results_reports_expected_quality_failures(monkeypatch, tmp_path, capsys):
    results_path = tmp_path / "tcga_results.md"
    results_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index |",
                "|---|---:|",
                "| ProtoSurv baseline | 0.646 |",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.671 |",
                "| w/o reconstruction loss | 0.659 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.667 |",
                "| 1.0 | 0.671 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "validate-results",
            "--experiment-results",
            str(results_path),
            "--summary",
            str(summary_path),
            "--strict",
            "--expected-dataset",
            "BLCA",
            "--expected-dataset",
            "BRCA",
            "--expected-metric",
            "C-INDEX",
            "--expected-method",
            "Hyper-ProtoSurv",
            "--expected-baseline",
            "ProtoSurv",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected strict validation to fail on missing expected dataset.")
    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "Experiment result quality: invalid" in output
    assert "QUALITY ERROR: Missing expected datasets: BRCA." in output
    assert summary["experiment_quality"]["status"] == "invalid"
    assert summary["experiment_quality"]["checks"]["missing_datasets"] == ["BRCA"]


def test_cli_draft_strict_results_fails_before_workflow(monkeypatch, tmp_path, capsys):
    baseline_dir = tmp_path / "baseline"
    code_dir = tmp_path / "code"
    baseline_dir.mkdir()
    code_dir.mkdir()
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (code_dir / "train.py").write_text("class HyperProtoSurv: pass\n", encoding="utf-8")
    experiment_path = tmp_path / "tcga_results_template.md"
    experiment_path.write_text(
        cli_module.experiment_results_template(datasets=["BLCA", "BRCA"]),
        encoding="utf-8",
    )

    class FakeWorkflow:
        def run(self, request):
            raise AssertionError("Workflow should not run after strict result preflight failure.")

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "draft",
            "--project-name",
            "hyper-protosurv-tcga",
            "--target-venue",
            "TPAMI",
            "--baseline",
            str(baseline_dir),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--strict-results",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected strict draft result preflight to fail.")
    output = capsys.readouterr().out
    assert "Experiment result contract: invalid" in output


def test_cli_draft_strict_results_uses_optional_contract_requirements(monkeypatch, tmp_path, capsys):
    baseline_dir = tmp_path / "baseline"
    code_dir = tmp_path / "code"
    latex_dir = tmp_path / "latex"
    baseline_dir.mkdir()
    code_dir.mkdir()
    latex_dir.mkdir()
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (code_dir / "train.py").write_text("class HyperProtoSurv: pass\n", encoding="utf-8")
    experiment_path = tmp_path / "tcga_results.md"
    experiment_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index |",
                "|---|---:|",
                "| ProtoSurv baseline | 0.646 |",
                "| Hyper-ProtoSurv ours | 0.671 |",
            ]
        ),
        encoding="utf-8",
    )
    (latex_dir / "main.tex").write_text("\\documentclass{IEEEtran}", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "llm_self_review": {"mode": "disabled"},
                    "experiment_result_tables": [{"title": "Main results"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "draft",
            "--project-name",
            "hyper-protosurv-tcga",
            "--target-venue",
            "TPAMI",
            "--baseline",
            str(baseline_dir),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--strict-results",
            "--no-require-ablation",
            "--no-require-sensitivity",
            "--no-require-statistical-tests",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert captured["request"].project_name == "hyper-protosurv-tcga"
    assert "Experiment result contract: complete" in output
    assert "LaTeX written to" in output


def test_cli_draft_enforces_min_llm_sections(monkeypatch, tmp_path):
    baseline_dir = tmp_path / "baseline"
    code_dir = tmp_path / "code"
    output_dir = tmp_path / "out"
    latex_dir = tmp_path / "latex"
    baseline_dir.mkdir()
    code_dir.mkdir()
    latex_dir.mkdir()
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (code_dir / "train.py").write_text("class HyperProtoSurv: pass\n", encoding="utf-8")
    experiment_path = tmp_path / "tcga_results.md"
    experiment_path.write_text(
        "| Method | BLCA C-index |\n"
        "|---|---:|\n"
        "| baseline | 0.646 |\n"
        "| ours | 0.671 |\n",
        encoding="utf-8",
    )
    (latex_dir / "main.tex").write_text("\\documentclass{IEEEtran}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")

    class FakeWorkflow:
        def run(self, request):
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "partial_llm",
                    "section_writer_llm_attempted_sections": ["abstract", "method"],
                    "section_writer_llm_successes": ["abstract"],
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "draft",
            "--project-name",
            "hyper-protosurv-tcga",
            "--target-venue",
            "TPAMI",
            "--baseline",
            str(baseline_dir),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--output",
            str(output_dir / "draft.md"),
            "--allow-llm",
            "--min-llm-sections",
            "2",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "expected at least 2 LLM-written sections" in str(exc)
    else:
        raise AssertionError("Expected draft command to fail when LLM section count is too low.")
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "| LLM section drafting | FAIL | 1/2 sections succeeded; required >= 2" in acceptance_report


def test_cli_sample_hyper_protosurv_writes_showcase_artifacts(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    dataset_dir = code_dir / "dataset_csv"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (dataset_dir / "BLCA.csv").write_text(
        ",case_id,slide_id,survival_months,censorship\n"
        "0,TCGA-AA-0001-01Z-00-DX1.A,TCGA-AA-0001-01Z-00-DX1.A.svs,12.0,0\n",
        encoding="utf-8",
    )
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
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--allow-llm",
            "--skip-llm-self-review",
        ],
    )

    cli_module.main()

    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert (output_dir / "draft.md").read_text(encoding="utf-8") == "# Draft"
    assert zip_path.exists()
    assert captured["request"].project_name == cli_module._default_project_name(output_dir)
    assert captured["request"].baseline_pdf_path.endswith("baseline.pdf")
    assert captured["request"].code_path.endswith("hyper-protosurv")
    assert "TCGA Cohort Data Summary" in captured["request"].experiment_results
    assert "not a model-performance result file" in captured["request"].experiment_results
    assert captured["request"].skip_llm_self_review
    assert summary["inputs"]["experiment_results_source"] == "tcga_cohort_csv"
    assert summary["inputs"]["experiment_results_path"].endswith("dataset_csv")
    assert summary["inputs"]["experiment_evidence_kind"] == "data_only"
    assert summary["llm_self_review_mode"] == "disabled"
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "Acceptance report written to" in capsys.readouterr().out
    assert "# Paper Agent Acceptance Report" in acceptance_report
    assert summary["outputs"]["acceptance_report_path"].endswith("ACCEPTANCE_REPORT.md")


def test_cli_sample_hyper_protosurv_strict_results_rejects_cohort_metadata(
    monkeypatch,
    tmp_path,
    capsys,
):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    dataset_dir = code_dir / "dataset_csv"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (dataset_dir / "BLCA.csv").write_text(
        ",case_id,slide_id,survival_months,censorship\n"
        "0,TCGA-AA-0001-01Z-00-DX1.A,TCGA-AA-0001-01Z-00-DX1.A.svs,12.0,0\n",
        encoding="utf-8",
    )

    class FakeWorkflow:
        def run(self, request):
            raise AssertionError("Workflow should not run when strict result preflight fails.")

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "sample-hyper-protosurv",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--zip",
            str(tmp_path / "sample.zip"),
            "--strict-results",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected strict sample run to fail on TCGA cohort metadata.")
    output = capsys.readouterr().out
    assert "Experiment evidence kind: data_only" in output


def test_default_project_name_uses_parent_for_generic_output_dir():
    assert (
        cli_module._default_project_name(Path("outputs") / "hyper-protosurv-tcga-real")
        == "hyper-protosurv-tcga-real"
    )
    assert (
        cli_module._default_project_name(Path("outputs") / "heartbeat-tcga-smoke" / "out")
        == "heartbeat-tcga-smoke-out"
    )
    assert cli_module._default_project_name(Path("outputs") / "run-1" / "results") == "run-1-results"


def test_cli_tcga_draft_uses_default_result_path_and_writes_reports(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    results_dir = example_root / "results"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (results_dir / "tcga_results.md").write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index | BRCA C-index | LGG C-index | LUAD C-index | UCEC C-index |",
                "|---|---:|---:|---:|---:|---:|",
                "| ProtoSurv baseline | 0.646 | 0.669 | 0.724 | 0.636 | 0.658 |",
                "| Hyper-ProtoSurv ours | 0.671 | 0.691 | 0.746 | 0.661 | 0.681 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.690 |",
                "| w/o reconstruction loss | 0.672 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.687 |",
                "| 1.0 | 0.690 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
            ]
        ),
        encoding="utf-8",
    )
    latex_dir = tmp_path / "latex"
    latex_dir.mkdir()
    (latex_dir / "main.tex").write_text("\\documentclass{IEEEtran}", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_available"] = bool(llm_client and llm_client.available)

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "llm",
                    "section_writer_llm_attempted_sections": [
                        "abstract",
                        "method",
                        "experiments",
                        "conclusion",
                    ],
                    "section_writer_llm_successes": [
                        "abstract",
                        "method",
                        "experiments",
                        "conclusion",
                    ],
                    "llm_self_review": {"mode": "llm"},
                    "experiment_result_tables": [{"title": "Main results"}],
                    "experiment_ablation_evidence": [{"variant": "w/o reconstruction loss"}],
                    "experiment_sensitivity_evidence": [{"parameter": "lambda_rec"}],
                    "experiment_statistical_tests": [{"comparison": "ours vs baseline"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    output_dir = tmp_path / "tcga-real"
    zip_path = tmp_path / "tcga-real.zip"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli_module, "_llm_preflight_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-draft",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--min-llm-sections",
            "4",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "TCGA draft run completed." in output
    assert "Experiment result contract: complete" in output
    assert captured["llm_available"]
    assert captured["request"].project_name == cli_module._default_project_name(output_dir)
    assert captured["request"].experiment_results
    assert not captured["request"].skip_llm_self_review
    assert (output_dir / "draft.md").read_text(encoding="utf-8") == "# Draft"
    assert zip_path.exists()
    assert summary["project_name"] == cli_module._default_project_name(output_dir)
    assert summary["inputs"]["experiment_results_path"].endswith("tcga_results.md")
    assert summary["inputs"]["experiment_evidence_kind"] == "real_result_file"
    assert summary["experiment_contract_status"] == "complete"
    assert summary["experiment_quality_status"] == "complete"
    assert "- Submission evidence status: PASS" in acceptance_report
    assert "| Experiment result quality | PASS | complete;" in acceptance_report


def test_cli_tcga_draft_uses_artifact_flow_summary_when_results_omitted(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")

    flow_dir = tmp_path / "artifact-flow"
    flow_dir.mkdir()
    result_path = flow_dir / "tcga_results.md"
    result_path.write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index | BRCA C-index | LGG C-index | LUAD C-index | UCEC C-index |",
                "|---|---:|---:|---:|---:|---:|",
                "| ProtoSurv baseline | 0.646 | 0.669 | 0.724 | 0.636 | 0.658 |",
                "| Hyper-ProtoSurv ours | 0.671 | 0.691 | 0.746 | 0.661 | 0.681 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.690 |",
                "| w/o reconstruction loss | 0.672 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.687 |",
                "| 1.0 | 0.690 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
            ]
        ),
        encoding="utf-8",
    )
    flow_summary_path = flow_dir / "RUN_SUMMARY.json"
    flow_summary_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "pipeline_phase": "tcga_artifact_flow_demo",
                "experiment_results": str(result_path),
                "artifacts_dir": str(flow_dir / "artifacts"),
                "artifact_files": {
                    "tcga_main_results.csv": str(flow_dir / "artifacts" / "tcga_main_results.csv"),
                },
                "experiment_contract_status": "complete",
                "experiment_quality_status": "complete",
                "experiment_provenance_status": "complete",
                "experiment_artifact_consistency_status": "complete",
                "artifact_consistency_matched": 15,
                "artifact_consistency_missing": 0,
                "artifact_consistency_mismatched": 0,
            }
        ),
        encoding="utf-8",
    )

    latex_dir = tmp_path / "latex"
    latex_dir.mkdir()
    (latex_dir / "main.tex").write_text("\\documentclass{IEEEtran}", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_client"] = llm_client

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "llm_self_review": {"mode": "disabled"},
                    "experiment_result_tables": [{"title": "Main results"}],
                    "experiment_ablation_evidence": [{"variant": "w/o reconstruction loss"}],
                    "experiment_sensitivity_evidence": [{"parameter": "lambda_rec"}],
                    "experiment_statistical_tests": [{"comparison": "ours vs baseline"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    output_dir = tmp_path / "tcga-draft"
    zip_path = tmp_path / "tcga-draft.zip"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-draft",
            "--example-root",
            str(example_root),
            "--artifact-flow-summary",
            str(flow_summary_path),
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--disable-llm",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "TCGA artifact flow summary loaded from" in output
    assert "TCGA draft run completed." in output
    assert captured["llm_client"] is None
    assert captured["request"].experiment_results
    assert summary["inputs"]["experiment_results_path"] == str(result_path)
    assert summary["inputs"]["tcga_artifact_flow_summary_path"] == str(flow_summary_path)
    assert summary["inputs"]["tcga_artifact_flow_summary_status"] == "pass"
    assert summary["inputs"]["tcga_artifact_flow_pipeline_phase"] == "tcga_artifact_flow_demo"
    assert summary["inputs"]["tcga_artifact_flow_validation"]["artifact_consistency_matched"] == 15
    assert f"- TCGA artifact-flow summary: {flow_summary_path}" in acceptance_report
    assert "| TCGA artifact flow trace | PASS |" in acceptance_report
    assert "phase=tcga_artifact_flow_demo" in acceptance_report
    assert "matched=15; missing=0; mismatched=0" in acceptance_report
    assert zip_path.exists()


def test_cli_tcga_demo_paper_runs_full_local_package(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")

    latex_dir = tmp_path / "latex"
    latex_dir.mkdir()
    (latex_dir / "main.tex").write_text("\\documentclass{IEEEtran}", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_client"] = llm_client

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "llm_self_review": {"mode": "disabled"},
                    "experiment_result_tables": [{"title": "Main results"}],
                    "experiment_ablation_evidence": [{"variant": "w/o reconstruction loss"}],
                    "experiment_sensitivity_evidence": [{"parameter": "lambda_rec"}],
                    "experiment_statistical_tests": [{"comparison": "ours vs baseline"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    output_dir = tmp_path / "demo-paper"
    zip_path = tmp_path / "demo-paper.zip"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-demo-paper",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--force",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    top_summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    draft_summary = json.loads((output_dir / "draft" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    acceptance_report = (output_dir / "draft" / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "TCGA demo paper: generating artifact flow" in output
    assert "TCGA demo paper: drafting paper package" in output
    assert "TCGA demo paper completed." in output
    assert captured["llm_client"] is None
    assert captured["request"].target_venue == "TPAMI"
    assert (output_dir / "artifact-flow" / "RUN_SUMMARY.json").is_file()
    assert (output_dir / "draft" / "draft.md").read_text(encoding="utf-8") == "# Draft"
    assert zip_path.exists()
    assert top_summary["status"] == "pass"
    assert top_summary["pipeline_phase"] == "tcga_demo_paper"
    assert top_summary["artifact_flow_summary_path"].endswith("artifact-flow\\RUN_SUMMARY.json")
    assert top_summary["draft_summary_path"].endswith("draft\\RUN_SUMMARY.json")
    assert top_summary["outputs"]["acceptance_report_path"].endswith("ACCEPTANCE_REPORT.md")
    assert top_summary["llm_mode"] == "disabled"
    assert "workflow validation only" in top_summary["note"]
    assert draft_summary["inputs"]["tcga_artifact_flow_summary_status"] == "pass"
    assert "| TCGA artifact flow trace | PASS |" in acceptance_report


def test_cli_tcga_submission_grade_forces_full_acceptance_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    results_dir = example_root / "results"
    logs_dir = results_dir / "logs"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    main_csv = logs_dir / "tcga_main_wide.csv"
    ablation_csv = logs_dir / "tcga_ablation_wide.csv"
    sensitivity_csv = logs_dir / "tcga_sensitivity_wide.csv"
    stats_csv = logs_dir / "tcga_stats.csv"
    main_csv.write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index,LGG C-index,LUAD C-index,UCEC C-index",
                "ProtoSurv baseline,0.646,0.669,0.724,0.636,0.658",
                "Hyper-ProtoSurv ours,0.671,0.691,0.746,0.661,0.681",
            ]
        ),
        encoding="utf-8",
    )
    ablation_csv.write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.690",
                "w/o reconstruction loss,0.672",
            ]
        ),
        encoding="utf-8",
    )
    sensitivity_csv.write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.687",
                "1.0,0.690",
            ]
        ),
        encoding="utf-8",
    )
    stats_csv.write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    main_hash = hashlib.sha256(main_csv.read_bytes()).hexdigest()
    ablation_hash = hashlib.sha256(ablation_csv.read_bytes()).hexdigest()
    sensitivity_hash = hashlib.sha256(sensitivity_csv.read_bytes()).hexdigest()
    stats_hash = hashlib.sha256(stats_csv.read_bytes()).hexdigest()
    (results_dir / "tcga_results.md").write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "Metric: C-index. Higher is better.",
                "",
                "| Method | BLCA C-index | BRCA C-index | LGG C-index | LUAD C-index | UCEC C-index |",
                "|---|---:|---:|---:|---:|---:|",
                "| ProtoSurv baseline | 0.646 | 0.669 | 0.724 | 0.636 | 0.658 |",
                "| Hyper-ProtoSurv ours | 0.671 | 0.691 | 0.746 | 0.661 | 0.681 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.690 |",
                "| w/o reconstruction loss | 0.672 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.687 |",
                "| 1.0 | 0.690 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
                "",
                "## Result Provenance",
                "",
                "| Artifact | Path | SHA256 | Description |",
                "|---|---|---|---|",
                f"| Main wide CSV | logs/tcga_main_wide.csv | {main_hash} | seed=2026; fold=0..4 |",
                f"| Ablation wide CSV | logs/tcga_ablation_wide.csv | {ablation_hash} | seed=2026; fold=0..4 |",
                f"| Sensitivity wide CSV | logs/tcga_sensitivity_wide.csv | {sensitivity_hash} | seed=2026; fold=0..4 |",
                f"| Statistical CSV | logs/tcga_stats.csv | {stats_hash} | seed=2026; fold=0..4 |",
            ]
        ),
        encoding="utf-8",
    )
    latex_dir = tmp_path / "latex"
    latex_dir.mkdir()
    (latex_dir / "main.tex").write_text("\\documentclass{IEEEtran}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_available"] = bool(llm_client and llm_client.available)

        def run(self, request):
            captured["request"] = request
            captured["disable_template_fetch"] = os.getenv("PAPER_AGENT_DISABLE_TEMPLATE_FETCH")
            captured["disable_reference_resolve"] = os.getenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE")
            captured["disable_related_work_discovery"] = os.getenv("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY")
            captured["compile_latex"] = os.getenv("PAPER_AGENT_RUN_LATEX_COMPILE")
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "llm",
                    "section_writer_llm_attempted_sections": [
                        "abstract",
                        "introduction",
                        "related_work",
                        "method",
                    ],
                    "section_writer_llm_successes": [
                        "abstract",
                        "introduction",
                        "related_work",
                        "method",
                    ],
                    "llm_self_review": {"mode": "llm"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                    "experiment_ablation_evidence": [{"variant": "w/o reconstruction loss"}],
                    "experiment_sensitivity_evidence": [{"parameter": "lambda_rec"}],
                    "experiment_statistical_tests": [{"comparison": "ours vs baseline"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    def fake_write_latex_zip_and_refresh(state, zip_path):
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "w") as archive:
            archive.writestr("main.tex", "\\documentclass{IEEEtran}")
        artifacts = state.setdefault("artifacts", {})
        artifacts["submission_package"] = {
            "status": "valid",
            "errors": [],
            "warnings": [],
            "checks": {
                "compile": {
                    "mode": "compile",
                    "status": "passed",
                    "tool": "tectonic.exe",
                }
            },
        }
        artifacts["submission_readiness"] = {"overall_score": 100, "status": "reviewable"}
        state["latex_zip_path"] = zip_path
        return zip_path

    output_dir = tmp_path / "submission-grade"
    zip_path = tmp_path / "submission-grade.zip"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli_module, "_write_latex_zip_and_refresh", fake_write_latex_zip_and_refresh)
    monkeypatch.setattr(cli_module, "_llm_preflight_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-draft",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--min-llm-sections",
            "1",
            "--submission-grade",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "TCGA draft run completed." in output
    assert "Submission grade: True" in output
    assert captured["llm_available"]
    assert captured["disable_template_fetch"] == "0"
    assert captured["disable_reference_resolve"] == "0"
    assert captured["disable_related_work_discovery"] == "0"
    assert captured["compile_latex"] == "1"
    assert captured["request"].skip_llm_self_review is False
    assert summary["inputs"]["network_mode"] == "online"
    assert summary["inputs"]["submission_grade"] is True
    assert summary["inputs"]["latex_compile_requested"] is True
    assert summary["inputs"]["min_llm_sections"] == 4
    assert summary["experiment_provenance_status"] == "complete"
    assert summary["experiment_artifact_consistency_status"] == "complete"
    assert "- Submission grade: True" in acceptance_report
    assert "| LLM self-review | PASS | mode=llm;" in acceptance_report
    assert zip_path.exists()


def test_cli_tcga_pipeline_generates_results_and_runs_submission_grade(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    logs_dir = example_root / "results" / "logs"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (logs_dir / "tcga_main_wide.csv").write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index,LGG C-index,LUAD C-index,UCEC C-index",
                "ProtoSurv baseline,0.646,0.669,0.724,0.636,0.658",
                "Hyper-ProtoSurv ours,0.671,0.691,0.746,0.661,0.681",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_ablation_wide.csv").write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.690",
                "w/o reconstruction loss,0.672",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_sensitivity_wide.csv").write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.687",
                "1.0,0.690",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_stats.csv").write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    latex_dir = tmp_path / "latex"
    latex_dir.mkdir()
    (latex_dir / "main.tex").write_text("\\documentclass{IEEEtran}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_available"] = bool(llm_client and llm_client.available)

        def run(self, request):
            captured["request"] = request
            captured["disable_template_fetch"] = os.getenv("PAPER_AGENT_DISABLE_TEMPLATE_FETCH")
            captured["disable_reference_resolve"] = os.getenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE")
            captured["disable_related_work_discovery"] = os.getenv("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY")
            captured["compile_latex"] = os.getenv("PAPER_AGENT_RUN_LATEX_COMPILE")
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "llm",
                    "section_writer_llm_attempted_sections": [
                        "abstract",
                        "introduction",
                        "related_work",
                        "method",
                    ],
                    "section_writer_llm_successes": [
                        "abstract",
                        "introduction",
                        "related_work",
                        "method",
                    ],
                    "llm_self_review": {"mode": "llm"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                    "experiment_ablation_evidence": [{"variant": "w/o reconstruction loss"}],
                    "experiment_sensitivity_evidence": [{"parameter": "lambda_rec"}],
                    "experiment_statistical_tests": [{"comparison": "ours vs baseline"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    def fake_write_latex_zip_and_refresh(state, zip_path):
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "w") as archive:
            archive.writestr("main.tex", "\\documentclass{IEEEtran}")
        artifacts = state.setdefault("artifacts", {})
        artifacts["submission_package"] = {
            "status": "valid",
            "errors": [],
            "warnings": [],
            "checks": {"compile": {"mode": "compile", "status": "passed", "tool": "tectonic.exe"}},
        }
        artifacts["submission_readiness"] = {"overall_score": 100, "status": "reviewable"}
        state["latex_zip_path"] = zip_path
        return zip_path

    output_dir = tmp_path / "pipeline"
    zip_path = tmp_path / "pipeline.zip"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli_module, "_write_latex_zip_and_refresh", fake_write_latex_zip_and_refresh)
    monkeypatch.setattr(cli_module, "_llm_preflight_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli_module,
        "_latex_toolchain_status",
        lambda: {"available": True, "preferred_tool": "tectonic.exe", "install_hint": ""},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-pipeline",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--min-llm-sections",
            "1",
            "--submission-grade",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    result_path = example_root / "results" / "tcga_results.md"
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert "TCGA pipeline: generating result file" in output
    assert "TCGA pipeline: running doctor checks" in output
    assert "TCGA pipeline: drafting paper" in output
    assert "TCGA draft run completed." in output
    assert "TCGA pipeline completed." in output
    assert result_path.exists()
    assert "## Result Provenance" in result_path.read_text(encoding="utf-8")
    assert captured["llm_available"]
    assert captured["disable_template_fetch"] == "0"
    assert captured["disable_reference_resolve"] == "0"
    assert captured["disable_related_work_discovery"] == "0"
    assert captured["compile_latex"] == "1"
    assert summary["inputs"]["submission_grade"] is True
    assert summary["inputs"]["experiment_results_path"] == str(result_path)
    assert summary["inputs"]["min_llm_sections"] == 4
    assert summary["status"] == "pass"
    assert summary["pipeline_phase"] == "tcga_pipeline_complete"
    assert summary["pipeline"]["result_generation"] == "generated_from_artifacts"
    assert summary["pipeline"]["doctor_checks"] == "completed"
    assert summary["pipeline"]["experiment_results_path"] == str(result_path)
    assert summary["readiness_contract"]["schema_version"] == "tcga-readiness-contract/v1"
    assert summary["readiness_contract"]["status"] == "ready"
    assert summary["readiness_contract"]["pipeline_phase"] == "tcga_pipeline_complete"
    assert summary["readiness_contract"]["ready_for_submission_grade"] is True
    assert summary["readiness_contract"]["blocking_categories"] == []
    assert summary["readiness_contract"]["requirements"]["pipeline_stage"]["status"] == "pass"
    assert summary["readiness_contract"]["requirements"]["result_artifacts"]["status"] == "pass"
    assert summary["next_actions"][0]["category"] == "pipeline_stage"
    assert summary["outputs"]["pipeline_summary_path"].endswith("RUN_SUMMARY.json")
    assert summary["experiment_provenance_status"] == "complete"
    assert summary["experiment_artifact_consistency_status"] == "complete"
    assert zip_path.exists()


def test_cli_tcga_pipeline_suggests_artifact_template_when_results_missing(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    output_dir = tmp_path / "pipeline"
    example_root.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-pipeline",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(output_dir),
            "--disable-llm",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        message = str(exc)
        assert "TCGA pipeline cannot generate the paper-facing result file" in message
        assert "paper-agent tcga-artifact-template --output-dir" in message
        assert "--write-artifact-template" in message
    else:
        raise AssertionError("Expected TCGA pipeline to fail with artifact-template guidance.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert "TCGA pipeline: generating result file" in output
    assert "Pipeline summary written to" in output
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "result_artifact_detection"
    assert "paper-agent tcga-artifact-template --output-dir" in summary["next_command"]
    assert any("tcga_main_results.csv" in item for item in summary["missing_inputs"])
    assert summary["readiness_contract"]["schema_version"] == "tcga-readiness-contract/v1"
    assert summary["readiness_contract"]["status"] == "blocked"
    assert summary["readiness_contract"]["pipeline_phase"] == "result_artifact_detection"
    assert summary["readiness_contract"]["requirements"]["result_artifacts"]["status"] == "fail"
    assert summary["readiness_contract"]["requirements"]["experiment_results"]["status"] == "fail"
    assert any(action["category"] == "result_artifacts" for action in summary["next_actions"])


def test_cli_tcga_pipeline_writes_artifact_template_and_stops(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    output_dir = tmp_path / "pipeline"
    example_root.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-pipeline",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(output_dir),
            "--write-artifact-template",
            "--dataset",
            "BLCA",
            "--dataset",
            "BRCA",
            "--disable-llm",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "stopped after writing artifact templates" in str(exc)
    else:
        raise AssertionError("Expected TCGA pipeline to stop after writing artifact templates.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    logs_dir = example_root / "results" / "logs"
    assert "TCGA pipeline: result artifacts are missing or incomplete; writing artifact templates" in output
    assert "Pipeline summary written to" in output
    assert (logs_dir / "EXPORT_CONTRACT.md").is_file()
    assert (logs_dir / "ARTIFACT_SCHEMA.json").is_file()
    main_csv = (logs_dir / "tcga_main_results.csv").read_text(encoding="utf-8")
    assert "ProtoSurv baseline,BLCA,C-index,0,2026,TODO" in main_csv
    assert "Hyper-ProtoSurv ours,BRCA,C-index,0,2026,TODO" in main_csv
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "artifact_template_written"
    assert summary["outputs"]["artifact_contract_path"] == str(logs_dir / "EXPORT_CONTRACT.md")
    assert summary["outputs"]["artifact_schema_path"] == str(logs_dir / "ARTIFACT_SCHEMA.json")
    assert "paper-agent tcga-pipeline" in summary["next_command"]
    assert any("tcga_stats.csv" in item for item in summary["missing_inputs"])
    assert summary["readiness_contract"]["schema_version"] == "tcga-readiness-contract/v1"
    assert summary["readiness_contract"]["requirements"]["result_artifacts"]["status"] == "ready_to_fill"
    assert summary["readiness_contract"]["requirements"]["experiment_results"]["status"] == "ready_to_generate"


def test_cli_tcga_pipeline_writes_summary_on_doctor_failure(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    logs_dir = example_root / "results" / "logs"
    output_dir = tmp_path / "pipeline"
    logs_dir.mkdir(parents=True)
    (logs_dir / "tcga_main_wide.csv").write_text(
        "\n".join(
            [
                "method,BLCA C-index,BRCA C-index,LGG C-index,LUAD C-index,UCEC C-index",
                "ProtoSurv baseline,0.646,0.669,0.724,0.636,0.658",
                "Hyper-ProtoSurv ours,0.671,0.691,0.746,0.661,0.681",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_ablation_wide.csv").write_text(
        "\n".join(
            [
                "variant,Average C-index",
                "Hyper-ProtoSurv ours,0.690",
                "w/o reconstruction loss,0.672",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_sensitivity_wide.csv").write_text(
        "\n".join(
            [
                "lambda_rec,Average C-index",
                "0.5,0.687",
                "1.0,0.690",
            ]
        ),
        encoding="utf-8",
    )
    (logs_dir / "tcga_stats.csv").write_text(
        "\n".join(
            [
                "comparison,metric,test,p_value",
                "Hyper-ProtoSurv ours vs ProtoSurv baseline,C-index,Wilcoxon signed-rank,0.018",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-pipeline",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(output_dir),
            "--disable-llm",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA doctor failed" in str(exc)
        assert "Pipeline summary" in str(exc)
    else:
        raise AssertionError("Expected TCGA pipeline to fail during doctor checks.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert "TCGA pipeline: running doctor checks" in output
    assert "Pipeline summary written to" in output
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "doctor_checks"
    assert any("Baseline PDF" in item for item in summary["missing_inputs"])
    assert any("Hyper-ProtoSurv code directory" in item for item in summary["missing_inputs"])
    assert summary["next_command"].startswith("paper-agent tcga-doctor")
    assert summary["readiness_contract"]["schema_version"] == "tcga-readiness-contract/v1"
    assert summary["readiness_contract"]["requirements"]["pipeline_stage"]["status"] == "fail"
    assert "pipeline_stage" in summary["readiness_contract"]["blocking_categories"]


def test_cli_tcga_pipeline_writes_summary_on_draft_llm_failure(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    results_dir = example_root / "results"
    output_dir = tmp_path / "pipeline"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (results_dir / "tcga_results.md").write_text("strict result placeholder", encoding="utf-8")

    def fake_validate(*args, **kwargs):
        return {
            "experiment_evidence": {"real_result_evidence": True},
            "experiment_contract": {"status": "complete"},
            "experiment_quality": {"status": "complete"},
            "experiment_provenance": {"status": "complete"},
            "experiment_artifact_consistency": {"status": "complete"},
        }

    monkeypatch.setattr(cli_module, "_validate_results_text", fake_validate)
    monkeypatch.setattr(
        cli_module,
        "load_llm_config",
        lambda: LLMConfig(api_key="", base_url="https://api.deepseek.com", model="deepseek-v4-pro"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-pipeline",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(output_dir),
            "--skip-result-generation",
            "--skip-doctor",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA draft requires a configured LLM" in str(exc)
        assert "Pipeline summary" in str(exc)
    else:
        raise AssertionError("Expected TCGA pipeline to fail during LLM preflight.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert "TCGA pipeline: drafting paper" in output
    assert "Pipeline summary written to" in output
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "llm_preflight"
    assert summary["next_command"] == "paper-agent llm-doctor"
    assert any("LLM API key" in item for item in summary["missing_inputs"])
    assert summary["readiness_contract"]["schema_version"] == "tcga-readiness-contract/v1"
    assert summary["readiness_contract"]["requirements"]["llm"]["status"] == "fail"
    assert summary["readiness_contract"]["requirements"]["llm"]["command"] == "paper-agent llm-doctor"
    assert summary["diagnostics"]["llm"]["failure_kind"] == "configuration"
    assert summary["diagnostics"]["llm"]["provider"] == "deepseek"
    assert summary["diagnostics"]["llm"]["model"] == "deepseek-v4-pro"
    assert summary["diagnostics"]["llm"]["configured"] is False


def test_cli_tcga_submission_grade_rejects_disabled_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-draft",
            "--submission-grade",
            "--disable-llm",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "submission-grade runs require the configured LLM" in str(exc)
    else:
        raise AssertionError("Expected submission-grade TCGA draft to reject --disable-llm.")


def test_llm_preflight_reports_insufficient_balance_without_api_key():
    class FailingClient:
        def chat(self, *args, **kwargs):
            raise cli_module.LLMError(
                'LLM HTTP 402: {"error":{"message":"Insufficient Balance for secret-key","code":"invalid_request_error"}}'
            )

    config = LLMConfig(
        api_key="secret-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
    )

    try:
        cli_module._llm_preflight_check(FailingClient(), config, context="TCGA draft")
    except SystemExit as exc:
        message = str(exc)
        assert "TCGA draft LLM preflight failed" in message
        assert "balance or quota is insufficient" in message
        assert "Failure kind: quota" in message
        assert "deepseek/deepseek-v4-pro" in message
        assert "secret-key" not in message
        assert "[redacted-api-key]" in message
    else:
        raise AssertionError("Expected LLM preflight to fail on provider 402.")


def test_llm_failure_diagnostics_classifies_and_sanitizes_provider_errors(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    config = LLMConfig(
        api_key="secret-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        timeout_seconds=45,
        connect_timeout_seconds=7,
        max_retries=5,
        retry_base_seconds=0.25,
    )

    diagnostics = cli_module._llm_failure_diagnostics(
        config,
        "LLM HTTP 402: Insufficient Balance for secret-key",
    )

    assert diagnostics["failure_kind"] == "quota"
    assert diagnostics["provider"] == "deepseek"
    assert diagnostics["model"] == "deepseek-v4-pro"
    assert diagnostics["endpoint_host"] == "api.deepseek.com"
    assert diagnostics["configured"] is True
    assert diagnostics["timeout_seconds"] == 45
    assert diagnostics["connect_timeout_seconds"] == 7
    assert diagnostics["max_retries"] == 5
    assert diagnostics["retry_base_seconds"] == 0.25
    assert "secret-key" not in json.dumps(diagnostics)
    assert "[redacted-api-key]" in diagnostics["raw_error"]


def test_llm_preflight_returns_usage_and_elapsed(monkeypatch):
    class PassingClient:
        def chat(self, *args, **kwargs):
            return SimpleNamespace(
                content="paper-agent-ok",
                model="deepseek-v4-pro",
                usage={"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            )

    times = iter([10.0, 10.257])
    monkeypatch.setattr(cli_module.time, "perf_counter", lambda: next(times))
    config = LLMConfig(api_key="test-key", base_url="https://api.deepseek.com", model="deepseek-v4-pro")

    diagnostics = cli_module._llm_preflight_check(PassingClient(), config, context="LLM doctor")

    assert diagnostics["elapsed_seconds"] == 0.257
    assert diagnostics["response_model"] == "deepseek-v4-pro"
    assert diagnostics["usage"] == {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}


def test_cli_llm_doctor_no_live_prints_static_config(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "doctor-secret")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    summary_file = tmp_path / "llm-doctor-summary.json"
    monkeypatch.setattr(
        "sys.argv",
        ["paper-agent", "llm-doctor", "--no-live", "--summary", str(summary_file)],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "LLM configuration:" in output
    assert "Provider/model: deepseek/deepseek-v4-pro at api.deepseek.com" in output
    assert "API key: configured via DEEPSEEK_API_KEY" in output
    assert "Configured: True" in output
    assert "Live preflight: SKIPPED (--no-live)" in output
    assert "LLM doctor summary written to" in output
    assert summary["status"] == "pass"
    assert summary["llm"]["provider"] == "deepseek"
    assert summary["llm"]["model"] == "deepseek-v4-pro"
    assert summary["llm"]["configured"] is True
    assert summary["llm"]["api_key_source"] == "configured via DEEPSEEK_API_KEY"
    assert summary["live_preflight"]["status"] == "skipped"
    assert "diagnostics" not in summary["live_preflight"]
    assert "doctor-secret" not in output
    assert "doctor-secret" not in json.dumps(summary)


def test_cli_llm_doctor_writes_live_pass_usage_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "doctor-secret")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    class PassingClient:
        def __init__(self, config):
            self.config = config

        def chat(self, *args, **kwargs):
            return SimpleNamespace(
                content="paper-agent-ok",
                model="deepseek-v4-pro",
                usage={"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            )

    times = iter([20.0, 20.125])
    summary_file = tmp_path / "llm-doctor-live-pass-summary.json"
    monkeypatch.setattr(cli_module, "LLMClient", PassingClient)
    monkeypatch.setattr(cli_module.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr("sys.argv", ["paper-agent", "llm-doctor", "--summary", str(summary_file)])

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "Live preflight: PASS" in output
    assert summary["status"] == "pass"
    assert summary["live_preflight"]["status"] == "pass"
    assert summary["live_preflight"]["elapsed_seconds"] == 0.125
    assert summary["live_preflight"]["response_model"] == "deepseek-v4-pro"
    assert summary["live_preflight"]["usage"]["total_tokens"] == 12
    assert "diagnostics" not in summary["live_preflight"]
    assert "doctor-secret" not in json.dumps(summary)


def test_cli_llm_live_smoke_writes_pass_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "smoke-secret")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    class PassingClient:
        def __init__(self, config):
            self.config = config

        def chat(self, messages, **kwargs):
            assert kwargs["temperature"] == 0.0
            assert kwargs["max_tokens"] == 32
            return SimpleNamespace(
                content="paper-agent-live-ok",
                model="deepseek-v4-pro",
                usage={"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
            )

    times = iter([30.0, 30.333])
    summary_file = tmp_path / "llm-live-smoke.json"
    monkeypatch.setattr(cli_module, "LLMClient", PassingClient)
    monkeypatch.setattr(cli_module.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr("sys.argv", ["paper-agent", "llm-live-smoke", "--summary", str(summary_file)])

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "LLM live smoke: PASS" in output
    assert summary["status"] == "pass"
    assert summary["llm"]["provider"] == "deepseek"
    assert summary["request"]["expected_response"] == "paper-agent-live-ok"
    assert summary["live_call"]["matched_expectation"] is True
    assert summary["live_call"]["elapsed_seconds"] == 0.333
    assert summary["live_call"]["usage"]["total_tokens"] == 14
    assert "smoke-secret" not in json.dumps(summary)


def test_cli_llm_live_smoke_writes_failure_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "smoke-secret")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    class FailingClient:
        def __init__(self, config):
            self.config = config

        def chat(self, *args, **kwargs):
            raise cli_module.LLMError("LLM HTTP 402: Insufficient Balance for smoke-secret")

    times = iter([40.0, 40.25])
    summary_file = tmp_path / "llm-live-smoke-fail.json"
    monkeypatch.setattr(cli_module, "LLMClient", FailingClient)
    monkeypatch.setattr(cli_module.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr("sys.argv", ["paper-agent", "llm-live-smoke", "--summary", str(summary_file)])

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "LLM live smoke failed" in str(exc)
    else:
        raise AssertionError("Expected llm-live-smoke to fail when provider call fails.")

    output = capsys.readouterr().out
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "LLM live smoke: FAIL" in output
    assert summary["status"] == "fail"
    assert summary["live_call"]["elapsed_seconds"] == 0.25
    assert summary["live_call"]["diagnostics"]["failure_kind"] == "quota"
    assert "smoke-secret" not in json.dumps(summary)


def test_cli_llm_doctor_reports_live_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "doctor-secret")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    class FailingClient:
        def __init__(self, config):
            self.config = config

        def chat(self, *args, **kwargs):
            raise cli_module.LLMError("LLM HTTP 402: Insufficient Balance for doctor-secret")

    monkeypatch.setattr(cli_module, "LLMClient", FailingClient)
    summary_file = tmp_path / "llm-doctor-live-failure-summary.json"
    monkeypatch.setattr("sys.argv", ["paper-agent", "llm-doctor", "--summary", str(summary_file)])

    try:
        cli_module.main()
    except SystemExit as exc:
        message = str(exc)
        assert "LLM doctor LLM preflight failed" in message
        assert "balance or quota is insufficient" in message
        assert "doctor-secret" not in message
    else:
        raise AssertionError("Expected llm-doctor to fail when live provider preflight fails.")
    output = capsys.readouterr().out
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "LLM configuration:" in output
    assert "Live preflight: FAIL" in output
    assert "LLM doctor summary written to" in output
    assert summary["status"] == "fail"
    assert summary["live_preflight"]["status"] == "fail"
    assert summary["live_preflight"]["diagnostics"]["failure_kind"] == "quota"
    assert summary["live_preflight"]["diagnostics"]["provider"] == "deepseek"
    assert "doctor-secret" not in json.dumps(summary)


def test_cli_tcga_draft_stops_before_workflow_on_llm_preflight_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    results_dir = example_root / "results"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    result_path = results_dir / "tcga_results.md"
    result_path.write_text("real result placeholder", encoding="utf-8")

    def fake_validate(*args, **kwargs):
        return {
            "experiment_evidence": {"real_result_evidence": True},
            "experiment_contract": {"status": "complete"},
            "experiment_quality": {"status": "complete"},
            "experiment_provenance": {"status": "complete"},
            "experiment_artifact_consistency": {"status": "complete"},
        }

    def fail_preflight(*args, **kwargs):
        raise SystemExit("TCGA draft LLM preflight failed for deepseek/deepseek-v4-pro: quota blocked.")

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            raise AssertionError("Workflow should not be constructed after LLM preflight failure.")

    monkeypatch.setattr(cli_module, "_validate_results_text", fake_validate)
    monkeypatch.setattr(cli_module, "_llm_preflight_check", fail_preflight)
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-draft",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "LLM preflight failed" in str(exc)
    else:
        raise AssertionError("Expected tcga-draft to stop at LLM preflight failure.")


def test_cli_tcga_draft_fails_when_default_result_file_is_missing(monkeypatch, tmp_path):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")

    class FakeWorkflow:
        def run(self, request):
            raise AssertionError("Workflow should not run without a real TCGA result file.")

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-draft",
            "--example-root",
            str(example_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--disable-llm",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA experiment results file not found" in str(exc)
        assert "experiment-template" in str(exc)
    else:
        raise AssertionError("Expected tcga-draft to fail without default result file.")


def test_cli_tcga_doctor_writes_missing_result_template(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-doctor",
            "--example-root",
            str(example_root),
            "--write-template",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA doctor failed" in str(exc)
    else:
        raise AssertionError("Expected tcga-doctor to fail until generated TODOs are filled.")

    output = capsys.readouterr().out
    result_path = example_root / "results" / "tcga_results.md"
    assert "TCGA project doctor:" in output
    assert "Result template written" in output
    assert "Overall: FAIL" in output
    assert result_path.exists()
    assert "TODO" in result_path.read_text(encoding="utf-8")


def test_cli_tcga_doctor_summary_records_next_actions_for_todo_template(monkeypatch, tmp_path, capsys):
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    results_dir = example_root / "results"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (results_dir / "tcga_results.md").write_text(
        cli_module.experiment_results_template(datasets=["BLCA", "BRCA"]),
        encoding="utf-8",
    )
    summary_path = tmp_path / "tcga-doctor-template-summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-doctor",
            "--example-root",
            str(example_root),
            "--summary",
            str(summary_path),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA doctor failed" in str(exc)
    else:
        raise AssertionError("Expected tcga-doctor to fail on TODO template.")

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "Next actions:" in output
    assert summary["next_action"].startswith("Replace every TODO")
    assert summary["next_command"].startswith("paper-agent tcga-artifact-template")
    assert summary["next_actions"][0]["phase"] == "result_validation"
    assert summary["next_actions"][0]["has_todo_placeholders"] is True
    assert "tcga-artifact-template" in summary["next_actions"][0]["artifact_template_command"]
    assert "tcga-results-from-artifacts" in summary["next_actions"][0]["results_from_artifacts_command"]
    assert summary["next_actions"][0]["validation_command"].startswith("paper-agent validate-results")


def test_cli_tcga_doctor_passes_complete_local_inputs(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    results_dir = example_root / "results"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (results_dir / "tcga_results.md").write_text(
        "\n".join(
            [
                "## Main Results",
                "",
                "| Method | BLCA C-index | BRCA C-index | LGG C-index | LUAD C-index | UCEC C-index |",
                "|---|---:|---:|---:|---:|---:|",
                "| ProtoSurv baseline | 0.646 | 0.669 | 0.724 | 0.636 | 0.658 |",
                "| Hyper-ProtoSurv ours | 0.671 | 0.691 | 0.746 | 0.661 | 0.681 |",
                "",
                "## Ablation Study",
                "",
                "| Variant | Average C-index |",
                "|---|---:|",
                "| Hyper-ProtoSurv ours | 0.690 |",
                "| w/o reconstruction loss | 0.672 |",
                "",
                "## Sensitivity Analysis",
                "",
                "| lambda_rec | Average C-index |",
                "|---:|---:|",
                "| 0.5 | 0.687 |",
                "| 1.0 | 0.690 |",
                "",
                "## Statistical Testing",
                "",
                "| Comparison | Metric | Test | p-value |",
                "|---|---|---|---:|",
                "| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | 0.018 |",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-doctor",
            "--example-root",
            str(example_root),
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert "TCGA project doctor:" in output
    assert "- Baseline PDF: PASS" in output
    assert "- Code path: PASS" in output
    assert "- Experiment results: PASS" in output
    assert "- LLM live preflight: SKIP" in output
    assert "Overall: PASS" in output


def test_cli_tcga_doctor_summary_records_live_llm_pass(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    results_dir = example_root / "results"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (results_dir / "tcga_results.md").write_text("strict result placeholder", encoding="utf-8")
    summary_path = tmp_path / "tcga-doctor-live-pass.json"

    monkeypatch.setattr(
        cli_module,
        "_validate_results_file",
        lambda *args, **kwargs: {
            "experiment_evidence": {"real_result_evidence": True},
            "experiment_contract": {"status": "complete"},
            "experiment_quality": {"status": "complete"},
            "experiment_provenance": {"status": "complete"},
            "experiment_artifact_consistency": {"status": "complete"},
        },
    )
    monkeypatch.setattr(
        cli_module,
        "_latex_toolchain_status",
        lambda: {"available": True, "preferred_tool": "tectonic.exe", "install_hint": ""},
    )
    monkeypatch.setattr(
        cli_module,
        "_llm_preflight_check",
        lambda *args, **kwargs: {
            "elapsed_seconds": 0.25,
            "response_model": "deepseek-v4-pro",
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-doctor",
            "--example-root",
            str(example_root),
            "--summary",
            str(summary_path),
            "--live-llm",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "- LLM live preflight: PASS" in output
    assert "TCGA doctor summary written to" in output
    assert summary["status"] == "pass"
    assert summary["llm_live_preflight"]["status"] == "pass"
    assert summary["llm_live_preflight"]["elapsed_seconds"] == 0.25
    assert summary["llm_live_preflight"]["usage"]["total_tokens"] == 12
    assert summary["llm"]["provider"] == "deepseek"
    assert "test-key" not in json.dumps(summary)


def test_cli_tcga_doctor_summary_records_live_llm_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    results_dir = example_root / "results"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    (results_dir / "tcga_results.md").write_text("strict result placeholder", encoding="utf-8")
    summary_path = tmp_path / "tcga-doctor-live-fail.json"

    monkeypatch.setattr(
        cli_module,
        "_validate_results_file",
        lambda *args, **kwargs: {
            "experiment_evidence": {"real_result_evidence": True},
            "experiment_contract": {"status": "complete"},
            "experiment_quality": {"status": "complete"},
            "experiment_provenance": {"status": "complete"},
            "experiment_artifact_consistency": {"status": "complete"},
        },
    )
    monkeypatch.setattr(
        cli_module,
        "_latex_toolchain_status",
        lambda: {"available": True, "preferred_tool": "tectonic.exe", "install_hint": ""},
    )

    def fail_preflight(*args, **kwargs):
        raise SystemExit("TCGA doctor LLM preflight failed for deepseek/deepseek-v4-pro: quota blocked.")

    monkeypatch.setattr(cli_module, "_llm_preflight_check", fail_preflight)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-doctor",
            "--example-root",
            str(example_root),
            "--summary",
            str(summary_path),
            "--live-llm",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "TCGA doctor failed" in str(exc)
    else:
        raise AssertionError("Expected TCGA doctor to fail when live LLM preflight fails.")

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "- LLM live preflight: FAIL" in output
    assert "TCGA doctor summary written to" in output
    assert summary["status"] == "fail"
    assert summary["llm_live_preflight"]["status"] == "fail"
    assert summary["llm_live_preflight"]["diagnostics"]["failure_kind"] == "quota"
    assert summary["llm_live_preflight"]["diagnostics"]["provider"] == "deepseek"
    assert any(action["phase"] == "llm_live_preflight" for action in summary["next_actions"])
    assert "test-key" not in json.dumps(summary)


def test_cli_llm_draft_smoke_requires_successful_llm_sections(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    experiment_path = tmp_path / "results.md"
    experiment_path.write_text(
        "| Method | BLCA C-index |\n"
        "|---|---:|\n"
        "| baseline | 0.646 |\n"
        "| ours | 0.671 |\n",
        encoding="utf-8",
    )
    latex_dir = tmp_path / "latex"
    latex_dir.mkdir()
    (latex_dir / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_available"] = bool(llm_client and llm_client.available)

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "partial_llm",
                    "section_writer_llm_attempted_sections": [
                        "abstract",
                        "method",
                        "experiments",
                    ],
                    "section_writer_llm_successes": ["abstract", "method"],
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    output_dir = tmp_path / "out"
    zip_path = tmp_path / "llm-smoke.zip"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli_module, "_llm_preflight_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "llm-draft-smoke",
            "--example-root",
            str(example_root),
            "--experiment-results",
            str(experiment_path),
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--min-llm-sections",
            "2",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "LLM draft smoke passed." in output
    assert "Acceptance report written to" in output
    assert captured["llm_available"]
    assert captured["request"].project_name == cli_module._default_project_name(output_dir)
    assert captured["request"].skip_llm_self_review
    assert summary["project_name"] == cli_module._default_project_name(output_dir)
    assert summary["section_writer_llm_successes"] == ["abstract", "method"]
    assert summary["inputs"]["experiment_results_source"] == "file"
    assert summary["inputs"]["experiment_evidence_kind"] == "real_result_file"
    assert summary["outputs"]["acceptance_report_path"].endswith("ACCEPTANCE_REPORT.md")
    assert "# Paper Agent Acceptance Report" in acceptance_report
    assert zip_path.exists()


def test_cli_llm_draft_smoke_strict_results_fails_before_workflow(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL", "deepseek-v4-pro")
    example_root = tmp_path / "example"
    baseline_dir = example_root / "baseline"
    code_dir = example_root / "code" / "hyper-protosurv"
    baseline_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    (baseline_dir / "baseline.pdf").write_bytes(b"%PDF-1.4\n")
    experiment_path = tmp_path / "tcga_results_template.md"
    experiment_path.write_text(
        cli_module.experiment_results_template(datasets=["BLCA", "BRCA"]),
        encoding="utf-8",
    )

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            raise AssertionError("Workflow should not be constructed after strict result preflight failure.")

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "llm-draft-smoke",
            "--example-root",
            str(example_root),
            "--experiment-results",
            str(experiment_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--strict-results",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected strict LLM smoke run to fail on TODO result template.")
    output = capsys.readouterr().out
    assert "Experiment result contract: invalid" in output


def test_cli_paper_e2e_smoke_runs_explicit_inputs_without_llm(monkeypatch, tmp_path, capsys):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    experiment_path = tmp_path / "results.md"
    latex_dir = tmp_path / "latex"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    latex_dir.mkdir()
    experiment_path.write_text(
        "| Method | BLCA C-index |\n"
        "|---|---:|\n"
        "| ProtoSurv baseline | 0.646 |\n"
        "| Hyper-ProtoSurv ours | 0.671 |\n",
        encoding="utf-8",
    )
    (latex_dir / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_client"] = llm_client

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "reference_resolver_mode": "openalex",
                    "related_work_discovery_mode": "openalex",
                    "related_work_field_query": "whole slide images survival prediction",
                    "related_work_baseline_mentioned_queries": [
                        "Mobadersany | Predicting cancer outcomes from histology"
                    ],
                    "related_work_discovery_error_details": [
                        {
                            "source": "baseline_mentions",
                            "query": "Mobadersany | Predicting cancer outcomes from histology",
                            "sort": "relevance_score:desc",
                            "error": "timeout",
                        }
                    ],
                    "related_work_candidates": [],
                    "related_work_discovery_errors": {"baseline_mentions": "timeout"},
                    "section_writer_llm_successes": [],
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    def fake_write_latex_zip_and_refresh(state, zip_path):
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "w") as archive:
            archive.writestr("main.tex", "\\documentclass{article}")
        state["latex_zip_path"] = zip_path
        return zip_path

    output_dir = tmp_path / "paper-smoke"
    zip_path = tmp_path / "paper-smoke.zip"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli_module, "_write_latex_zip_and_refresh", fake_write_latex_zip_and_refresh)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-smoke",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--target-venue",
            "TPAMI",
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--no-strict-results",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "paper-e2e-smoke passed." in output
    assert "Artifact manifest written to" in output
    assert captured["llm_client"] is None
    assert captured["request"].baseline_pdf_path == str(baseline_pdf)
    assert captured["request"].code_path == str(code_dir)
    assert captured["request"].target_venue == "TPAMI"
    assert (output_dir / "draft.md").is_file()
    assert (output_dir / "ACCEPTANCE_REPORT.md").is_file()
    assert (output_dir / "ARTIFACT_MANIFEST.json").is_file()
    assert zip_path.exists()
    assert summary["smoke_contract"]["schema_version"] == "paper-e2e-smoke/v1"
    assert summary["smoke_contract"]["required_inputs"]["baseline_pdf"] == str(baseline_pdf)
    assert summary["smoke_contract"]["required_inputs"]["code_path"] == str(code_dir)
    assert summary["smoke_contract"]["required_inputs"]["experiment_results"] == str(experiment_path)
    assert summary["smoke_contract"]["required_inputs"]["target_venue"] == "TPAMI"
    assert summary["smoke_contract"]["checks"]["llm_mode"] == "disabled"
    assert summary["smoke_contract"]["outputs"]["artifact_manifest"].endswith("ARTIFACT_MANIFEST.json")
    assert summary["outputs"]["artifact_manifest_path"].endswith("ARTIFACT_MANIFEST.json")
    assert summary["inputs"]["experiment_results_path"] == str(experiment_path)
    assert [action["category"] for action in summary["next_actions"]] == [
        "related_work_online",
        "related_work_retry",
        "related_work_seed_review",
    ]
    assert summary["next_command"].startswith("paper-agent paper-e2e-smoke")
    assert "--online" in summary["next_command"]
    assert "--offline" not in summary["next_command"]
    assert "- Artifact manifest:" in acceptance_report
    assert "## Recommended Next Actions" in acceptance_report
    assert "- Related-work field query: `whole slide images survival prediction`" in acceptance_report
    assert (
        "- Baseline mention query 1: `Mobadersany | Predicting cancer outcomes from histology`"
        in acceptance_report
    )
    assert (
        "- Related-work error detail 1: source=baseline_mentions; query=Mobadersany | Predicting cancer outcomes from histology; sort=relevance_score:desc; error=timeout"
        in acceptance_report
    )
    assert "related_work_online" in acceptance_report
    assert "related_work_seed_review" in acceptance_report
    assert manifest["schema_version"] == "paper-e2e-artifact-manifest/v1"
    assert manifest["project_name"] == cli_module._default_project_name(output_dir)
    assert manifest["smoke_contract_status"] == "pass"
    assert manifest["llm"]["mode"] == "disabled"
    labels = {item["label"]: item for item in manifest["artifacts"]}
    assert labels["markdown_draft"]["exists"] is True
    assert labels["markdown_draft"]["sha256"] == hashlib.sha256(
        (output_dir / "draft.md").read_bytes()
    ).hexdigest()
    assert labels["run_summary"]["exists"] is True
    assert labels["acceptance_report"]["exists"] is True
    assert labels["overleaf_zip"]["exists"] is True
    assert labels["main_tex"]["exists"] is True
    assert labels["latex_project"]["kind"] == "directory"


def test_cli_paper_e2e_smoke_records_successful_llm_preflight(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    experiment_path = tmp_path / "results.md"
    latex_dir = tmp_path / "latex"
    output_dir = tmp_path / "paper-smoke"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    latex_dir.mkdir()
    experiment_path.write_text(
        "| Method | BLCA C-index |\n"
        "|---|---:|\n"
        "| ProtoSurv baseline | 0.646 |\n"
        "| Hyper-ProtoSurv ours | 0.671 |\n",
        encoding="utf-8",
    )
    (latex_dir / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    llm_config = LLMConfig(
        api_key="secret-test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
    )
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_client"] = llm_client

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "llm",
                    "section_writer_llm_attempted_sections": ["abstract", "method"],
                    "section_writer_llm_successes": ["abstract", "method"],
                    "section_writer_llm_call_trace": [
                        {
                            "section": "abstract",
                            "phase": "draft",
                            "status": "success",
                            "model": "deepseek-v4-pro",
                            "usage": {"total_tokens": 21},
                        },
                        {
                            "section": "method",
                            "phase": "draft",
                            "status": "success",
                            "model": "deepseek-v4-pro",
                            "usage": {"total_tokens": 34},
                        },
                    ],
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    def pass_preflight(client, config, *, context):
        return {
            "elapsed_seconds": 0.25,
            "response_model": "deepseek-v4-pro",
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli_module, "load_llm_config", lambda: llm_config)
    monkeypatch.setattr(cli_module, "_llm_preflight_check", pass_preflight)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-smoke",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--target-venue",
            "TPAMI",
            "--output-dir",
            str(output_dir),
            "--zip",
            "",
            "--no-strict-results",
            "--require-llm",
            "--min-llm-sections",
            "2",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    serialized = json.dumps(summary)
    assert "paper-e2e-smoke passed." in output
    assert captured["llm_client"].available is True
    assert summary["inputs"]["llm_mode"] == "required"
    assert summary["llm_preflight_status"] == "pass"
    assert summary["llm_preflight"]["elapsed_seconds"] == 0.25
    assert summary["llm_preflight_total_tokens"] == 12
    assert summary["section_writer_llm_total_tokens"] == 55
    assert summary["smoke_contract"]["checks"]["llm_preflight_status"] == "pass"
    assert "- LLM preflight: pass; elapsed=0.25; total_tokens=12" in report
    assert "| LLM preflight | PASS | status=pass; elapsed=0.25; total_tokens=12 |" in report
    assert "| LLM call trace | PASS | 2/2 calls succeeded; total_tokens=55; required >= 2 |" in report
    assert "secret-test-key" not in serialized
    assert "secret-test-key" not in report


def test_cli_paper_e2e_smoke_generates_results_from_artifacts_before_strict_validation(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    logs_dir = tmp_path / "logs"
    experiment_path = tmp_path / "results" / "tcga_results.md"
    latex_dir = tmp_path / "latex"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    latex_dir.mkdir()
    _write_complete_tcga_artifacts(logs_dir)
    (latex_dir / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_client"] = llm_client

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "section_writer_llm_successes": [],
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    output_dir = tmp_path / "paper-smoke"
    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-smoke",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--target-venue",
            "TPAMI",
            "--output-dir",
            str(output_dir),
            "--zip",
            "",
            "--generate-results-from-artifacts",
            "--artifacts-dir",
            str(logs_dir),
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    result_text = experiment_path.read_text(encoding="utf-8")
    assert "TCGA result file written to" in output
    assert "paper-e2e-smoke passed." in output
    assert captured["llm_client"] is None
    assert "## Result Provenance" in captured["request"].experiment_results
    assert "## Result Provenance" in result_text
    assert summary["smoke_contract"]["checks"]["generated_results_from_artifacts"] is True
    assert summary["smoke_contract"]["checks"]["artifacts_dir"] == str(logs_dir)
    assert summary["smoke_contract"]["checks"]["strict_results_accepted"] is True
    assert summary["inputs"]["experiment_results_path"] == str(experiment_path)
    labels = {item["label"]: item for item in manifest["artifacts"]}
    assert labels["generated_experiment_results"]["path"] == str(experiment_path)
    assert labels["generated_experiment_results"]["exists"] is True


def test_cli_paper_e2e_report_writes_showcase_markdown(monkeypatch, tmp_path, capsys):
    draft_path = tmp_path / "draft.md"
    summary_path = tmp_path / "RUN_SUMMARY.json"
    acceptance_path = tmp_path / "ACCEPTANCE_REPORT.md"
    manifest_path = tmp_path / "ARTIFACT_MANIFEST.json"
    output_path = tmp_path / "SHOWCASE_REPORT.md"
    draft_path.write_text("# Draft", encoding="utf-8")
    acceptance_path.write_text(
        "\n".join(
            [
                "# Paper Agent Acceptance Report",
                "- Overall status: PASS",
                "- Pipeline status: PASS",
                "- Submission evidence status: PASS",
                "| LLM section drafting | PASS | 2/2 sections succeeded |",
                "| LLM call trace | PASS | 2/2 calls succeeded; total_tokens=55 |",
                "| Output artifacts | PASS | markdown=draft.md |",
            ]
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "project_name": "paper-smoke",
                "target_venue": "TPAMI",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "paper-e2e-artifact-manifest/v1",
                "status": "pass",
                "project_name": "paper-smoke",
                "target_venue": "TPAMI",
                "smoke_contract_status": "pass",
                "llm": {
                    "mode": "required",
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                    "endpoint_host": "api.deepseek.com",
                    "preflight_status": "pass",
                    "preflight_total_tokens": 12,
                    "section_call_count": 2,
                    "section_call_successes": 2,
                    "section_total_tokens": 55,
                    "self_review_mode": "disabled",
                },
                "experiment": {
                    "source": "file",
                    "path": "results/tcga_results.md",
                    "evidence_kind": "real_result_file",
                    "result_tables": 1,
                    "contract_status": "complete",
                    "quality_status": "not_configured",
                    "provenance_status": "complete",
                    "artifact_consistency_status": "complete",
                },
                "artifacts": [
                    {
                        "label": "markdown_draft",
                        "kind": "file",
                        "path": str(draft_path),
                        "exists": True,
                        "size_bytes": draft_path.stat().st_size,
                        "sha256": hashlib.sha256(draft_path.read_bytes()).hexdigest(),
                    },
                    {
                        "label": "run_summary",
                        "kind": "file",
                        "path": str(summary_path),
                        "exists": True,
                        "size_bytes": summary_path.stat().st_size,
                        "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    },
                    {
                        "label": "acceptance_report",
                        "kind": "file",
                        "path": str(acceptance_path),
                        "exists": True,
                        "size_bytes": acceptance_path.stat().st_size,
                        "sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-report",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    report = output_path.read_text(encoding="utf-8")
    assert "Paper E2E showcase report written to" in output
    assert "# Paper E2E Showcase Report" in report
    assert "- Project: paper-smoke" in report
    assert "- Provider/model: deepseek / deepseek-v4-pro" in report
    assert "- Preflight: pass; tokens=12" in report
    assert "- Contract: complete" in report
    assert "- Overall status: PASS" in report
    assert "| LLM call trace | PASS | 2/2 calls succeeded; total_tokens=55 |" in report
    assert "| markdown_draft | yes |" in report
    assert hashlib.sha256(draft_path.read_bytes()).hexdigest()[:12] in report


def test_cli_paper_e2e_acceptance_runs_smoke_and_showcase_report(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    experiment_path = tmp_path / "results.md"
    latex_dir = tmp_path / "latex"
    output_dir = tmp_path / "paper-acceptance"
    showcase_path = tmp_path / "paper-acceptance-showcase.md"
    zip_path = tmp_path / "paper-acceptance.zip"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    latex_dir.mkdir()
    experiment_path.write_text(
        "| Method | BLCA C-index |\n"
        "|---|---:|\n"
        "| ProtoSurv baseline | 0.646 |\n"
        "| Hyper-ProtoSurv ours | 0.671 |\n",
        encoding="utf-8",
    )
    (latex_dir / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_client"] = llm_client

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "section_writer_llm_successes": [],
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    def fake_write_latex_zip_and_refresh(state, zip_path_arg):
        zip_path_arg.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path_arg, "w") as archive:
            archive.writestr("main.tex", "\\documentclass{article}")
        state["latex_zip_path"] = zip_path_arg
        return zip_path_arg

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli_module, "_write_latex_zip_and_refresh", fake_write_latex_zip_and_refresh)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-acceptance",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--target-venue",
            "TPAMI",
            "--output-dir",
            str(output_dir),
            "--zip",
            str(zip_path),
            "--showcase-report",
            str(showcase_path),
            "--no-strict-results",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    showcase_report = showcase_path.read_text(encoding="utf-8")
    assert "paper-e2e-smoke passed." in output
    assert "Paper E2E showcase report written to" in output
    assert "paper-e2e-acceptance passed." in output
    assert captured["llm_client"] is None
    assert captured["request"].baseline_pdf_path == str(baseline_pdf)
    assert captured["request"].code_path == str(code_dir)
    assert captured["request"].target_venue == "TPAMI"
    assert (output_dir / "draft.md").is_file()
    assert (output_dir / "ACCEPTANCE_REPORT.md").is_file()
    assert (output_dir / "ARTIFACT_MANIFEST.json").is_file()
    assert showcase_path.is_file()
    assert zip_path.is_file()
    assert summary["smoke_contract"]["schema_version"] == "paper-e2e-smoke/v1"
    assert manifest["smoke_contract_status"] == "pass"
    assert "# Paper E2E Showcase Report" in showcase_report
    assert "- Project: paper-acceptance" in showcase_report
    assert "| markdown_draft | yes |" in showcase_report
    assert "| overleaf_zip | yes |" in showcase_report


def test_cli_paper_e2e_acceptance_writes_showcase_on_strict_block(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    experiment_path = tmp_path / "tcga_results_template.md"
    output_dir = tmp_path / "blocked-acceptance"
    showcase_path = tmp_path / "blocked-showcase.md"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    experiment_path.write_text(
        cli_module.experiment_results_template(datasets=["BLCA", "BRCA"]),
        encoding="utf-8",
    )

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            raise AssertionError("Workflow should not start after strict result preflight failure.")

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-acceptance",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--target-venue",
            "TPAMI",
            "--output-dir",
            str(output_dir),
            "--zip",
            "",
            "--showcase-report",
            str(showcase_path),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "paper-e2e-smoke failed: experiment result validation failed in strict mode" in str(exc)
    else:
        raise AssertionError("Expected paper-e2e-acceptance to keep the strict block non-zero.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    acceptance_report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    showcase_report = showcase_path.read_text(encoding="utf-8")
    assert "Paper E2E showcase report written to" in output
    assert summary["status"] == "blocked"
    assert manifest["status"] == "blocked"
    assert manifest["smoke_contract_status"] == "blocked"
    assert manifest["experiment"]["contract_status"] == "invalid"
    assert manifest["llm"]["mode"] == "not_started"
    labels = {item["label"]: item for item in manifest["artifacts"]}
    assert labels["markdown_draft"]["exists"] is False
    assert labels["run_summary"]["exists"] is True
    assert labels["acceptance_report"]["exists"] is True
    assert "# Paper Agent Blocked Acceptance Report" in acceptance_report
    assert "- Overall status: FAIL" in showcase_report
    assert "| Experiment result contract | FAIL | invalid;" in showcase_report
    assert "## Repair Plan" in showcase_report
    assert "### Blocking Items" in showcase_report
    assert "### Immediate Next Step" in showcase_report
    assert "paper-agent validate-results" in showcase_report
    assert "tcga-artifact-template --output-dir" in showcase_report
    assert "tcga-results-from-artifacts --artifacts-dir" in showcase_report
    assert "paper-e2e-smoke --baseline-pdf" in showcase_report
    assert "| markdown_draft | no |" in showcase_report


def test_cli_research_paper_guide_writes_result_templates_when_artifacts_missing(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code" / "hyper-protosurv"
    output_dir = tmp_path / "research-guide"
    logs_dir = tmp_path / "results" / "logs"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "research-paper-guide",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--target-venue",
            "TPAMI",
            "--artifacts-dir",
            str(logs_dir),
            "--output-dir",
            str(output_dir),
            "--dataset",
            "BLCA",
            "--dataset",
            "BRCA",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "research-paper-guide blocked during result completion" in str(exc)
    else:
        raise AssertionError("Expected guide to stop after writing result templates.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RESEARCH_GUIDE_SUMMARY.json").read_text(encoding="utf-8"))
    result_summary = json.loads((output_dir / "RESULT_GUIDE_SUMMARY.json").read_text(encoding="utf-8"))
    report = (output_dir / "RESEARCH_GUIDE_REPORT.md").read_text(encoding="utf-8")
    assert "TCGA result CSV templates written" in output
    assert (logs_dir / "tcga_main_results.csv").is_file()
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "result_guide_blocked"
    assert summary["outputs"]["research_guide_report"].endswith("RESEARCH_GUIDE_REPORT.md")
    assert summary["outputs"]["result_guide_summary"].endswith("RESULT_GUIDE_SUMMARY.json")
    assert result_summary["pipeline_phase"] == "artifact_template_written"
    assert "# Research Paper Guide Report" in report
    assert "- Status: blocked" in report
    assert "- Result guide status: blocked" in report
    assert summary["next_actions"][0]["source"] == "result_guide"
    assert summary["next_actions"][0]["category"] == "result_artifacts"
    assert "## Next Actions" in report
    assert "Fill every TODO value with real trained-model outputs." in report
    assert "| Paper artifact manifest | no |" in report
    assert not (output_dir / "paper" / "draft.md").exists()


def test_cli_research_paper_guide_generates_results_and_runs_paper_acceptance(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code" / "hyper-protosurv"
    logs_dir = tmp_path / "logs"
    latex_dir = tmp_path / "latex"
    output_dir = tmp_path / "research-guide"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir(parents=True)
    latex_dir.mkdir()
    _write_complete_tcga_artifacts(logs_dir)
    (latex_dir / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_client"] = llm_client

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "section_writer_llm_successes": [],
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [{"title": "Main results"}],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "research-paper-guide",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--target-venue",
            "TPAMI",
            "--artifacts-dir",
            str(logs_dir),
            "--output-dir",
            str(output_dir),
            "--zip",
            "",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RESEARCH_GUIDE_SUMMARY.json").read_text(encoding="utf-8"))
    result_summary = json.loads((output_dir / "RESULT_GUIDE_SUMMARY.json").read_text(encoding="utf-8"))
    report = (output_dir / "RESEARCH_GUIDE_REPORT.md").read_text(encoding="utf-8")
    assert "research-paper-guide completed." in output
    assert "TCGA results guide completed." in output
    assert "paper-e2e-acceptance passed." in output
    assert summary["status"] == "pass"
    assert summary["pipeline_phase"] == "paper_e2e_acceptance_passed"
    assert summary["outputs"]["research_guide_report"].endswith("RESEARCH_GUIDE_REPORT.md")
    assert summary["outputs"]["overleaf_zip"] == ""
    assert result_summary["status"] == "pass"
    assert result_summary["experiment_contract_status"] == "complete"
    assert (output_dir / "tcga_results.md").is_file()
    assert (output_dir / "paper" / "draft.md").is_file()
    assert (output_dir / "SHOWCASE_REPORT.md").is_file()
    assert "# Research Paper Guide Report" in report
    assert "- Status: pass" in report
    assert "- Result guide status: pass" in report
    assert "- Manifest status: pass" in report
    assert "| Paper artifact manifest | yes |" in report
    assert captured["request"].experiment_results.startswith("# Real Experiment Results")


def test_research_paper_guide_report_surfaces_quality_evidence(tmp_path):
    output_dir = tmp_path / "research-guide"
    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True)
    result_summary_path = output_dir / "RESULT_GUIDE_SUMMARY.json"
    paper_summary_path = paper_dir / "RUN_SUMMARY.json"
    paper_manifest_path = paper_dir / "ARTIFACT_MANIFEST.json"
    result_summary_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "pipeline_phase": "strict_result_validated",
                "experiment_contract_status": "complete",
            }
        ),
        encoding="utf-8",
    )
    paper_summary_path.write_text(
        json.dumps(
            {
                "inputs": {"llm_mode": "required", "llm_provider": "deepseek", "llm_model": "deepseek-v4-pro"},
                "section_writer_llm_attempted_sections": ["abstract", "introduction", "method"],
                "section_writer_llm_successes": ["abstract", "method"],
                "section_writer_llm_call_count": 3,
                "section_writer_llm_call_successes": 2,
                "section_writer_llm_total_tokens": 512,
                "llm_preflight_status": "pass",
                "llm_self_review_mode": "llm",
                "llm_self_review_auto_revisions": 1,
                "submission_compile_mode": "compile",
                "submission_compile_status": "passed",
                "submission_compile_tool": "tectonic.exe",
                "experiment_contract_status": "complete",
                "experiment_provenance": {"status": "complete"},
                "experiment_artifact_consistency": {"status": "complete"},
                "related_work_discovery_mode": "openalex",
                "related_work_candidates": 6,
                "related_work_baseline_lineage_candidates": 3,
                "related_work_influential_candidates": 2,
                "related_work_recent_candidates": 1,
                "related_work_field_query": "whole slide images survival prediction",
                "related_work_baseline_mentioned_queries": [
                    "Mobadersany | Predicting cancer outcomes from histology",
                    "Chen | Whole slide images are 2d point clouds",
                ],
                "related_work_candidate_preview": [
                    {
                        "category": "baseline_mentioned",
                        "discovery_path_label": "baseline related-work mention",
                        "title": "Predicting cancer outcomes from histology",
                        "source_query": "Mobadersany | Predicting cancer outcomes from histology",
                    },
                    {
                        "category": "influential",
                        "discovery_path_label": "high-citation field search",
                        "title": "Computational pathology survey",
                        "source_query": "whole slide images survival prediction",
                    },
                ],
                "related_work_discovery_error_count": 1,
                "related_work_discovery_error_sources": ["recent_search"],
                "related_work_discovery_error_details": [
                    {
                        "source": "recent_search",
                        "query": "whole slide images survival prediction",
                        "sort": "publication_date:desc",
                        "error": "timeout",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    paper_manifest_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "smoke_contract_status": "pass",
                "experiment": {"contract_status": "complete"},
                "llm": {
                    "mode": "required",
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                    "preflight_status": "pass",
                    "section_call_count": 3,
                    "section_call_successes": 2,
                    "section_total_tokens": 512,
                    "self_review_mode": "llm",
                },
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "pass",
        "pipeline_phase": "paper_e2e_acceptance_passed",
        "project_name": "quality-demo",
        "target_venue": "TPAMI",
        "inputs": {"experiment_results": str(output_dir / "tcga_results.md")},
        "outputs": {
            "result_guide_summary": str(result_summary_path),
            "paper_run_summary": str(paper_summary_path),
            "paper_artifact_manifest": str(paper_manifest_path),
        },
    }

    quality = cli_module._research_paper_guide_quality_evidence(summary["outputs"])
    report = cli_module._build_research_paper_guide_report(summary)

    assert quality["llm_section_successes"] == ["abstract", "method"]
    assert quality["llm_section_total_tokens"] == 512
    assert quality["latex_compile_status"] == "passed"
    assert quality["experiment_provenance_status"] == "complete"
    assert quality["experiment_artifact_consistency_status"] == "complete"
    assert quality["related_work_discovery_mode"] == "openalex"
    assert quality["related_work_candidates"] == 6
    assert quality["related_work_field_query"] == "whole slide images survival prediction"
    assert quality["related_work_baseline_mentioned_queries"][0].startswith("Mobadersany |")
    assert quality["related_work_candidate_preview"][0]["category"] == "baseline_mentioned"
    assert quality["related_work_candidate_preview"][0]["discovery_path_label"] == "baseline related-work mention"
    assert quality["related_work_discovery_error_details"][0]["source"] == "recent_search"
    assert "- LLM sections: 2/3; successes: abstract, method" in report
    assert "- LLM calls: 2/3; tokens=512" in report
    assert "- LLM self-review: llm; auto_revisions=1" in report
    assert "- LaTeX compile: passed; tool=tectonic.exe; mode=compile" in report
    assert "- Discovery mode: openalex" in report
    assert "- Candidates: 6" in report
    assert "- Baseline-lineage candidates: 3" in report
    assert "- Influential candidates: 2" in report
    assert "- Recent candidates: 1" in report
    assert "- Field query: `whole slide images survival prediction`" in report
    assert "- Baseline mention query 1: `Mobadersany | Predicting cancer outcomes from histology`" in report
    assert (
        "- Candidate preview: [baseline_mentioned: baseline related-work mention] Predicting cancer outcomes from histology <= Mobadersany | Predicting cancer outcomes from histology; [influential: high-citation field search] Computational pathology survey <= whole slide images survival prediction"
        in report
    )
    assert (
        "- Discovery error detail 1: source=recent_search; query=whole slide images survival prediction; sort=publication_date:desc; error=timeout"
        in report
    )
    assert "- Discovery errors: 1; sources=recent_search" in report


def test_cli_research_paper_guide_use_existing_skips_result_guide(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code" / "hyper-protosurv"
    result_path = tmp_path / "provided_results.md"
    latex_dir = tmp_path / "latex"
    output_dir = tmp_path / "research-guide"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir(parents=True)
    latex_dir.mkdir()
    result_path.write_text("# Existing Results\n\nTODO: user-provided draft.", encoding="utf-8")
    (latex_dir / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (latex_dir / "DRAFT_REPORT.md").write_text("# Report", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            captured["llm_client"] = llm_client

        def run(self, request):
            captured["request"] = request
            return {
                "request": request,
                "final_markdown": "# Draft",
                "venue_template": VenueTemplate(venue="TPAMI", template_source="built-in"),
                "bibliography": [],
                "artifacts": {
                    "section_writer_mode": "deterministic",
                    "reference_resolver_mode": "openalex",
                    "related_work_discovery_mode": "openalex",
                    "related_work_candidates": [],
                    "related_work_discovery_errors": {"baseline_mentions": "timeout"},
                    "section_writer_llm_successes": [],
                    "llm_self_review": {"mode": "disabled"},
                    "draft_report_path": str(latex_dir / "DRAFT_REPORT.md"),
                    "experiment_result_tables": [],
                },
                "latex_output_path": latex_dir / "main.tex",
                "latex_project_dir": latex_dir,
                "review_findings": [],
            }

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "research-paper-guide",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--target-venue",
            "TPAMI",
            "--experiment-results",
            str(result_path),
            "--results-mode",
            "use-existing",
            "--output-dir",
            str(output_dir),
            "--zip",
            "",
            "--no-strict-results",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RESEARCH_GUIDE_SUMMARY.json").read_text(encoding="utf-8"))
    report = (output_dir / "RESEARCH_GUIDE_REPORT.md").read_text(encoding="utf-8")
    assert "research-paper-guide completed." in output
    assert "TCGA results guide completed." not in output
    assert summary["status"] == "pass"
    assert summary["inputs"]["results_mode"] == "use-existing"
    assert "--results-mode use-existing" in summary["next_command"]
    assert not (output_dir / "RESULT_GUIDE_SUMMARY.json").exists()
    assert summary["quality_evidence"]["result_guide_status"] == "not_run"
    assert summary["quality_evidence"]["result_guide_phase"] == "use_existing_results"
    assert [action["source"] for action in summary["next_actions"]] == ["paper_e2e"] * 3
    assert [action["category"] for action in summary["next_actions"]] == [
        "related_work_online",
        "related_work_retry",
        "related_work_seed_review",
    ]
    assert "- Result guide status: not_run" in report
    assert "- Result guide phase: use_existing_results" in report
    assert "## Next Actions" in report
    assert "related_work_online" in report
    assert "--online" in report
    assert captured["request"].experiment_results == result_path.read_text(encoding="utf-8")


def test_cli_research_paper_guide_use_existing_requires_experiment_results(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code" / "hyper-protosurv"
    output_dir = tmp_path / "research-guide"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "research-paper-guide",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--target-venue",
            "TPAMI",
            "--results-mode",
            "use-existing",
            "--output-dir",
            str(output_dir),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "research-paper-guide blocked by result mode" in str(exc)
    else:
        raise AssertionError("Expected use-existing mode to require --experiment-results.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RESEARCH_GUIDE_SUMMARY.json").read_text(encoding="utf-8"))
    assert "Research paper guide summary written" in output
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "result_mode_blocked"
    assert summary["inputs"]["results_mode"] == "use-existing"
    assert summary["blocking_items"] == ["--results-mode use-existing requires --experiment-results."]


def test_cli_research_paper_guide_propagates_blocked_paper_next_actions(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code" / "hyper-protosurv"
    experiment_path = tmp_path / "tcga_results_template.md"
    output_dir = tmp_path / "research-guide"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir(parents=True)
    experiment_path.write_text(
        cli_module.experiment_results_template(datasets=["BLCA", "BRCA"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "research-paper-guide",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--target-venue",
            "TPAMI",
            "--experiment-results",
            str(experiment_path),
            "--results-mode",
            "use-existing",
            "--output-dir",
            str(output_dir),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "research-paper-guide blocked during paper acceptance" in str(exc)
    else:
        raise AssertionError("Expected research-paper-guide to surface blocked paper acceptance.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RESEARCH_GUIDE_SUMMARY.json").read_text(encoding="utf-8"))
    paper_summary = json.loads((output_dir / "paper" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    report = (output_dir / "RESEARCH_GUIDE_REPORT.md").read_text(encoding="utf-8")
    assert "Research paper guide summary written" in output
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "paper_e2e_acceptance_blocked"
    assert [action["source"] for action in summary["next_actions"]] == ["paper_e2e"] * 4
    assert summary["blocking_evidence"]["experiment_contract_status"] == "invalid"
    assert summary["blocking_evidence"]["experiment_contract_errors"]
    assert summary["quality_evidence"]["result_guide_status"] == "not_run"
    assert summary["quality_evidence"]["result_contract_status"] == "invalid"
    assert summary["quality_evidence"]["result_provenance_status"] == "invalid"
    assert summary["outputs"]["overleaf_zip"] == str(output_dir / "paper.zip")
    assert f"--zip {output_dir / 'paper.zip'}" in summary["next_command"]
    assert [action["category"] for action in summary["next_actions"]] == [
        "validate_results",
        "result_artifacts",
        "experiment_results",
        "paper_e2e_smoke",
    ]
    assert summary["next_actions"][0]["command"] == paper_summary["next_actions"][0]["command"]
    assert str(output_dir / "paper.zip") in summary["next_actions"][3]["command"]
    assert "## Next Actions" in report
    assert "- Result guide status: not_run" in report
    assert "- Result guide phase: use_existing_results" in report
    assert "- Contract: invalid" in report
    assert "- Provenance: invalid" in report
    assert "- Artifact consistency: not_configured" in report
    assert "- Experiment contract: invalid" in report
    assert "- Contract issue:" in report
    assert f"| Overleaf zip | no | `{output_dir / 'paper.zip'}` |" in report
    assert "paper-agent validate-results" in report
    assert "paper-agent paper-e2e-smoke" in report


def test_cli_research_paper_guide_surfaces_llm_preflight_blocking_evidence(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code" / "hyper-protosurv"
    logs_dir = tmp_path / "logs"
    output_dir = tmp_path / "research-guide"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir(parents=True)
    _write_complete_tcga_artifacts(logs_dir)
    llm_config = LLMConfig(
        api_key="secret-test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
    )

    def fail_preflight(client, config, *, context):
        raise SystemExit(
            f"{context} LLM preflight failed for deepseek/deepseek-v4-pro: "
            "quota blocked for secret-test-key"
        )

    monkeypatch.setattr(cli_module, "load_llm_config", lambda: llm_config)
    monkeypatch.setattr(cli_module, "_llm_preflight_check", fail_preflight)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "research-paper-guide",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--target-venue",
            "TPAMI",
            "--artifacts-dir",
            str(logs_dir),
            "--output-dir",
            str(output_dir),
            "--zip",
            "",
            "--require-llm",
            "--min-llm-sections",
            "4",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "research-paper-guide blocked during paper acceptance" in str(exc)
    else:
        raise AssertionError("Expected research-paper-guide to surface blocked LLM preflight.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RESEARCH_GUIDE_SUMMARY.json").read_text(encoding="utf-8"))
    report = (output_dir / "RESEARCH_GUIDE_REPORT.md").read_text(encoding="utf-8")
    serialized = json.dumps(summary)
    assert "Research paper guide summary written" in output
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "paper_e2e_acceptance_blocked"
    assert summary["blocking_evidence"]["source"] == "paper_e2e"
    assert summary["blocking_evidence"]["pipeline_phase"] == "paper_e2e_smoke_llm_preflight"
    assert summary["blocking_evidence"]["llm_failure_kind"] == "quota"
    assert summary["blocking_evidence"]["llm_provider"] == "deepseek"
    assert summary["blocking_evidence"]["llm_model"] == "deepseek-v4-pro"
    assert "secret-test-key" not in serialized
    assert summary["quality_evidence"]["llm_preflight_status"] == "fail"
    assert "## Blocking Evidence" in report
    assert "- LLM preflight: fail" in report
    assert "LLM failure kind: quota" in report
    assert "LLM provider/model: deepseek / deepseek-v4-pro" in report
    assert "paper-agent llm-doctor" in report
    assert "secret-test-key" not in report


def test_cli_paper_e2e_smoke_strict_results_fails_before_workflow(monkeypatch, tmp_path, capsys):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    experiment_path = tmp_path / "tcga_results_template.md"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    experiment_path.write_text(
        cli_module.experiment_results_template(datasets=["BLCA", "BRCA"]),
        encoding="utf-8",
    )

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            raise AssertionError("Workflow should not start after strict result preflight failure.")

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-smoke",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--target-venue",
            "TPAMI",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "paper-e2e-smoke failed: experiment result validation failed in strict mode" in str(exc)
        assert "RUN_SUMMARY.json" in str(exc)
    else:
        raise AssertionError("Expected paper-e2e-smoke to fail on TODO result template.")
    output = capsys.readouterr().out
    summary_path = tmp_path / "out" / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    report = (tmp_path / "out" / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    assert "Experiment result contract: invalid" in output
    assert "Run summary written to" in output
    assert "Acceptance report written to" in output
    assert "Artifact manifest written to" in output
    assert not (tmp_path / "out" / "draft.md").exists()
    assert (tmp_path / "out" / "ACCEPTANCE_REPORT.md").is_file()
    assert (tmp_path / "out" / "ARTIFACT_MANIFEST.json").is_file()
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "paper_e2e_smoke_preflight"
    assert summary["outputs"]["artifact_manifest_path"].endswith("ARTIFACT_MANIFEST.json")
    assert summary["smoke_contract"]["schema_version"] == "paper-e2e-smoke/v1"
    assert summary["smoke_contract"]["status"] == "blocked"
    assert summary["smoke_contract"]["outputs"]["artifact_manifest"].endswith("ARTIFACT_MANIFEST.json")
    assert summary["smoke_contract"]["checks"]["strict_results"] is True
    assert summary["smoke_contract"]["checks"]["strict_results_accepted"] is False
    assert summary["smoke_contract"]["checks"]["llm_mode"] == "not_started"
    assert manifest["status"] == "blocked"
    assert manifest["smoke_contract_status"] == "blocked"
    assert manifest["experiment"]["contract_status"] == "invalid"
    assert "# Paper Agent Blocked Acceptance Report" in report
    assert "| Experiment result contract | FAIL | invalid;" in report
    assert summary["next_command"].startswith("paper-agent validate-results")
    assert [action["category"] for action in summary["next_actions"]] == [
        "validate_results",
        "result_artifacts",
        "experiment_results",
        "paper_e2e_smoke",
    ]
    assert "tcga-artifact-template --output-dir" in summary["next_actions"][1]["command"]
    assert "tcga-results-from-artifacts --artifacts-dir" in summary["next_actions"][2]["command"]
    assert "paper-e2e-smoke --baseline-pdf" in summary["next_actions"][3]["command"]
    assert any("contract:" in item for item in summary["blocking_items"])


def test_cli_paper_e2e_smoke_writes_summary_on_llm_preflight_failure(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    logs_dir = tmp_path / "logs"
    experiment_path = tmp_path / "results" / "tcga_results.md"
    output_dir = tmp_path / "out"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    _write_complete_tcga_artifacts(logs_dir)
    llm_config = LLMConfig(
        api_key="secret-test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
    )

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            raise AssertionError("Workflow should not start after LLM preflight failure.")

    def fail_preflight(client, config, *, context):
        raise SystemExit(
            f"{context} LLM preflight failed for deepseek/deepseek-v4-pro: "
            "quota blocked for secret-test-key"
        )

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(cli_module, "load_llm_config", lambda: llm_config)
    monkeypatch.setattr(cli_module, "_llm_preflight_check", fail_preflight)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-smoke",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--target-venue",
            "TPAMI",
            "--output-dir",
            str(output_dir),
            "--zip",
            "",
            "--generate-results-from-artifacts",
            "--artifacts-dir",
            str(logs_dir),
            "--require-llm",
            "--min-llm-sections",
            "4",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "paper-e2e-smoke failed: LLM preflight failed" in str(exc)
        assert "RUN_SUMMARY.json" in str(exc)
    else:
        raise AssertionError("Expected paper-e2e-smoke to fail during LLM preflight.")

    output = capsys.readouterr().out
    summary = json.loads((output_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    report = (output_dir / "ACCEPTANCE_REPORT.md").read_text(encoding="utf-8")
    serialized = json.dumps(summary)
    assert "Run summary written to" in output
    assert "Acceptance report written to" in output
    assert "Artifact manifest written to" in output
    assert not (output_dir / "draft.md").exists()
    assert (output_dir / "ARTIFACT_MANIFEST.json").is_file()
    assert summary["status"] == "blocked"
    assert summary["pipeline_phase"] == "paper_e2e_smoke_llm_preflight"
    assert summary["outputs"]["artifact_manifest_path"].endswith("ARTIFACT_MANIFEST.json")
    assert summary["smoke_contract"]["checks"]["strict_results_accepted"] is True
    assert summary["smoke_contract"]["checks"]["llm_mode"] == "failed_preflight"
    assert summary["smoke_contract"]["checks"]["generated_results_from_artifacts"] is True
    assert manifest["status"] == "blocked"
    assert manifest["llm"]["mode"] == "failed_preflight"
    assert manifest["llm"]["provider"] == "deepseek"
    assert manifest["llm"]["model"] == "deepseek-v4-pro"
    assert manifest["experiment"]["contract_status"] == "complete"
    assert summary["llm_diagnostics"]["failure_kind"] == "quota"
    assert summary["llm_diagnostics"]["provider"] == "deepseek"
    assert "secret-test-key" not in serialized
    assert "[redacted-api-key]" in summary["llm_diagnostics"]["raw_error"]
    assert [action["category"] for action in summary["next_actions"]] == ["llm", "paper_e2e_smoke"]
    assert "paper-agent llm-doctor" in summary["next_actions"][0]["command"]
    assert "paper-e2e-smoke --baseline-pdf" in summary["next_actions"][1]["command"]
    assert "# Paper Agent Blocked Acceptance Report" in report
    assert "Failure kind: quota" in report
    assert "secret-test-key" not in report


def test_cli_paper_e2e_smoke_can_write_artifact_templates_on_strict_failure(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline_pdf = tmp_path / "baseline.pdf"
    code_dir = tmp_path / "code"
    experiment_path = tmp_path / "tcga_results_template.md"
    template_dir = tmp_path / "repair-logs"
    baseline_pdf.write_bytes(b"%PDF-1.4\n")
    code_dir.mkdir()
    experiment_path.write_text(
        cli_module.experiment_results_template(datasets=["BLCA", "BRCA"]),
        encoding="utf-8",
    )

    class FakeWorkflow:
        def __init__(self, llm_client=None):
            raise AssertionError("Workflow should not start after strict result preflight failure.")

    monkeypatch.setattr(cli_module, "PaperWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "paper-e2e-smoke",
            "--baseline-pdf",
            str(baseline_pdf),
            "--code-path",
            str(code_dir),
            "--experiment-results",
            str(experiment_path),
            "--target-venue",
            "TPAMI",
            "--output-dir",
            str(tmp_path / "out"),
            "--write-artifact-template",
            "--artifact-template-dir",
            str(template_dir),
            "--artifact-template-dataset",
            "BLCA",
            "--artifact-template-dataset",
            "BRCA",
        ],
    )

    try:
        cli_module.main()
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("Expected paper-e2e-smoke to fail on TODO result template.")

    output = capsys.readouterr().out
    summary = json.loads((tmp_path / "out" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    assert "Artifact templates written to" in output
    assert (template_dir / "tcga_main_results.csv").is_file()
    assert (template_dir / "tcga_ablation.csv").is_file()
    assert (template_dir / "tcga_sensitivity.csv").is_file()
    assert (template_dir / "tcga_stats.csv").is_file()
    assert (template_dir / "EXPORT_CONTRACT.md").is_file()
    assert (template_dir / "ARTIFACT_SCHEMA.json").is_file()
    assert summary["artifact_template"]["status"] == "written"
    assert summary["artifact_template"]["output_dir"] == str(template_dir)
    assert summary["artifact_template"]["datasets"] == ["BLCA", "BRCA"]
    assert summary["artifact_template"]["contains_todo"] is True
    assert str(template_dir) in summary["next_actions"][1]["command"]
    assert "--write-artifact-template" in summary["next_actions"][3]["command"]
    labels = {item["label"]: item for item in manifest["artifacts"]}
    assert labels["artifact_template_tcga_main_results"]["exists"] is True
    assert labels["artifact_template_tcga_ablation"]["exists"] is True
    assert labels["artifact_template_tcga_sensitivity"]["exists"] is True
    assert labels["artifact_template_tcga_stats"]["exists"] is True
    assert labels["artifact_template_export_contract"]["exists"] is True
    assert labels["artifact_template_artifact_schema"]["exists"] is True


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
    monkeypatch.setenv("PAPER_AGENT_DISABLE_LLM_SELF_REWRITE", "1")
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
        "sections": DraftSections(method="The method describes its computational setup."),
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
                "auto_revisions": [
                    {
                        "section": "introduction",
                        "removed_text": "The method improves every cohort without uncertainty evidence.",
                    }
                ],
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
    assert "Auto revisions: 1 unsupported claim edit(s) applied." in report
    assert "introduction: The method improves every cohort without uncertainty evidence." in report
    assert "experiments: The method obtains 0.999 on XYZ." in report
    assert "Evidence needed: Add the missing experiment result." in report


def test_draft_report_submission_reminder_asks_for_real_performance_tables(tmp_path):
    state = {
        "request": PaperRequest(project_name="tcga-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {},
        "bibliography": [],
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "Add real trained-model performance tables" in report
    assert "synthetic or mock" not in report


def test_submission_checklist_flags_tcga_cohort_summary_as_data_only(tmp_path):
    state = {
        "request": PaperRequest(project_name="tcga-checklist-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "experiment_results_source": "tcga_cohort_csv",
            "experiment_results_path": "D:/code/agent/example/code/hyper-protosurv/dataset_csv",
        },
    }

    DraftReportAgent().run(state)

    checklist = (tmp_path / "SUBMISSION_CHECKLIST.md").read_text(encoding="utf-8")
    assert "Experiment evidence kind: data_only" in checklist
    assert "data/cohort metadata only" in checklist
    assert "add trained-model performance tables" in checklist


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


def test_reviewer_links_ablation_evidence_to_innovation():
    state = {
        "experiments": ExperimentSummary(
            ablation_evidence=[
                AblationEvidence(
                    table_caption="Ablation Results",
                    dataset="Average",
                    metric="C-INDEX",
                    reference="Full Hyper-ProtoSurv",
                    variant="w/o bidirectional hyperedge update",
                    reference_value=0.690,
                    variant_value=0.678,
                    signed_drop=0.012,
                    supports=["bidirectional hyperedge updates"],
                ),
                AblationEvidence(
                    table_caption="Ablation Results",
                    dataset="Average",
                    metric="C-INDEX",
                    reference="Full Hyper-ProtoSurv",
                    variant="w/o L_rec",
                    reference_value=0.690,
                    variant_value=0.672,
                    signed_drop=0.018,
                    supports=["reconstruction regularization"],
                )
            ]
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Bidirectional hyperedge updates",
                motivation="Static hyperedges miss reciprocal dependencies.",
                technical_idea="Use bidirectional hyperedge updates for WSI survival modeling.",
                evidence=["Repository contains a bidirectional update block."],
            ),
            InnovationPoint(
                name="Innovation 2: Survival reconstruction objective",
                motivation="The loss should preserve survival supervision and feature recovery.",
                technical_idea="Combine a Cox survival term with reconstruction regularization.",
                evidence=["Repository exposes L_surv and L_rec."],
            )
        ],
        "sections": DraftSections(
            method=(
                "### Bidirectional hyperedge updates\n"
                "The model updates hyperedges in both directions.\n"
                "### Survival reconstruction objective\n"
                "The objective combines survival prediction and reconstruction regularization."
            )
        ),
        "artifacts": {},
    }

    reviewed = ReviewerAgent().run(state)

    traceability = reviewed["artifacts"]["innovation_traceability"]
    assert traceability[0]["mentioned_in_method"]
    assert traceability[0]["ablation_evidence_count"] == 1
    assert "signed drop +0.012" in traceability[0]["ablation_evidence_preview"][0]
    assert traceability[1]["ablation_evidence_count"] == 1
    assert "w/o L_rec" in traceability[1]["ablation_evidence_preview"][0]


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
                    "ablation_evidence_preview": [
                        "w/o adaptive prototype calibration on Average C-INDEX: 0.690 -> 0.674 (signed drop +0.016)"
                    ],
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
    assert "signed drop +0.016" in report


def test_draft_report_includes_code_baseline_comparison(tmp_path):
    state = {
        "request": PaperRequest(project_name="comparison-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "code_baseline_comparison": {
                "mode": "compared",
                "overlapping_terms": ["prototype learning", "survival prediction"],
                "code_only_terms": ["hypergraph modeling"],
                "likely_method_shifts": [
                    {
                        "technique": "hypergraph modeling",
                        "rationale": "Repository evidence supports this as a proposed-method component.",
                        "evidence": [
                            "models/model.py:14 (BHE/HCoN module) self.hcon = HCoN(...)"
                        ],
                    }
                ],
                "innovation_seeds": [
                    "Introduce hypergraph structure modeling for higher-order tissue and prototype relations."
                ],
            }
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Code-Baseline Comparison" in report
    assert "Shared technical context: prototype learning, survival prediction" in report
    assert "Code-side innovation candidates: hypergraph modeling" in report
    assert "Introduce hypergraph structure modeling" in report


def test_submission_package_validator_accepts_project_zip(tmp_path):
    project_dir = tmp_path / "latex"
    project_dir.mkdir()
    main_tex = project_dir / "main.tex"
    main_tex.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\title{Demo}",
                r"\begin{document}",
                r"\begin{abstract}A concise abstract.\end{abstract}",
                r"Prior work \cite{paper}.",
                r"\bibliography{references}",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    (project_dir / "references.bib").write_text(
        "@article{paper,\n  title={Paper},\n  author={Ada Lovelace},\n  year={2024}\n}\n",
        encoding="utf-8",
    )
    zip_path = zip_latex_project(project_dir, tmp_path / "paper.zip")
    state = {
        "latex_project_dir": project_dir,
        "latex_output_path": main_tex,
        "latex_zip_path": zip_path,
        "artifacts": {},
    }

    SubmissionPackageValidatorAgent().run(state)

    package = state["artifacts"]["submission_package"]
    assert package["status"] != "invalid"
    assert not package["errors"]
    assert package["checks"]["citation_keys"] == ["paper"]
    assert package["checks"]["bib_keys"] == ["paper"]
    assert package["checks"]["zip"]["present"]
    assert package["checks"]["zip"]["contains_main_tex"]


def test_submission_package_validator_flags_missing_graphic(tmp_path):
    project_dir = tmp_path / "latex"
    project_dir.mkdir()
    main_tex = project_dir / "main.tex"
    main_tex.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\title{Demo}",
                r"\begin{document}",
                r"\begin{abstract}A concise abstract.\end{abstract}",
                r"\includegraphics{figures/missing-figure}",
                r"\bibliography{references}",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    (project_dir / "references.bib").write_text("", encoding="utf-8")
    state = {
        "latex_project_dir": project_dir,
        "latex_output_path": main_tex,
        "artifacts": {},
    }

    SubmissionPackageValidatorAgent().run(state)

    package = state["artifacts"]["submission_package"]
    assert package["status"] == "invalid"
    assert any("Missing graphics" in error for error in package["errors"])
    assert package["checks"]["missing_graphics"] == ["figures/missing-figure"]


def test_submission_package_validator_uses_tectonic_when_enabled(tmp_path, monkeypatch):
    project_dir = tmp_path / "latex"
    project_dir.mkdir()
    main_tex = project_dir / "main.tex"
    main_tex.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\title{Demo}",
                r"\begin{document}",
                r"\begin{abstract}A concise abstract.\end{abstract}",
                r"\bibliography{references}",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    commands = []

    def fake_find(self, name):
        return "C:/tools/tectonic.exe" if name == "tectonic" else ""

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("PAPER_AGENT_RUN_LATEX_COMPILE", "1")
    monkeypatch.setattr(SubmissionPackageValidatorAgent, "_find_executable", fake_find)
    monkeypatch.setattr(
        "paper_agent.agents.submission_package_validator.subprocess.run",
        fake_run,
    )

    warnings = []
    result = SubmissionPackageValidatorAgent()._compile_check(project_dir, main_tex, warnings)

    assert result["status"] == "passed"
    assert result["tool"] == "tectonic.exe"
    assert commands[0][0] == [
        "C:/tools/tectonic.exe",
        "--keep-logs",
        "--keep-intermediates",
        "main.tex",
    ]
    assert commands[0][1]["cwd"] == project_dir
    assert commands[0][1]["encoding"] == "utf-8"
    assert commands[0][1]["errors"] == "replace"
    assert commands[0][1]["timeout"] == SubmissionPackageValidatorAgent.COMPILE_TIMEOUT_SECONDS
    assert warnings == []


def test_submission_package_validator_reports_latex_install_hint(monkeypatch, tmp_path):
    project_dir = tmp_path / "latex"
    project_dir.mkdir()
    main_tex = project_dir / "main.tex"
    main_tex.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}",
        encoding="utf-8",
    )
    monkeypatch.setattr(SubmissionPackageValidatorAgent, "_find_executable", lambda self, name: "")

    warnings = []
    result = SubmissionPackageValidatorAgent()._compile_check(project_dir, main_tex, warnings)

    assert result["status"] == "tool_unavailable"
    assert result["install_hint"] == "conda install -n agent -c conda-forge tectonic"
    assert warnings == ["No local LaTeX compiler was found; static package checks were run only."]


def test_cli_latex_doctor_reports_missing_toolchain(monkeypatch, capsys):
    monkeypatch.setattr(SubmissionPackageValidatorAgent, "_find_executable", lambda self, name: "")
    monkeypatch.setattr("sys.argv", ["paper-agent", "latex-doctor"])

    cli_module.main()

    output = capsys.readouterr().out
    assert "LaTeX toolchain:" in output
    assert "- tectonic: not found" in output
    assert "Install hint: conda install -n agent -c conda-forge tectonic" in output


def test_draft_report_includes_submission_package_validation(tmp_path):
    state = {
        "request": PaperRequest(project_name="package-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "submission_package": {
                "status": "needs_attention",
                "errors": [],
                "warnings": ["No local LaTeX compiler was found; static package checks were run only."],
                "checks": {
                    "zip": {"present": True, "entries": 4},
                    "compile": {"status": "tool_unavailable", "tool": ""},
                },
            }
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Submission Package" in report
    assert "- Status: needs_attention" in report
    assert "- Zip: present; entries: 4" in report
    assert "static package checks" in report
    checklist = (tmp_path / "SUBMISSION_CHECKLIST.md").read_text(encoding="utf-8")
    assert "## Quick Status" in checklist
    assert "- Package: needs_attention" in checklist
    assert "static package checks" in checklist
    assert "Upload the generated zip file to Overleaf" in checklist
    assert state["artifacts"]["submission_checklist_path"].endswith("SUBMISSION_CHECKLIST.md")


def test_cli_zip_refreshes_submission_package_and_readiness(tmp_path):
    project_dir = tmp_path / "latex"
    project_dir.mkdir()
    main_tex = project_dir / "main.tex"
    main_tex.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\title{Demo}",
                r"\begin{document}",
                r"\begin{abstract}A concise abstract.\end{abstract}",
                r"Prior work \cite{paper}.",
                r"\bibliography{references}",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    (project_dir / "references.bib").write_text(
        "@article{paper,\n  title={Paper},\n  author={Ada Lovelace},\n  year={2024}\n}\n",
        encoding="utf-8",
    )
    state = {
        "request": PaperRequest(project_name="zip-refresh-demo", target_venue="TPAMI"),
        "latex_project_dir": project_dir,
        "latex_output_path": main_tex,
        "venue_template": VenueTemplate(venue="TPAMI"),
        "bibliography": [CitationEntry(key="paper", title="Paper", authors=["Ada"], year="2024")],
        "artifacts": {},
    }

    zip_path = cli_module._write_latex_zip_and_refresh(state, tmp_path / "paper.zip")

    assert zip_path.exists()
    assert state["artifacts"]["submission_package"]["checks"]["zip"]["present"]
    assert state["artifacts"]["submission_readiness"]["scores"]["venue_package"] >= 90
    report = (project_dir / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Submission Package" in report
    assert "- Zip: present" in report
    checklist = (project_dir / "SUBMISSION_CHECKLIST.md").read_text(encoding="utf-8")
    assert "## Quick Status" in checklist
    assert "- Zip entries:" in checklist
    assert "missing helper notes: SUBMISSION_CHECKLIST.md" not in checklist
    with ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert "DRAFT_REPORT.md" in names
        assert "SUBMISSION_CHECKLIST.md" in names


def test_presentation_planner_creates_evidence_bound_figure_and_table_plan():
    state = {
        "request": PaperRequest(
            project_name="presentation-demo",
            target_venue="TPAMI",
            experiment_results=(
                "## Main Results\n"
                "Metric: C-index.\n\n"
                "| Method | BLCA C-index |\n"
                "|---|---:|\n"
                "| ProtoSurv baseline | 0.646 |\n"
                "| Hyper-ProtoSurv ours | 0.671 |\n"
            ),
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype hypergraph",
                motivation="Prototype geometry should be explicit.",
                technical_idea="Construct adaptive prototype geometry with optimal transport.",
                evidence=[
                    "data_preparation/hypergraph.py:20 (OT/Wasserstein hypergraph construction)"
                ],
            )
        ],
        "experiments": ExperimentSummary(
            datasets=["BLCA"],
            metrics=["C-INDEX"],
            result_tables=[
                ExperimentTableSummary(
                    caption="Main Results",
                    method="Hyper-ProtoSurv",
                    baseline="ProtoSurv",
                    comparisons=[
                        ExperimentComparison(
                            dataset="BLCA",
                            metric="C-INDEX",
                            method="Hyper-ProtoSurv",
                            baseline="ProtoSurv",
                            method_value=0.671,
                            baseline_value=0.646,
                            signed_improvement=0.025,
                            improved=True,
                        )
                    ],
                )
            ],
            ablation_evidence=[
                AblationEvidence(
                    variant="w/o OT-driven adaptive hyperedges",
                    reference="Full",
                    dataset="Average",
                    metric="C-INDEX",
                    reference_value=0.690,
                    variant_value=0.674,
                    signed_drop=0.016,
                )
            ],
            sensitivity_evidence=[
                SensitivityEvidence(
                    parameter="lambda_rec",
                    dataset="Average",
                    metric="C-INDEX",
                    best_parameter_value="1.0",
                    best_metric_value=0.690,
                    worst_metric_value=0.681,
                    tested_values=["0.5", "1.0"],
                    metric_values=[0.687, 0.690],
                )
            ],
            statistical_tests=[
                StatisticalTestEvidence(
                    comparison="Hyper-ProtoSurv vs ProtoSurv",
                    metric="C-INDEX",
                    test="Wilcoxon",
                    p_value=0.018,
                    p_value_text="p=0.018",
                    significant=True,
                )
            ],
        ),
        "artifacts": {
            "code_baseline_comparison": {
                "code_only_terms": ["optimal transport geometry", "hypergraph modeling"]
            }
        },
    }

    PresentationPlannerAgent().run(state)

    plan = state["artifacts"]["presentation_plan"]
    labels = {item["label"] for item in plan["figures"]}
    assert "fig:method-overview" in labels
    assert "fig:prototype-hypergraph" in labels
    assert "fig:main-results" in labels
    assert "fig:ablation-summary" in labels
    assert "fig:sensitivity-summary" in labels
    table_labels = {item["label"] for item in plan["tables"]}
    assert any(table["label"].startswith("tab:main-results") for table in plan["tables"])
    assert "tab:sensitivity-summary" in table_labels
    assert "tab:statistical-tests" in table_labels
    assert plan["open_items"]


def test_latex_composer_writes_figure_table_plan(tmp_path):
    state = {
        "request": PaperRequest(
            project_name="figure-table-plan-demo",
            target_venue="TPAMI",
            experiment_results=(
                "## Main Results\n\n"
                "| Method | BLCA C-index |\n"
                "|---|---:|\n"
                "| baseline | 0.646 |\n"
                "| ours | 0.671 |\n"
            ),
        ),
        "venue_template": VenueTemplate(venue="TPAMI", family="ieee_journal"),
        "outline": PaperOutline(title_candidates=["Figure Table Plan Demo"]),
        "sections": DraftSections(
            abstract="Abstract.",
            introduction="Introduction.",
            related_work="Related work.",
            method="### Method Overview\nMethod.",
            experiments="### Main Results\nResults.",
            conclusion="Conclusion.",
        ),
        "bibliography": [],
        "artifacts": {
            "presentation_plan": {
                "figures": [
                    {
                        "label": "fig:method-overview",
                        "title": "Method Overview",
                        "section": "Method",
                        "asset_path": "figures/method_overview.pdf",
                        "caption": "Overview of the proposed method.",
                        "evidence": ["Repository evidence."],
                        "status": "planned",
                    }
                ],
                "tables": [
                    {
                        "label": "tab:main-results",
                        "caption": "Main result table.",
                        "section": "Experiments",
                        "columns": 2,
                        "rows": 2,
                        "status": "planned",
                    }
                ],
                "open_items": ["Create the method overview figure."],
            }
        },
    }

    LatexComposerAgent().run(state)

    plan_path = state["latex_project_dir"] / "FIGURE_TABLE_PLAN.md"
    plan = plan_path.read_text(encoding="utf-8")
    assert plan_path.exists()
    assert "`fig:method-overview`" in plan
    assert "Overview of the proposed method." in plan
    assert "Create the method overview figure." in plan
    assert state["artifacts"]["presentation_plan_path"] == str(plan_path)
    assert state["artifacts"]["latex_tables"][0]["label"].startswith("tab:main-results")
    assert "## Figure and Table Plan" in state["final_markdown"]


def test_latex_composer_converts_nested_markdown_headings():
    composer = LatexComposerAgent()

    latex = composer._latex_escape("### Main Module\nText.\n\n#### Inner Block\nDetails.")

    assert r"\subsection{Main Module}" in latex
    assert r"\subsubsection{Inner Block}" in latex
    assert r"\#\#\#\#" not in latex


def test_latex_composer_drops_missing_local_template_packages(tmp_path):
    sample = tmp_path / "sample.tex"
    sample.write_text(
        "\n".join(
            [
                r"\documentclass{IEEEtran}",
                r"\usepackage{officialstyle}",
                r"\usepackage{amsmath,missinglocal}",
                r"\begin{document}",
                r"Template body",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    state = {"artifacts": {}}
    values = {
        "title": "Demo",
        "abstract": "Abstract.",
        "introduction": "Intro.",
        "related_work": "Related.",
        "method": "Method.",
        "experiments": "Experiments.",
        "conclusion": "Conclusion.",
    }

    rendered = LatexComposerAgent()._render_from_sample_main(sample, values, tmp_path, state)

    assert r"\usepackage{officialstyle}" not in rendered
    assert "missinglocal" not in rendered
    assert r"\usepackage{amsmath}" in rendered
    assert state["artifacts"]["dropped_missing_template_packages"] == [
        "officialstyle",
        "missinglocal",
    ]


def test_latex_composer_escapes_bibtex_special_characters():
    escaped = LatexComposerAgent()._bibtex_escape("category=baseline_mentioned & 50% #1")

    assert escaped == r"category=baseline\_mentioned \& 50\% \#1"


def test_latex_composer_rewrites_unicode_lambda_for_compile():
    latex = LatexComposerAgent()._latex_escape("Sensitivity uses λ_rec = 1.0 and λ = 0.5.")

    assert "λ" not in latex
    assert r"\(\lambda_{\mathrm{rec}}\)" in latex
    assert r"\(\lambda\) = 0.5" in latex


def test_latex_composer_normalizes_windows_paths_without_breaking_math_commands():
    latex = LatexComposerAgent()._latex_escape(
        r"Evidence: data_preparation\hypergraph_construction_wb.py uses λ_rec."
    )

    assert "data\\_preparation/hypergraph\\_construction\\_wb.py" in latex
    assert r"\(\lambda_{\mathrm{rec}}\)" in latex
    assert r"\hypergraph" not in latex


def test_latex_composer_generates_result_figures_and_inserts_existing_assets(tmp_path):
    state = {
        "request": PaperRequest(
            project_name="generated-figures-demo",
            target_venue="TPAMI",
            experiment_results=(
                "## Main Results\n\n"
                "| Method | BLCA C-index | BRCA C-index |\n"
                "|---|---:|---:|\n"
                "| ProtoSurv baseline | 0.646 | 0.669 |\n"
                "| Hyper-ProtoSurv ours | 0.671 | 0.691 |\n"
                "\n"
                "## Ablation Results\n\n"
                "| Variant | Average C-index |\n"
                "|---|---:|\n"
                "| Full Hyper-ProtoSurv | 0.690 |\n"
                "| w/o L_rec | 0.672 |\n"
            ),
        ),
        "venue_template": VenueTemplate(venue="TPAMI", family="ieee_journal"),
        "outline": PaperOutline(title_candidates=["Generated Figures Demo"]),
        "sections": DraftSections(
            abstract="Abstract.",
            introduction="Introduction.",
            related_work="Related work.",
            method="### Method Overview\nMethod.",
            experiments="### Main Results\nResults.",
            conclusion="Conclusion.",
        ),
        "bibliography": [],
        "experiments": ExperimentSummary(
            datasets=["BLCA", "BRCA"],
            metrics=["C-INDEX"],
            result_tables=[
                ExperimentTableSummary(
                    caption="Main Results",
                    method="Hyper-ProtoSurv ours",
                    baseline="ProtoSurv baseline",
                    comparisons=[
                        ExperimentComparison(
                            dataset="BLCA",
                            metric="C-INDEX",
                            method="Hyper-ProtoSurv ours",
                            baseline="ProtoSurv baseline",
                            method_value=0.671,
                            baseline_value=0.646,
                            signed_improvement=0.025,
                            improved=True,
                        ),
                        ExperimentComparison(
                            dataset="BRCA",
                            metric="C-INDEX",
                            method="Hyper-ProtoSurv ours",
                            baseline="ProtoSurv baseline",
                            method_value=0.691,
                            baseline_value=0.669,
                            signed_improvement=0.022,
                            improved=True,
                        ),
                    ],
                )
            ],
            ablation_evidence=[
                AblationEvidence(
                    variant="w/o L_rec",
                    reference="Full Hyper-ProtoSurv",
                    dataset="Average",
                    metric="C-INDEX",
                    reference_value=0.690,
                    variant_value=0.672,
                    signed_drop=0.018,
                )
            ],
            sensitivity_evidence=[
                SensitivityEvidence(
                    parameter="lambda_rec",
                    dataset="Average",
                    metric="C-INDEX",
                    best_parameter_value="1.0",
                    best_metric_value=0.690,
                    worst_metric_value=0.687,
                    tested_values=["0.5", "1.0"],
                    metric_values=[0.687, 0.690],
                )
            ],
        ),
        "artifacts": {
            "presentation_plan": {
                "figures": [
                    {
                        "label": "fig:method-overview",
                        "title": "Method Overview",
                        "section": "Method",
                        "asset_path": "figures/method_overview.pdf",
                        "caption": "Overview of the proposed method.",
                        "evidence": [],
                        "status": "planned",
                    },
                    {
                        "label": "fig:main-results",
                        "title": "Main Result Summary",
                        "section": "Experiments",
                        "asset_path": "figures/main_results.pdf",
                        "caption": "Summary visualization of the main results.",
                        "evidence": [],
                        "status": "planned",
                    },
                    {
                        "label": "fig:ablation-summary",
                        "title": "Ablation Summary",
                        "section": "Experiments",
                        "asset_path": "figures/ablation_summary.pdf",
                        "caption": "Ablation summary.",
                        "evidence": [],
                        "status": "planned",
                    },
                    {
                        "label": "fig:sensitivity-summary",
                        "title": "Sensitivity Summary",
                        "section": "Experiments",
                        "asset_path": "figures/sensitivity_summary.pdf",
                        "caption": "Sensitivity summary.",
                        "evidence": [],
                        "status": "planned",
                    },
                ],
                "tables": [],
                "open_items": [
                    "Create or attach the planned figure asset `figures/method_overview.pdf` for `fig:method-overview`.",
                    "Create or attach the planned figure asset `figures/main_results.pdf` for `fig:main-results`.",
                    "Create or attach the planned figure asset `figures/ablation_summary.pdf` for `fig:ablation-summary`.",
                    "Create or attach the planned figure asset `figures/sensitivity_summary.pdf` for `fig:sensitivity-summary`.",
                ],
            }
        },
    }

    LatexComposerAgent().run(state)

    main_pdf = state["latex_project_dir"] / "figures" / "main_results.pdf"
    ablation_pdf = state["latex_project_dir"] / "figures" / "ablation_summary.pdf"
    sensitivity_pdf = state["latex_project_dir"] / "figures" / "sensitivity_summary.pdf"
    tex = state["latex_output_path"].read_text(encoding="utf-8")
    plan = (state["latex_project_dir"] / "FIGURE_TABLE_PLAN.md").read_text(encoding="utf-8")
    assert main_pdf.exists()
    assert main_pdf.read_bytes().startswith(b"%PDF-1.4")
    assert ablation_pdf.exists()
    assert sensitivity_pdf.exists()
    assert r"\includegraphics[width=\columnwidth]{figures/main_results.pdf}" in tex
    assert r"\includegraphics[width=\columnwidth]{figures/ablation_summary.pdf}" in tex
    assert r"\includegraphics[width=\columnwidth]{figures/sensitivity_summary.pdf}" in tex
    assert "figures/method_overview.pdf" not in tex
    assert state["artifacts"]["generated_figure_count"] == 3
    assert {item["label"] for item in state["artifacts"]["generated_figures"]} == {
        "fig:main-results",
        "fig:ablation-summary",
        "fig:sensitivity-summary",
    }
    assert "Status: generated" in plan
    assert state["artifacts"]["presentation_plan"]["open_items"] == [
        "Create or attach the planned figure asset `figures/method_overview.pdf` for `fig:method-overview`."
    ]

    SubmissionPackageValidatorAgent().run(state)
    assert not state["artifacts"]["submission_package"]["errors"]
    assert state["artifacts"]["submission_package"]["checks"]["missing_graphics"] == []


def test_latex_composer_generates_method_figures_from_code_evidence(tmp_path):
    state = {
        "request": PaperRequest(
            project_name="generated-method-figures-demo",
            target_venue="TPAMI",
        ),
        "venue_template": VenueTemplate(venue="TPAMI", family="ieee_journal"),
        "outline": PaperOutline(title_candidates=["Generated Method Figures Demo"]),
        "sections": DraftSections(
            abstract="Abstract.",
            introduction="Introduction.",
            related_work="Related work.",
            method="### Method Overview\nMethod.\n\n### Prototype Hypergraph\nConstruction.",
            experiments="Experiments.",
            conclusion="Conclusion.",
        ),
        "bibliography": [],
        "code": CodeSummary(
            likely_method_files=["data_preparation/hypergraph_construction_wb.py"],
            implementation_evidence=[
                "data_preparation/hypergraph_construction_wb.py:146 "
                "(OT/Wasserstein hypergraph construction) X_bar = ot.lp.free_support_barycenter(",
                "models/HCoN/model.py:55 (BHE/HCoN module) self.hcon = HCoN(...)",
            ],
            method_claims=["Construct adaptive hyperedges and exchange node-hyperedge context."],
        ),
        "innovations": [
            InnovationPoint(
                name="Innovation 1: Adaptive prototype geometry",
                motivation="Prototype geometry should be explicit.",
                technical_idea="Construct adaptive prototype geometry with optimal transport.",
                evidence=["Wasserstein hypergraph construction evidence."],
            ),
            InnovationPoint(
                name="Innovation 2: Bidirectional hyperedge convolution",
                motivation="Context exchange should be explicit.",
                technical_idea="Use bidirectional hyperedge convolution to exchange node and hyperedge context.",
                evidence=["HCoN implementation evidence."],
            ),
        ],
        "artifacts": {
            "code_baseline_comparison": {
                "code_only_terms": [
                    "optimal transport geometry",
                    "hypergraph modeling",
                    "bidirectional hyperedge convolution",
                ],
                "innovation_seeds": [
                    "Construct adaptive prototype geometry with optimal-transport evidence.",
                    "Use bidirectional hyperedge convolution to exchange node- and hyperedge-level context.",
                ],
            },
            "presentation_plan": {
                "figures": [
                    {
                        "label": "fig:method-overview",
                        "title": "Method Overview",
                        "section": "Method",
                        "asset_path": "figures/method_overview.pdf",
                        "caption": "Overview of the proposed method.",
                        "evidence": [],
                        "status": "planned",
                    },
                    {
                        "label": "fig:prototype-hypergraph",
                        "title": "Adaptive Prototype Hypergraph Construction",
                        "section": "Method",
                        "asset_path": "figures/prototype_hypergraph.pdf",
                        "caption": "Adaptive prototype and hypergraph construction.",
                        "evidence": [],
                        "status": "planned",
                    },
                ],
                "tables": [],
                "open_items": [
                    "Create or attach the planned figure asset `figures/method_overview.pdf` for `fig:method-overview`.",
                    "Create or attach the planned figure asset `figures/prototype_hypergraph.pdf` for `fig:prototype-hypergraph`.",
                ],
            },
        },
    }

    LatexComposerAgent().run(state)

    method_pdf = state["latex_project_dir"] / "figures" / "method_overview.pdf"
    hypergraph_pdf = state["latex_project_dir"] / "figures" / "prototype_hypergraph.pdf"
    tex = state["latex_output_path"].read_text(encoding="utf-8")
    plan = (state["latex_project_dir"] / "FIGURE_TABLE_PLAN.md").read_text(encoding="utf-8")
    assert method_pdf.exists()
    assert method_pdf.read_bytes().startswith(b"%PDF-1.4")
    assert hypergraph_pdf.exists()
    assert hypergraph_pdf.read_bytes().startswith(b"%PDF-1.4")
    assert r"\includegraphics[width=\columnwidth]{figures/method_overview.pdf}" in tex
    assert r"\includegraphics[width=\columnwidth]{figures/prototype_hypergraph.pdf}" in tex
    assert state["artifacts"]["generated_figure_count"] == 2
    assert {item["label"] for item in state["artifacts"]["generated_figures"]} == {
        "fig:method-overview",
        "fig:prototype-hypergraph",
    }
    assert "Status: generated" in plan
    assert state["artifacts"]["presentation_plan"]["open_items"] == []

    SubmissionPackageValidatorAgent().run(state)
    assert not state["artifacts"]["submission_package"]["errors"]
    assert state["artifacts"]["submission_package"]["checks"]["missing_graphics"] == []


def test_draft_report_includes_presentation_plan(tmp_path):
    state = {
        "request": PaperRequest(project_name="presentation-report-demo", target_venue="TPAMI"),
        "latex_project_dir": tmp_path,
        "artifacts": {
            "presentation_plan_path": str(tmp_path / "FIGURE_TABLE_PLAN.md"),
            "presentation_plan": {
                "figures": [
                    {
                        "label": "fig:method-overview",
                        "section": "Method",
                        "caption": "Overview of the proposed method.",
                    }
                ],
                "tables": [
                    {"label": "tab:main-results", "caption": "Main result table."}
                ],
                "open_items": ["Create the method overview figure."],
            }
        },
    }

    DraftReportAgent().run(state)

    report = (tmp_path / "DRAFT_REPORT.md").read_text(encoding="utf-8")
    assert "## Figure and Table Plan" in report
    assert "- Planned figures: 1" in report
    assert "- Generated figures: 0" in report
    assert "`fig:method-overview`" in report
    assert "Create the method overview figure." in report


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
