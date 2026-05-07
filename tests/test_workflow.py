from paper_agent.state import PaperRequest
from paper_agent.workflow import PaperWorkflow
from paper_agent.agents.evidence_guard import EvidenceGuardAgent
from paper_agent.agents.experiment_analyzer import ExperimentAnalyzerAgent
from paper_agent.state import CodeSummary, DraftSections, ExperimentSummary


def test_workflow_generates_latex_and_sections():
    request = PaperRequest(
        project_name="demo-paper",
        target_venue="IEEE Conference",
        method_notes="Adaptive feature calibration",
        experiment_results="baseline accuracy 80, ours accuracy 83 on DatasetA",
    )

    state = PaperWorkflow().run(request)

    assert state["sections"].abstract
    assert state["innovations"]
    assert state["venue_template"].family == "ieee"
    assert state["latex_output_path"].name == "main.tex"


def test_tpami_uses_ieee_journal_template():
    request = PaperRequest(
        project_name="demo-paper",
        target_venue="TPAMI",
        method_notes="Adaptive feature calibration",
    )

    state = PaperWorkflow().run(request)

    assert state["venue_template"].family == "ieee_journal"


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
