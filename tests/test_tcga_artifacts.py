import json

from paper_agent import cli as cli_module
from paper_agent.tcga_artifacts import (
    TCGAAblationResult,
    TCGAMainResult,
    TCGASensitivityResult,
    TCGAStatisticalTest,
    write_tcga_artifact_exports,
)


def test_tcga_artifact_exporter_writes_cli_consumable_artifacts(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    written = write_tcga_artifact_exports(
        logs_dir,
        main_results=[
            TCGAMainResult("ProtoSurv baseline", "BLCA", 0.646),
            TCGAMainResult("ProtoSurv baseline", "BRCA", 0.669),
            TCGAMainResult("Hyper-ProtoSurv ours", "BLCA", 0.671),
            TCGAMainResult("Hyper-ProtoSurv ours", "BRCA", 0.691),
        ],
        ablation_results=[
            TCGAAblationResult("Hyper-ProtoSurv ours", 0.681),
            TCGAAblationResult("w/o reconstruction loss", 0.665),
        ],
        sensitivity_results=[
            TCGASensitivityResult("lambda_rec", 0.5, 0.676),
            TCGASensitivityResult("lambda_rec", 1.0, 0.681),
        ],
        statistical_tests=[
            TCGAStatisticalTest(
                "Hyper-ProtoSurv ours vs ProtoSurv baseline",
                0.018,
            )
        ],
    )

    assert written["tcga_main_results.csv"].is_file()
    manifest = json.loads(written["ARTIFACT_SCHEMA.json"].read_text(encoding="utf-8"))
    assert manifest["source"] == "paper_agent.tcga_artifacts.write_tcga_artifact_exports"
    assert manifest["datasets"] == ["BLCA", "BRCA"]
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-results-from-artifacts",
            "--artifacts-dir",
            str(logs_dir),
            "--output",
            str(tmp_path / "tcga_results.md"),
            "--strict",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    result_text = (tmp_path / "tcga_results.md").read_text(encoding="utf-8")
    assert "TCGA result file written" in output
    assert "Experiment artifact consistency: complete" in output
    assert "| Hyper-ProtoSurv ours | 0.671 | 0.691 |" in result_text


def test_tcga_artifact_exporter_rejects_todo_values(tmp_path):
    try:
        write_tcga_artifact_exports(
            tmp_path,
            main_results=[TCGAMainResult("ProtoSurv baseline", "BLCA", "TODO")],
            ablation_results=[TCGAAblationResult("Hyper-ProtoSurv ours", 0.681)],
            sensitivity_results=[TCGASensitivityResult("lambda_rec", 1.0, 0.681)],
            statistical_tests=[TCGAStatisticalTest("ours vs baseline", 0.018)],
        )
    except ValueError as exc:
        assert "TODO" in str(exc)
    else:
        raise AssertionError("Expected exporter to reject TODO artifact values.")
