import csv
import json
from pathlib import Path

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


def test_cli_tcga_export_artifacts_converts_flat_training_csv(monkeypatch, tmp_path, capsys):
    input_csv = tmp_path / "training_summary.csv"
    fieldnames = [
        "role",
        "method",
        "dataset",
        "metric",
        "fold",
        "seed",
        "value",
        "variant",
        "parameter",
        "parameter_value",
        "comparison",
        "test",
        "p_value",
    ]
    rows = [
        {"role": "main", "method": "ProtoSurv baseline", "dataset": "BLCA", "metric": "C-index", "fold": 0, "seed": 2026, "value": 0.646},
        {"role": "main", "method": "ProtoSurv baseline", "dataset": "BRCA", "metric": "C-index", "fold": 0, "seed": 2026, "value": 0.669},
        {"role": "main", "method": "Hyper-ProtoSurv ours", "dataset": "BLCA", "metric": "C-index", "fold": 0, "seed": 2026, "value": 0.671},
        {"role": "main", "method": "Hyper-ProtoSurv ours", "dataset": "BRCA", "metric": "C-index", "fold": 0, "seed": 2026, "value": 0.691},
        {"role": "ablation", "dataset": "Average", "metric": "C-index", "fold": 0, "seed": 2026, "value": 0.681, "variant": "Hyper-ProtoSurv ours"},
        {"role": "ablation", "dataset": "Average", "metric": "C-index", "fold": 0, "seed": 2026, "value": 0.665, "variant": "w/o reconstruction loss"},
        {"role": "sensitivity", "dataset": "Average", "metric": "C-index", "fold": 0, "seed": 2026, "value": 0.676, "parameter": "lambda_rec", "parameter_value": 0.5},
        {"role": "sensitivity", "dataset": "Average", "metric": "C-index", "fold": 0, "seed": 2026, "value": 0.681, "parameter": "lambda_rec", "parameter_value": 1.0},
        {"role": "stats", "metric": "C-index", "comparison": "Hyper-ProtoSurv ours vs ProtoSurv baseline", "test": "Wilcoxon signed-rank", "p_value": 0.018},
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-export-artifacts",
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(logs_dir),
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert "TCGA artifact exports written" in output
    assert (logs_dir / "tcga_main_results.csv").is_file()
    assert (logs_dir / "ARTIFACT_SCHEMA.json").is_file()

    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-results-from-artifacts",
            "--artifacts-dir",
            str(logs_dir),
            "--output",
            str(tmp_path / "tcga_results_from_flat.md"),
            "--strict",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    result_text = (tmp_path / "tcga_results_from_flat.md").read_text(encoding="utf-8")
    assert "Experiment artifact consistency: complete" in output
    assert "| Hyper-ProtoSurv ours | 0.671 | 0.691 |" in result_text


def test_cli_tcga_demo_artifact_flow_uses_bundled_example(monkeypatch, tmp_path, capsys):
    input_csv = Path(__file__).resolve().parents[1] / "examples" / "tcga_training_summary.csv"
    output_dir = tmp_path / "demo-flow"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-agent",
            "tcga-demo-artifact-flow",
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--force",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert "TCGA demo artifacts written" in output
    assert "TCGA demo result Markdown written" in output
    assert "Experiment artifact consistency: complete" in output
    assert (output_dir / "artifacts" / "tcga_main_results.csv").is_file()
    assert (output_dir / "artifacts" / "ARTIFACT_SCHEMA.json").is_file()
    result_text = (output_dir / "tcga_results.md").read_text(encoding="utf-8")
    assert "| Hyper-ProtoSurv ours | 0.671 | 0.691 | 0.746 | 0.661 | 0.681 |" in result_text
