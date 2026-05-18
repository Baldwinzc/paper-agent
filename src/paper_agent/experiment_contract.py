"""Validation contract for real experiment result files."""

from __future__ import annotations

from paper_agent.state import ExperimentSummary


def validate_experiment_contract(experiments: ExperimentSummary | None) -> dict[str, object]:
    """Return structural checks for a result file intended to support paper claims."""

    errors: list[str] = []
    warnings: list[str] = []

    result_tables = experiments.result_tables if experiments else []
    comparisons = [
        comparison
        for table in result_tables
        for comparison in table.comparisons
    ]
    datasets = experiments.datasets if experiments else []
    metrics = experiments.metrics if experiments else []
    ablations = experiments.ablation_evidence if experiments else []
    sensitivity = experiments.sensitivity_evidence if experiments else []
    statistical_tests = experiments.statistical_tests if experiments else []

    if not result_tables:
        errors.append("Missing main trained-model result table with proposed-method and baseline rows.")
    if result_tables and not comparisons:
        errors.append("No numeric proposed-method versus baseline comparisons were parsed.")
    if not datasets:
        errors.append("Dataset names are not explicit.")
    if not metrics:
        errors.append("Evaluation metrics are not explicit.")

    if not ablations:
        warnings.append("Missing ablation table; component claims should remain provisional.")
    if not sensitivity:
        warnings.append("Missing sensitivity analysis table; hyperparameter robustness claims should be omitted.")
    if not statistical_tests:
        warnings.append("Missing statistical-test table; significance claims should be omitted.")

    checks = {
        "result_tables": len(result_tables),
        "numeric_comparisons": len(comparisons),
        "datasets": len(datasets),
        "metrics": len(metrics),
        "ablation_items": len(ablations),
        "sensitivity_items": len(sensitivity),
        "statistical_tests": len(statistical_tests),
    }
    return {
        "status": "invalid" if errors else "needs_attention" if warnings else "complete",
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "checks": checks,
    }


def experiment_results_template(
    *,
    method: str = "Hyper-ProtoSurv ours",
    baseline: str = "ProtoSurv baseline",
    datasets: list[str] | None = None,
) -> str:
    """Create a fill-in Markdown template for real result files."""

    datasets = datasets or ["BLCA", "BRCA", "LGG", "LUAD", "UCEC"]
    metric_headers = " | ".join(f"{dataset} C-index" for dataset in datasets)
    empty_values = " | ".join("TODO" for _ in datasets)
    return "\n".join(
        [
            "# Real Experiment Results",
            "",
            "Replace every TODO with real trained-model outputs before using this file for a paper draft.",
            "Do not put cohort metadata, mock numbers, or synthetic pipeline checks in this file.",
            "",
            "## Main Results",
            "",
            "Metric: C-index. Higher is better.",
            "",
            f"| Method | {metric_headers} |",
            f"|---|{'|'.join('---:' for _ in datasets)}|",
            f"| {baseline} | {empty_values} |",
            f"| {method} | {empty_values} |",
            "",
            "## Ablation Study",
            "",
            "Metric: Average C-index. Higher is better.",
            "",
            "| Variant | Average C-index |",
            "|---|---:|",
            f"| {method} | TODO |",
            "| w/o key component 1 | TODO |",
            "| w/o key component 2 | TODO |",
            "",
            "## Sensitivity Analysis",
            "",
            "Metric: Average C-index. Higher is better.",
            "",
            "| lambda_rec | Average C-index |",
            "|---:|---:|",
            "| 0.1 | TODO |",
            "| 0.5 | TODO |",
            "| 1.0 | TODO |",
            "| 2.0 | TODO |",
            "",
            "## Statistical Testing",
            "",
            "| Comparison | Metric | Test | p-value |",
            "|---|---|---|---:|",
            f"| {method} vs {baseline} | C-index | Wilcoxon signed-rank | TODO |",
            "",
        ]
    )
