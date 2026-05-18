"""Experiment result alignment checks."""

from __future__ import annotations

from paper_agent.state import ExperimentSummary


DEFAULT_TCGA_DATASETS = ["BLCA", "BRCA", "LGG", "LUAD", "UCEC"]
DEFAULT_TCGA_METRICS = ["C-INDEX"]
DEFAULT_TCGA_METHOD = "Hyper-ProtoSurv"
DEFAULT_TCGA_BASELINE = "ProtoSurv"


def assess_experiment_quality(
    experiments: ExperimentSummary | None,
    *,
    expected_datasets: list[str] | None = None,
    expected_metrics: list[str] | None = None,
    expected_method: str = "",
    expected_baseline: str = "",
) -> dict[str, object]:
    """Check whether parsed result tables match a declared experiment target."""

    expected_datasets = _normalize_list(expected_datasets or [])
    expected_metrics = _normalize_list(expected_metrics or [])
    expected_method_norm = _normalize(expected_method)
    expected_baseline_norm = _normalize(expected_baseline)
    configured = bool(
        expected_datasets
        or expected_metrics
        or expected_method_norm
        or expected_baseline_norm
    )
    if not configured:
        return {
            "status": "not_configured",
            "errors": [],
            "warnings": [],
            "checks": {},
        }

    experiments = experiments or ExperimentSummary()
    parsed_datasets = _normalize_list(experiments.datasets)
    parsed_metrics = _normalize_list(experiments.metrics)
    table_methods = _normalize_list(table.method for table in experiments.result_tables)
    table_baselines = _normalize_list(table.baseline for table in experiments.result_tables)

    errors: list[str] = []
    warnings: list[str] = []
    missing_datasets = [item for item in expected_datasets if item not in parsed_datasets]
    missing_metrics = [item for item in expected_metrics if item not in parsed_metrics]
    unexpected_datasets = [
        item for item in parsed_datasets if expected_datasets and item not in expected_datasets
    ]

    if missing_datasets:
        errors.append("Missing expected datasets: " + ", ".join(missing_datasets) + ".")
    if missing_metrics:
        errors.append("Missing expected metrics: " + ", ".join(missing_metrics) + ".")
    if expected_method_norm and not _contains_expected(table_methods, expected_method_norm):
        errors.append(f"Proposed-method rows do not match expected method: {expected_method}.")
    if expected_baseline_norm and not _contains_expected(table_baselines, expected_baseline_norm):
        errors.append(f"Baseline rows do not match expected baseline: {expected_baseline}.")
    if unexpected_datasets:
        warnings.append("Unexpected parsed datasets: " + ", ".join(unexpected_datasets) + ".")

    return {
        "status": "invalid" if errors else "needs_attention" if warnings else "complete",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "expected_datasets": expected_datasets,
            "parsed_datasets": parsed_datasets,
            "missing_datasets": missing_datasets,
            "unexpected_datasets": unexpected_datasets,
            "expected_metrics": expected_metrics,
            "parsed_metrics": parsed_metrics,
            "missing_metrics": missing_metrics,
            "expected_method": expected_method,
            "expected_baseline": expected_baseline,
            "parsed_methods": table_methods,
            "parsed_baselines": table_baselines,
        },
    }


def tcga_experiment_quality_kwargs() -> dict[str, object]:
    return {
        "expected_datasets": DEFAULT_TCGA_DATASETS,
        "expected_metrics": DEFAULT_TCGA_METRICS,
        "expected_method": DEFAULT_TCGA_METHOD,
        "expected_baseline": DEFAULT_TCGA_BASELINE,
    }


def _normalize_list(values) -> list[str]:
    return list(dict.fromkeys(_normalize(str(value)) for value in values if _normalize(str(value))))


def _normalize(value: str) -> str:
    return " ".join(value.strip().upper().replace("_", "-").split())


def _contains_expected(values: list[str], expected: str) -> bool:
    expected_compact = _compact(expected)
    return any(expected_compact in _compact(value) for value in values)


def _compact(value: str) -> str:
    return "".join(ch for ch in value.upper() if ch.isalnum())
