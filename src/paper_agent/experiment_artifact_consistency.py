"""Cross-check paper result tables against provenance artifacts."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

from paper_agent.state import ExperimentSummary


DEFAULT_VALUE_TOLERANCE = 1e-3


def assess_experiment_artifact_consistency(
    experiments: ExperimentSummary | None,
    provenance: dict[str, object],
    *,
    require_consistency: bool = False,
    tolerance: float = DEFAULT_VALUE_TOLERANCE,
) -> dict[str, object]:
    """Check whether local CSV provenance artifacts support parsed paper values."""

    paper_values = _paper_values(experiments)
    csv_entries = [
        entry
        for entry in provenance.get("entries", [])
        if isinstance(entry, dict)
        and entry.get("kind") == "local"
        and entry.get("is_file")
        and str(entry.get("resolved_path", "")).lower().endswith(".csv")
    ]
    checks = {
        "csv_artifacts": len(csv_entries),
        "checkable_csv_artifacts": 0,
        "paper_values": len(paper_values),
        "main_values": sum(1 for value in paper_values if str(value.get("role", "")).startswith("main_")),
        "ablation_values": sum(1 for value in paper_values if str(value.get("role", "")).startswith("ablation_")),
        "sensitivity_values": sum(1 for value in paper_values if value.get("role") == "sensitivity"),
        "statistical_values": sum(1 for value in paper_values if value.get("role") == "statistical_test"),
        "matched_values": 0,
        "missing_values": 0,
        "mismatched_values": 0,
        "aggregated_values": 0,
        "tolerance": tolerance,
    }
    if not paper_values:
        return _result("not_configured", checks=checks)
    if not csv_entries:
        if require_consistency:
            return _result(
                "invalid",
                errors=["No local CSV provenance artifact is available for result-value consistency checks."],
                checks=checks,
            )
        return _result("not_configured", checks=checks)

    artifact_values: list[dict[str, object]] = []
    parse_warnings: list[str] = []
    for entry in csv_entries:
        values, warning = _read_csv_values(entry)
        if values:
            checks["checkable_csv_artifacts"] += 1
            checks["aggregated_values"] += sum(1 for value in values if int(value.get("fold_count", 1) or 1) > 1)
            artifact_values.extend(values)
        elif warning:
            parse_warnings.append(warning)

    if not artifact_values:
        if require_consistency:
            return _result(
                "invalid",
                errors=[
                    "No checkable result rows were found in local CSV provenance artifacts. "
                    "Use columns such as method,dataset,metric,value."
                ],
                warnings=parse_warnings,
                checks=checks,
            )
        return _result("not_configured", warnings=parse_warnings, checks=checks)

    matches: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    for expected in paper_values:
        candidates = [
            value
            for value in artifact_values
            if _expected_matches_artifact(expected, value)
        ]
        if not candidates:
            missing.append(expected)
            continue
        best = min(candidates, key=lambda item: abs(float(item["value"]) - float(expected["value"])))
        delta = abs(float(best["value"]) - float(expected["value"]))
        record = {
            **expected,
            "artifact_value": best["value"],
            "artifact_path": best.get("artifact_path", ""),
            "row_number": best.get("row_number", 0),
            "absolute_delta": delta,
            "aggregation": best.get("aggregation", ""),
            "fold_count": best.get("fold_count", 1),
            "artifact_std": best.get("std", 0.0),
        }
        if delta <= tolerance:
            matches.append(record)
        else:
            mismatches.append(record)

    checks["matched_values"] = len(matches)
    checks["missing_values"] = len(missing)
    checks["mismatched_values"] = len(mismatches)
    errors = [
        (
            f"Artifact value mismatch for {item['role']} {item['method']} "
            f"{item['dataset']} {item['metric']}: paper={item['value']}, "
            f"artifact={item['artifact_value']} ({item['artifact_path']} row {item['row_number']})."
        )
        for item in mismatches[:8]
    ]
    warnings: list[str] = []
    missing_messages = [
        (
            f"No artifact value found for {item['role']} {item['method']} "
            f"{item['dataset']} {item['metric']}."
        )
        for item in missing[:8]
    ]
    if require_consistency:
        errors.extend(missing_messages)
    else:
        warnings.extend(missing_messages)
    if missing and parse_warnings:
        warnings.extend(parse_warnings)

    status = "invalid" if errors else "needs_attention" if warnings else "complete"
    return _result(
        status,
        errors=errors,
        warnings=warnings,
        checks=checks,
        matches=matches,
        missing=missing,
        mismatches=mismatches,
    )


def _paper_values(experiments: ExperimentSummary | None) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    if not experiments:
        return values
    for table in experiments.result_tables:
        for comparison in table.comparisons:
            values.append(
                {
                    "role": "main_method",
                    "method": comparison.method,
                    "dataset": comparison.dataset,
                    "metric": comparison.metric,
                    "value": comparison.method_value,
                }
            )
            values.append(
                {
                    "role": "main_baseline",
                    "method": comparison.baseline,
                    "dataset": comparison.dataset,
                    "metric": comparison.metric,
                    "value": comparison.baseline_value,
                }
            )
    for item in experiments.ablation_evidence:
        values.append(
            {
                "role": "ablation_reference",
                "method": item.reference,
                "dataset": item.dataset,
                "metric": item.metric,
                "value": item.reference_value,
            }
        )
        values.append(
            {
                "role": "ablation_variant",
                "method": item.variant,
                "dataset": item.dataset,
                "metric": item.metric,
                "value": item.variant_value,
            }
        )
    for item in experiments.sensitivity_evidence:
        for parameter_value, metric_value in zip(item.tested_values, item.metric_values):
            values.append(
                {
                    "role": "sensitivity",
                    "method": f"{item.parameter}={parameter_value}",
                    "parameter": item.parameter,
                    "parameter_value": parameter_value,
                    "dataset": item.dataset,
                    "metric": item.metric,
                    "value": metric_value,
                }
            )
    for item in experiments.statistical_tests:
        values.append(
            {
                "role": "statistical_test",
                "method": item.comparison,
                "comparison": item.comparison,
                "dataset": "",
                "metric": item.metric,
                "test": item.test,
                "value": item.p_value,
            }
        )
    return _dedupe_paper_values(values)


def _dedupe_paper_values(values: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
    for value in values:
        key = (
            str(value.get("role", "")),
            _normalize_label(str(value.get("method", ""))),
            _normalize_label(str(value.get("dataset", ""))),
            _normalize_metric(str(value.get("metric", ""))),
            _normalize_label(str(value.get("parameter", ""))),
            _normalize_label(str(value.get("parameter_value", ""))),
            _normalize_label(str(value.get("comparison", ""))),
            _normalize_label(str(value.get("test", ""))),
        )
        deduped.setdefault(key, value)
    return list(deduped.values())


def _read_csv_values(entry: dict[str, object]) -> tuple[list[dict[str, object]], str]:
    path = Path(str(entry.get("resolved_path", "")))
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            schema = _csv_schema(headers)
            if not schema:
                return [], f"CSV provenance artifact has no supported result schema: {entry.get('path', path)}."
            values: list[dict[str, object]] = []
            for row_number, row in enumerate(reader, start=2):
                parsed = _csv_row_value(row, schema, entry, row_number)
                if parsed:
                    values.append(parsed)
            if not values:
                return [], f"CSV provenance artifact has no numeric result rows: {entry.get('path', path)}."
            return _aggregate_csv_values(values), ""
    except OSError as exc:
        return [], f"CSV provenance artifact could not be read: {entry.get('path', path)} ({exc})."


def _csv_schema(headers: list[str]) -> dict[str, str] | None:
    method = _find_column(headers, ["method", "model", "variant", "approach"])
    dataset = _find_column(headers, ["dataset", "cohort", "cancer", "project"])
    metric = _find_column(headers, ["metric", "measure"])
    parameter = _find_column(headers, ["parameter", "hyperparameter", "param"])
    parameter_value = _find_column(headers, ["parameter_value", "param_value", "setting", "tested_value"])
    comparison = _find_column(headers, ["comparison", "contrast", "pair"])
    test = _find_column(headers, ["test", "statistical_test", "statistic"])
    p_value = _find_p_value_column(headers)
    fold = _find_column(headers, ["fold", "split"])
    seed = _find_column(headers, ["seed", "run"])
    value = _find_column(headers, ["value", "score", "result", "mean", "estimate"])
    if not value:
        value = _find_metric_value_column(headers)
    has_result_schema = bool(dataset and value and (method or (parameter and parameter_value)))
    has_statistical_schema = bool(comparison and p_value)
    if has_result_schema or has_statistical_schema:
        return {
            "method": method,
            "dataset": dataset,
            "metric": metric or "",
            "parameter": parameter,
            "parameter_value": parameter_value,
            "comparison": comparison,
            "test": test,
            "p_value": p_value,
            "fold": fold,
            "seed": seed,
            "value": value,
        }
    return None


def _csv_row_value(
    row: dict[str, str],
    schema: dict[str, str],
    entry: dict[str, object],
    row_number: int,
) -> dict[str, object] | None:
    if schema.get("comparison") and schema.get("p_value"):
        p_value = _numeric_value(row.get(schema["p_value"], ""))
        comparison = row.get(schema["comparison"], "").strip()
        if p_value is not None and comparison:
            return {
                "kind": "statistical",
                "method": comparison,
                "comparison": comparison,
                "test": row.get(schema["test"], "") if schema.get("test") else "",
                "dataset": "",
                "metric": row.get(schema["metric"], "") if schema.get("metric") else "",
                "value": p_value,
                "artifact_path": entry.get("path", ""),
                "resolved_path": entry.get("resolved_path", ""),
                "row_number": row_number,
            }
    if not schema.get("value") or not schema.get("dataset"):
        return None
    raw_value = row.get(schema["value"], "")
    value = _numeric_value(raw_value)
    if value is None:
        return None
    method = row.get(schema["method"], "") if schema.get("method") else ""
    parameter = row.get(schema["parameter"], "") if schema.get("parameter") else ""
    parameter_value = row.get(schema["parameter_value"], "") if schema.get("parameter_value") else ""
    if not method and not (parameter and parameter_value):
        return None
    metric = row.get(schema["metric"], "") if schema.get("metric") else _metric_from_header(schema["value"])
    return {
        "kind": "result",
        "method": method,
        "parameter": parameter,
        "parameter_value": parameter_value,
        "dataset": row.get(schema["dataset"], ""),
        "metric": metric,
        "value": value,
        "fold": row.get(schema["fold"], "") if schema.get("fold") else "",
        "seed": row.get(schema["seed"], "") if schema.get("seed") else "",
        "artifact_path": entry.get("path", ""),
        "resolved_path": entry.get("resolved_path", ""),
        "row_number": row_number,
    }


def _aggregate_csv_values(values: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str, str, str, str], list[dict[str, object]]] = {}
    for value in values:
        key = (
            str(value.get("kind", "result")),
            _normalize_label(str(value.get("method", ""))),
            _normalize_label(str(value.get("parameter", ""))),
            _normalize_label(str(value.get("parameter_value", ""))),
            _normalize_label(str(value.get("comparison", ""))),
            _normalize_label(str(value.get("test", ""))),
            _normalize_label(str(value.get("dataset", ""))),
            _normalize_metric(str(value.get("metric", ""))),
            str(value.get("artifact_path", "")),
        )
        groups.setdefault(key, []).append(value)

    aggregated: list[dict[str, object]] = []
    for group in groups.values():
        if str(group[0].get("kind", "result")) == "statistical":
            for raw_item in group:
                item = dict(raw_item)
                item.setdefault("fold_count", 1)
                item.setdefault("std", 0.0)
                item.setdefault("aggregation", "single_row")
                aggregated.append(item)
            continue
        if len(group) == 1:
            item = dict(group[0])
            item.setdefault("fold_count", 1)
            item.setdefault("std", 0.0)
            item.setdefault("aggregation", "single_row")
            aggregated.append(item)
            continue
        numeric_values = [float(item["value"]) for item in group]
        row_numbers = [str(item.get("row_number", "")) for item in group if item.get("row_number")]
        folds = [str(item.get("fold", "")) for item in group if str(item.get("fold", "")).strip()]
        seeds = [str(item.get("seed", "")) for item in group if str(item.get("seed", "")).strip()]
        first = group[0]
        aggregated.append(
            {
                "kind": first.get("kind", "result"),
                "method": first.get("method", ""),
                "parameter": first.get("parameter", ""),
                "parameter_value": first.get("parameter_value", ""),
                "comparison": first.get("comparison", ""),
                "test": first.get("test", ""),
                "dataset": first.get("dataset", ""),
                "metric": first.get("metric", ""),
                "value": sum(numeric_values) / len(numeric_values),
                "std": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
                "fold_count": len(numeric_values),
                "folds": list(dict.fromkeys(folds)),
                "seeds": list(dict.fromkeys(seeds)),
                "artifact_path": first.get("artifact_path", ""),
                "resolved_path": first.get("resolved_path", ""),
                "row_number": ",".join(row_numbers),
                "aggregation": "mean",
            }
        )
    return aggregated


def _expected_matches_artifact(expected: dict[str, object], value: dict[str, object]) -> bool:
    if expected.get("role") == "statistical_test":
        return bool(
            _same_label(str(expected.get("comparison", "")), str(value.get("comparison", value.get("method", ""))))
            and _metric_matches_optional(str(expected.get("metric", "")), str(value.get("metric", "")))
            and _label_matches_optional(str(expected.get("test", "")), str(value.get("test", "")))
        )
    if expected.get("role") == "sensitivity":
        return bool(
            _same_label(str(expected.get("parameter", "")), str(value.get("parameter", "")))
            and _same_label(str(expected.get("parameter_value", "")), str(value.get("parameter_value", "")))
            and _same_label(str(expected.get("dataset", "")), str(value.get("dataset", "")))
            and _same_metric(str(expected.get("metric", "")), str(value.get("metric", "")))
        )
    return bool(
        _same_label(str(expected.get("method", "")), str(value.get("method", "")))
        and _same_label(str(expected.get("dataset", "")), str(value.get("dataset", "")))
        and _same_metric(str(expected.get("metric", "")), str(value.get("metric", "")))
    )


def _metric_matches_optional(expected: str, actual: str) -> bool:
    if not expected or not actual:
        return True
    return _same_metric(expected, actual)


def _label_matches_optional(expected: str, actual: str) -> bool:
    if not expected or not actual:
        return True
    return _same_label(expected, actual)


def _find_column(headers: list[str], names: list[str]) -> str:
    normalized = {_normalize_header(header): header for header in headers}
    for name in names:
        key = _normalize_header(name)
        if key in normalized:
            return normalized[key]
    for header in headers:
        header_key = _normalize_header(header)
        if any(_normalize_header(name) in header_key for name in names):
            return header
    return ""


def _find_p_value_column(headers: list[str]) -> str:
    for header in headers:
        key = _normalize_header(header)
        if key in {"p", "pvalue", "pval"}:
            return header
    for header in headers:
        if re.search(r"\bp\s*[-_ ]?value\b|\bp\s*[-_ ]?val\b", header, flags=re.I):
            return header
    return ""


def _find_metric_value_column(headers: list[str]) -> str:
    for header in headers:
        if _metric_from_header(header):
            return header
    return ""


def _metric_from_header(header: str) -> str:
    lowered = header.lower()
    if re.search(r"\bc[-_ ]?index\b|concordance", lowered):
        return "C-INDEX"
    if re.search(r"\bibs\b|integrated brier", lowered):
        return "IBS"
    if re.search(r"\bauc\b", lowered):
        return "AUC"
    return ""


def _numeric_value(text: object) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
    if not match:
        return None
    return float(match.group(0))


def _same_label(expected: str, actual: str) -> bool:
    expected_key = _normalize_label(expected)
    actual_key = _normalize_label(actual)
    return bool(
        expected_key
        and actual_key
        and (expected_key == actual_key or expected_key in actual_key or actual_key in expected_key)
    )


def _same_metric(expected: str, actual: str) -> bool:
    expected_key = _normalize_metric(expected)
    actual_key = _normalize_metric(actual)
    return bool(expected_key and actual_key and expected_key == actual_key)


def _normalize_metric(metric: str) -> str:
    key = _normalize_label(metric)
    aliases = {
        "cindex": "cindex",
        "concordanceindex": "cindex",
        "ibs": "ibs",
        "integratedbrierscore": "ibs",
        "auc": "auc",
    }
    return aliases.get(key, key)


def _normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _normalize_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _result(
    status: str,
    *,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    checks: dict[str, object] | None = None,
    matches: list[dict[str, object]] | None = None,
    missing: list[dict[str, object]] | None = None,
    mismatches: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "errors": list(dict.fromkeys(errors or [])),
        "warnings": list(dict.fromkeys(warnings or [])),
        "checks": checks or {},
        "matches": matches or [],
        "missing": missing or [],
        "mismatches": mismatches or [],
    }
