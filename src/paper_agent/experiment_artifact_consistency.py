"""Cross-check paper result tables against provenance artifacts."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from paper_agent.state import ExperimentComparison, ExperimentSummary


DEFAULT_VALUE_TOLERANCE = 1e-3


def assess_experiment_artifact_consistency(
    experiments: ExperimentSummary | None,
    provenance: dict[str, object],
    *,
    require_consistency: bool = False,
    tolerance: float = DEFAULT_VALUE_TOLERANCE,
) -> dict[str, object]:
    """Check whether local CSV provenance artifacts support parsed paper values."""

    comparisons = [
        comparison
        for table in (experiments.result_tables if experiments else [])
        for comparison in table.comparisons
    ]
    paper_values = _paper_values(comparisons)
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
        "matched_values": 0,
        "missing_values": 0,
        "mismatched_values": 0,
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
            if _same_label(str(expected["method"]), str(value.get("method", "")))
            and _same_label(str(expected["dataset"]), str(value.get("dataset", "")))
            and _same_metric(str(expected["metric"]), str(value.get("metric", "")))
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


def _paper_values(comparisons: list[ExperimentComparison]) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for comparison in comparisons:
        values.append(
            {
                "role": "method",
                "method": comparison.method,
                "dataset": comparison.dataset,
                "metric": comparison.metric,
                "value": comparison.method_value,
            }
        )
        values.append(
            {
                "role": "baseline",
                "method": comparison.baseline,
                "dataset": comparison.dataset,
                "metric": comparison.metric,
                "value": comparison.baseline_value,
            }
        )
    return values


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
            return values, ""
    except OSError as exc:
        return [], f"CSV provenance artifact could not be read: {entry.get('path', path)} ({exc})."


def _csv_schema(headers: list[str]) -> dict[str, str] | None:
    method = _find_column(headers, ["method", "model", "variant", "approach"])
    dataset = _find_column(headers, ["dataset", "cohort", "cancer", "project"])
    metric = _find_column(headers, ["metric", "measure"])
    value = _find_column(headers, ["value", "score", "result", "mean", "estimate"])
    if not value:
        value = _find_metric_value_column(headers)
    if method and dataset and value:
        return {"method": method, "dataset": dataset, "metric": metric or "", "value": value}
    return None


def _csv_row_value(
    row: dict[str, str],
    schema: dict[str, str],
    entry: dict[str, object],
    row_number: int,
) -> dict[str, object] | None:
    raw_value = row.get(schema["value"], "")
    value = _numeric_value(raw_value)
    if value is None:
        return None
    metric = row.get(schema["metric"], "") if schema.get("metric") else _metric_from_header(schema["value"])
    return {
        "method": row.get(schema["method"], ""),
        "dataset": row.get(schema["dataset"], ""),
        "metric": metric,
        "value": value,
        "artifact_path": entry.get("path", ""),
        "resolved_path": entry.get("resolved_path", ""),
        "row_number": row_number,
    }


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
