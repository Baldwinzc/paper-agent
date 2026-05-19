"""Helpers for exporting TCGA result artifacts from training code."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TCGAMainResult:
    method: str
    dataset: str
    value: float | int | str
    metric: str = "C-index"
    fold: int | str = 0
    seed: int | str = 2026


@dataclass(frozen=True)
class TCGAAblationResult:
    variant: str
    value: float | int | str
    metric: str = "C-index"
    dataset: str = "Average"
    fold: int | str = 0
    seed: int | str = 2026


@dataclass(frozen=True)
class TCGASensitivityResult:
    parameter: str
    parameter_value: float | int | str
    value: float | int | str
    metric: str = "C-index"
    dataset: str = "Average"
    fold: int | str = 0
    seed: int | str = 2026


@dataclass(frozen=True)
class TCGAStatisticalTest:
    comparison: str
    p_value: float | int | str
    metric: str = "C-index"
    test: str = "Wilcoxon signed-rank"


RowInput = Mapping[str, object] | object


def write_tcga_artifact_exports(
    output_dir: str | Path,
    *,
    main_results: Sequence[RowInput],
    ablation_results: Sequence[RowInput] = (),
    sensitivity_results: Sequence[RowInput] = (),
    statistical_tests: Sequence[RowInput] = (),
    method: str = "Hyper-ProtoSurv ours",
    baseline: str = "ProtoSurv baseline",
    metric: str = "C-index",
    seed: int | str = 2026,
    force: bool = False,
    require_complete: bool = True,
) -> dict[str, Path]:
    """Write paper-agent compatible TCGA artifact CSVs.

    The inputs can be dataclass instances from this module or dictionaries with
    matching keys. This helper is intentionally dependency-free so training code
    can use it without pandas.
    """

    output_path = Path(output_dir)
    rows_by_name: dict[str, list[dict[str, str]]] = {
        "tcga_main_results.csv": [
            _main_row(row, default_metric=metric, default_seed=seed) for row in main_results
        ],
        "tcga_ablation.csv": [
            _ablation_row(row, default_metric=metric, default_seed=seed) for row in ablation_results
        ],
        "tcga_sensitivity.csv": [
            _sensitivity_row(row, default_metric=metric, default_seed=seed) for row in sensitivity_results
        ],
        "tcga_stats.csv": [_stats_row(row, default_metric=metric) for row in statistical_tests],
    }
    if require_complete:
        missing = [name for name, rows in rows_by_name.items() if not rows]
        if missing:
            raise ValueError("Missing required TCGA artifact rows for: " + ", ".join(missing))

    output_path.mkdir(parents=True, exist_ok=True)
    planned_names = [name for name, rows in rows_by_name.items() if rows]
    planned_names.append("ARTIFACT_SCHEMA.json")
    conflicts = [name for name in planned_names if (output_path / name).exists()]
    if conflicts and not force:
        raise FileExistsError(
            "Refusing to overwrite existing TCGA artifact export files: "
            + ", ".join(conflicts)
            + ". Pass force=True to overwrite."
        )

    written: dict[str, Path] = {}
    fieldnames = {
        "tcga_main_results.csv": ["method", "dataset", "metric", "fold", "seed", "value"],
        "tcga_ablation.csv": ["method", "dataset", "metric", "fold", "seed", "value"],
        "tcga_sensitivity.csv": [
            "parameter",
            "parameter_value",
            "dataset",
            "metric",
            "fold",
            "seed",
            "value",
        ],
        "tcga_stats.csv": ["comparison", "metric", "test", "p_value"],
    }
    for name, rows in rows_by_name.items():
        if not rows:
            continue
        path = output_path / name
        _write_csv(path, fieldnames[name], rows)
        written[name] = path

    manifest_path = output_path / "ARTIFACT_SCHEMA.json"
    manifest_path.write_text(
        json.dumps(
            _export_manifest(
                method=method,
                baseline=baseline,
                metric=metric,
                seed=str(seed),
                datasets=sorted({_main_row_dataset(row) for row in rows_by_name["tcga_main_results.csv"]}),
                files=sorted(written),
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    written["ARTIFACT_SCHEMA.json"] = manifest_path
    return written


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_dict(row: RowInput) -> dict[str, object]:
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, Mapping):
        return dict(row)
    raise TypeError(f"Unsupported TCGA artifact row type: {type(row).__name__}")


def _main_row(row: RowInput, *, default_metric: str, default_seed: int | str) -> dict[str, str]:
    values = _row_dict(row)
    return {
        "method": _required_text(values, "method"),
        "dataset": _required_text(values, "dataset"),
        "metric": _text(values.get("metric", default_metric), "metric"),
        "fold": _text(values.get("fold", 0), "fold"),
        "seed": _text(values.get("seed", default_seed), "seed"),
        "value": _text(values.get("value"), "value"),
    }


def _ablation_row(row: RowInput, *, default_metric: str, default_seed: int | str) -> dict[str, str]:
    values = _row_dict(row)
    variant = values.get("variant", values.get("method"))
    return {
        "method": _text(variant, "variant"),
        "dataset": _text(values.get("dataset", "Average"), "dataset"),
        "metric": _text(values.get("metric", default_metric), "metric"),
        "fold": _text(values.get("fold", 0), "fold"),
        "seed": _text(values.get("seed", default_seed), "seed"),
        "value": _text(values.get("value"), "value"),
    }


def _sensitivity_row(row: RowInput, *, default_metric: str, default_seed: int | str) -> dict[str, str]:
    values = _row_dict(row)
    return {
        "parameter": _required_text(values, "parameter"),
        "parameter_value": _required_text(values, "parameter_value"),
        "dataset": _text(values.get("dataset", "Average"), "dataset"),
        "metric": _text(values.get("metric", default_metric), "metric"),
        "fold": _text(values.get("fold", 0), "fold"),
        "seed": _text(values.get("seed", default_seed), "seed"),
        "value": _text(values.get("value"), "value"),
    }


def _stats_row(row: RowInput, *, default_metric: str) -> dict[str, str]:
    values = _row_dict(row)
    return {
        "comparison": _required_text(values, "comparison"),
        "metric": _text(values.get("metric", default_metric), "metric"),
        "test": _text(values.get("test", "Wilcoxon signed-rank"), "test"),
        "p_value": _required_text(values, "p_value"),
    }


def _main_row_dataset(row: dict[str, str]) -> str:
    return row["dataset"]


def _required_text(values: Mapping[str, object], key: str) -> str:
    return _text(values.get(key), key)


def _text(value: object, label: str) -> str:
    if value is None:
        raise ValueError(f"TCGA artifact field `{label}` is required.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"TCGA artifact field `{label}` is empty.")
    if text.upper() == "TODO":
        raise ValueError(f"TCGA artifact field `{label}` still contains TODO.")
    return text


def _export_manifest(
    *,
    method: str,
    baseline: str,
    metric: str,
    seed: str,
    datasets: list[str],
    files: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "paper_agent.tcga_artifacts.write_tcga_artifact_exports",
        "method": method,
        "baseline": baseline,
        "metric": metric,
        "seed": seed,
        "datasets": datasets,
        "files": files,
        "validation_commands": [
            "paper-agent tcga-artifacts-doctor --artifacts-dir . --summary artifact-doctor.json",
            "paper-agent tcga-results-from-artifacts --artifacts-dir . --strict",
        ],
    }
