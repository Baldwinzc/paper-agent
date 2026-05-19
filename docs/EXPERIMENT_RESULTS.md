# Experiment Results Format

`paper-agent` accepts Markdown, CSV-like text, or pasted notes, but the most reliable
format is a Markdown table with one method column and one numeric column per
dataset-metric pair.

Use real trained-model results only. Cohort metadata is useful for dataset
description, but it is not evidence for performance claims.

Generate a fill-in template with:

```powershell
python -m paper_agent.cli experiment-template --output examples\tcga_results_template.md
```

The committed `examples/tcga_results_template.md` file uses the same structure.
The validator treats the main result table as required. Ablation, sensitivity,
and statistical-test tables are reported as contract warnings when absent; keep
component, robustness, or significance claims out of the draft until those tables
are supplied.

If the experiment pipeline does not yet export the required CSV schemas, create
the export contract and fill-in templates first:

```powershell
python -m paper_agent.cli tcga-artifact-template `
  --output-dir D:\code\agent\example\results\logs `
  --style long
```

This writes `tcga_main_results.csv`, `tcga_ablation.csv`,
`tcga_sensitivity.csv`, `tcga_stats.csv`, and `EXPORT_CONTRACT.md`. Replace every
`TODO` with real trained-model outputs. These files are templates, not evidence.

If the experiment pipeline already exports CSV artifacts, generate the Markdown
result file instead of copying values by hand:

```powershell
python -m paper_agent.cli tcga-preflight `
  --example-root D:\code\agent\example `
  --artifacts-dir D:\code\agent\example\results\logs `
  --summary outputs\tcga-preflight.json `
  --submission-grade
```

`tcga-preflight` is a read-only report for real-run readiness. It checks the
baseline PDF, code directory, result artifacts, existing or generatable result
Markdown, LLM configuration, and LaTeX compiler requirements. The JSON summary
contains every PASS/WARN/FAIL item and blocking item. When `--live-llm` is used,
it also records live provider status, elapsed time, token usage when available,
and sanitized failure diagnostics.

For the stricter doctor gate before drafting, write the same style of diagnostic
summary:

```powershell
python -m paper_agent.cli tcga-doctor `
  --example-root D:\code\agent\example `
  --summary outputs\tcga-doctor.json `
  --live-llm
```

If the live provider call fails, `llm_live_preflight.diagnostics.failure_kind`
classifies common configuration, authentication, quota, timeout, transport, and
model lookup failures without writing the API key. Failed doctor summaries also
write `next_actions`, so a TODO template, invalid result contract, quota failure,
or missing LaTeX compiler has a concrete command to run next. For TODO result
templates, the first command is `tcga-artifact-template`, followed by recorded
`tcga-results-from-artifacts --strict` and `validate-results --strict` commands.

```powershell
python -m paper_agent.cli tcga-artifacts-doctor `
  --artifacts-dir D:\code\agent\example\results\logs
```

`tcga-artifacts-doctor` reports which files were auto-detected for the main,
ablation, sensitivity, and statistical-test roles, then checks row counts,
columns, parsed value counts, and expected schemas. It exits non-zero when a
required role is missing or a CSV cannot support the paper-facing result table.

```powershell
python -m paper_agent.cli tcga-results-from-artifacts `
  --artifacts-dir D:\code\agent\example\results\logs `
  --output D:\code\agent\example\results\tcga_results.md `
  --strict
```

The artifact directory is scanned recursively for CSV files. The command
auto-detects main, ablation, sensitivity, and statistical-test artifacts from
file names and CSV schemas. If a single combined fold-level CSV contains all
roles, the same file is used for each section and written once in the provenance
table. Pass explicit files when auto-detection is ambiguous:

```powershell
python -m paper_agent.cli tcga-results-from-artifacts `
  --main-csv D:\code\agent\example\results\logs\tcga_main.csv `
  --ablation-csv D:\code\agent\example\results\logs\tcga_ablation.csv `
  --sensitivity-csv D:\code\agent\example\results\logs\tcga_sensitivity.csv `
  --stats-csv D:\code\agent\example\results\logs\tcga_stats.csv `
  --output D:\code\agent\example\results\tcga_results.md `
  --strict
```

`tcga-results-from-artifacts` writes the main, ablation, sensitivity,
statistical-test, and provenance sections, including SHA-256 digests for each
local artifact. It accepts wide CSV tables and long/fold-level CSV tables. For
repeated fold rows with the same method, dataset, and metric, it writes the mean
value and then, with `--strict`, checks that the generated Markdown is supported
by the source CSV artifacts.

Validate a completed result file before running the full paper workflow:

```powershell
python -m paper_agent.cli validate-results `
  --experiment-results D:\code\agent\example\results\tcga_results.md `
  --summary outputs\validate-results.json `
  --strict
```

`--strict` exits with a non-zero status unless the file is classified as real
result evidence and the experiment-result contract is complete.
The full `draft` command performs the same preflight and prints the same status.
Use `--strict-results` there to stop generation when the result file fails the
strict contract.
For the local Hyper-ProtoSurv TCGA example, use the dedicated real-result entry
after filling `D:\code\agent\example\results\tcga_results.md`:

```powershell
python -m paper_agent.cli tcga-draft `
  --example-root D:\code\agent\example `
  --experiment-results D:\code\agent\example\results\tcga_results.md `
  --output-dir outputs\hyper-protosurv-tcga-real
```

`tcga-draft` always runs the strict real-result preflight before spending LLM
generation calls. It fails on TODO templates, synthetic/mock result files, and
TCGA cohort metadata summaries.
By default, ablation, sensitivity, and statistical-test evidence are required for
a complete contract. Disable requirements that are out of scope with
`--no-require-ablation`, `--no-require-sensitivity`, or
`--no-require-statistical-tests`.

To generate the result file and immediately run the TCGA checks plus draft path,
use `tcga-pipeline`:

```powershell
python -m paper_agent.cli tcga-pipeline `
  --example-root D:\code\agent\example `
  --artifacts-dir D:\code\agent\example\results\logs `
  --output-dir outputs\hyper-protosurv-tcga-submission `
  --zip outputs\hyper-protosurv-tcga-submission-overleaf.zip `
  --submission-grade
```

The pipeline writes `results\tcga_results.md`, runs strict result validation,
runs `tcga-doctor`, and then calls `tcga-draft` with the generated result file.
Use `--skip-result-generation` only when the result Markdown already exists.
If the result CSV artifacts do not exist yet, add `--write-artifact-template`.
The pipeline writes the CSV templates and `EXPORT_CONTRACT.md`, then stops
before doctor checks or drafting. Replace every `TODO` with real trained-model
outputs and rerun the pipeline. Pipeline stops before or during doctor/draft also
write `RUN_SUMMARY.json` under `--output-dir` with the current phase, blocking
items, missing inputs, and next command. LLM failures include structured
provider diagnostics such as failure kind, model, endpoint host, timeout, and
retry settings without exposing API keys.

For result-file quality checks, declare the expected experiment target:

```powershell
python -m paper_agent.cli validate-results `
  --experiment-results D:\code\agent\example\results\tcga_results.md `
  --strict `
  --expected-dataset BLCA `
  --expected-dataset BRCA `
  --expected-dataset LGG `
  --expected-dataset LUAD `
  --expected-dataset UCEC `
  --expected-metric C-INDEX `
  --expected-method Hyper-ProtoSurv `
  --expected-baseline ProtoSurv
```

The quality report flags missing expected cohorts, missing metrics, and result
rows whose proposed method or baseline names do not match the declared target.
`tcga-draft` enables the common TCGA defaults automatically.

For submission-grade result files, also attach provenance for the numeric
tables. Provenance entries should point to fold-level CSVs, evaluation logs,
seed records, W&B exports, or other source artifacts. Local paths are resolved
relative to the result file; remote references such as `https://`, `s3://`,
`gs://`, `oss://`, and `wandb://` are accepted as external records.
For local files, the validator records `sha256` and byte size in the JSON
summary. If a `SHA256`/`Checksum` column is supplied, the computed digest must
match the declared value.

```powershell
python -m paper_agent.cli validate-results `
  --experiment-results D:\code\agent\example\results\tcga_results.md `
  --strict `
  --require-provenance `
  --require-artifact-consistency
```

Without `--require-provenance`, missing provenance is reported as a warning.
With `--require-provenance`, strict validation fails unless a provenance table is
present and all local artifact paths exist.
`--require-artifact-consistency` additionally fails strict validation unless a
checkable local CSV artifact supports every parsed method and baseline value in
the main result tables.

## Main Result Table

```markdown
## Main Results

Metric: C-index. Higher is better.

| Method | BLCA C-index | BRCA C-index | LGG C-index | LUAD C-index | UCEC C-index |
|---|---:|---:|---:|---:|---:|
| ProtoSurv baseline | 0.646 | 0.669 | 0.724 | 0.636 | 0.658 |
| Hyper-ProtoSurv ours | 0.671 | 0.691 | 0.746 | 0.661 | 0.681 |
```

The analyzer reads this as structured evidence:

- dataset: `BLCA`, `BRCA`, `LGG`, `LUAD`, `UCEC`
- metric: `C-INDEX`
- baseline value and proposed-method value per column
- signed improvement, respecting whether higher or lower is better

## Lower-Is-Better Metrics

For metrics such as `IBS`, `Brier`, `MAE`, `RMSE`, `loss`, or `error`, lower
values are treated as better.

```markdown
## Calibration Results

Metric: IBS. Lower is better.

| Method | BLCA IBS | BRCA IBS |
|---|---:|---:|
| ProtoSurv baseline | 0.180 | 0.210 |
| Hyper-ProtoSurv ours | 0.160 | 0.190 |
```

## Ablations

Ablation tables should keep the full method name for the proposed model and use
variant names for removed components.

```markdown
## Ablation Study

Metric: C-index. Higher is better.

| Variant | BLCA C-index | BRCA C-index |
|---|---:|---:|
| Hyper-ProtoSurv ours | 0.671 | 0.691 |
| w/o bidirectional HCoN | 0.655 | 0.674 |
| w/o reconstruction loss | 0.659 | 0.678 |
```

The analyzer reads these as component-level evidence against the full method. It
stores the reference value, variant value, signed drop, metric direction, and a
lightweight support tag such as `bidirectional hyperedge updates` or
`reconstruction regularization`. The draft then uses those records to write a
bounded ablation paragraph and to connect component evidence back to innovation
traceability. The agent should still avoid claims beyond the exact supplied
variant rows.

## Sensitivity Analysis

Use a parameter column plus one numeric metric column. Names such as `lambda_rec`,
`alpha`, `dropout`, or `temperature` are recognized as sensitivity parameters.

```markdown
## Sensitivity Analysis

Metric: Average C-index. Higher is better.

| lambda_rec | Average C-index |
|---:|---:|
| 0.1 | 0.681 |
| 0.5 | 0.687 |
| 1.0 | 0.690 |
| 2.0 | 0.686 |
```

The analyzer records the tested values, the best parameter value, the best metric
value, and whether higher or lower is better. The draft can then write a bounded
sensitivity paragraph without inventing additional tuning details.

## Statistical Tests

Use a table with a comparison column and a `p-value` column. Optional `Metric`
and `Test` columns are preserved.

```markdown
## Statistical Testing

| Comparison | Metric | Test | p-value |
|---|---|---|---:|
| Hyper-ProtoSurv vs ProtoSurv | C-index | Wilcoxon signed-rank | 0.018 |
```

The analyzer stores the comparison, metric, test name, exact p-value text, and
whether the result is significant at `alpha=0.05`. The paper agent will not
invent p-values, confidence intervals, or statistical tests that are not present
in the supplied results.

## Result Provenance

Use a provenance table to tie the paper-facing numbers to source artifacts.

```markdown
## Result Provenance

| Artifact | Path | SHA256 | Description |
|---|---|---|---|
| Paper result values | logs/tcga_values.csv | 64_HEX_DIGEST | method,dataset,metric,value rows |
| Fold-level result CSV | logs/tcga_folds.csv | 64_HEX_DIGEST | seed=2026; folds=0..4 |
| Evaluation log | logs/tcga_eval.log | 64_HEX_DIGEST | command=python eval.py; commit=abc123 |
| Experiment tracker export | wandb://entity/project/run-id | - | final metrics snapshot |
```

The validator records how many provenance entries were found, whether local
paths exist, each local file's byte size and SHA-256 digest, and whether
seed/fold identifiers are visible in the table. Missing local paths and checksum
mismatches are treated as provenance errors.

For artifact consistency checks, provide at least one local CSV with long-form
result rows. Main result and ablation rows use `method` or `variant` labels:

```csv
method,dataset,metric,value
ProtoSurv baseline,BLCA,C-index,0.646
Hyper-ProtoSurv ours,BLCA,C-index,0.671
ProtoSurv baseline,BRCA,C-index,0.669
Hyper-ProtoSurv ours,BRCA,C-index,0.691
Hyper-ProtoSurv ours,Average,C-index,0.681
w/o reconstruction loss,Average,C-index,0.665
```

The validator matches method, dataset, and metric labels, then compares numeric
values with a tolerance of `0.001` to allow normal paper-table rounding.
Sensitivity rows use `parameter` and `parameter_value`:

```csv
parameter,parameter_value,dataset,metric,value
lambda_rec,0.5,Average,C-index,0.676
lambda_rec,1.0,Average,C-index,0.681
```

Statistical-test rows use `comparison`, optional `metric`, optional `test`, and
`p_value`:

```csv
comparison,metric,test,p_value
Hyper-ProtoSurv vs ProtoSurv,C-index,Wilcoxon signed-rank,0.018
```

Common wide CSV tables are also accepted and expanded internally. Main results
and ablations can use one method-like label column plus metric columns:

```csv
method,BLCA C-index,BRCA C-index
ProtoSurv baseline,0.646,0.669
Hyper-ProtoSurv ours,0.671,0.691
```

```csv
variant,Average C-index
Hyper-ProtoSurv ours,0.681
w/o reconstruction loss,0.665
```

Sensitivity sweeps can use the parameter name as the first column:

```csv
lambda_rec,Average C-index
0.5,0.676
1.0,0.681
```

If the CSV contains repeated rows for the same method, dataset, and metric, such
as one row per fold, the validator compares the paper value against the mean of
those rows and records the fold count and sample standard deviation.

```csv
method,dataset,metric,fold,seed,value
ProtoSurv baseline,BLCA,C-index,0,2026,0.640
ProtoSurv baseline,BLCA,C-index,1,2026,0.652
Hyper-ProtoSurv ours,BLCA,C-index,0,2026,0.660
Hyper-ProtoSurv ours,BLCA,C-index,1,2026,0.682
```
