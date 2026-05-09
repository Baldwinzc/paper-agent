# Experiment Results Format

`paper-agent` accepts Markdown, CSV-like text, or pasted notes, but the most reliable
format is a Markdown table with one method column and one numeric column per
dataset-metric pair.

Use real trained-model results only. Cohort metadata is useful for dataset
description, but it is not evidence for performance claims.

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
