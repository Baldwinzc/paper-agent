# Real Experiment Results

Replace every TODO with real trained-model outputs before using this file for a paper draft.
Do not put cohort metadata, mock numbers, or synthetic pipeline checks in this file.

## Main Results

Metric: C-index. Higher is better.

| Method | BLCA C-index | BRCA C-index | LGG C-index | LUAD C-index | UCEC C-index |
|---|---:|---:|---:|---:|---:|
| ProtoSurv baseline | TODO | TODO | TODO | TODO | TODO |
| Hyper-ProtoSurv ours | TODO | TODO | TODO | TODO | TODO |

## Ablation Study

Metric: Average C-index. Higher is better.

| Variant | Average C-index |
|---|---:|
| Hyper-ProtoSurv ours | TODO |
| w/o key component 1 | TODO |
| w/o key component 2 | TODO |

## Sensitivity Analysis

Metric: Average C-index. Higher is better.

| lambda_rec | Average C-index |
|---:|---:|
| 0.1 | TODO |
| 0.5 | TODO |
| 1.0 | TODO |
| 2.0 | TODO |

## Statistical Testing

| Comparison | Metric | Test | p-value |
|---|---|---|---:|
| Hyper-ProtoSurv ours vs ProtoSurv baseline | C-index | Wilcoxon signed-rank | TODO |
