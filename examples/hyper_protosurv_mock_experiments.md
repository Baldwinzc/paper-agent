# Synthetic Mock Experiment Results for Hyper-ProtoSurv

This file is synthetic mock data for pipeline testing only. It is not a real
experimental result and must not be used as evidence in a submitted paper.

## Main Results: 5-fold Cross Validation

Metric: C-index, mean +/- standard deviation.

| Method | BLCA | BRCA | LGG | LUAD | UCEC | Average |
|---|---:|---:|---:|---:|---:|---:|
| DeepSurv-WSI | 0.612 +/- 0.031 | 0.641 +/- 0.027 | 0.682 +/- 0.024 | 0.603 +/- 0.029 | 0.628 +/- 0.026 | 0.633 |
| GraphSurv | 0.628 +/- 0.026 | 0.654 +/- 0.024 | 0.701 +/- 0.022 | 0.617 +/- 0.027 | 0.641 +/- 0.025 | 0.648 |
| ProtoSurv baseline | 0.646 +/- 0.024 | 0.669 +/- 0.021 | 0.724 +/- 0.019 | 0.636 +/- 0.025 | 0.658 +/- 0.023 | 0.667 |
| Hyper-ProtoSurv ours | 0.671 +/- 0.021 | 0.691 +/- 0.019 | 0.746 +/- 0.017 | 0.661 +/- 0.022 | 0.681 +/- 0.020 | 0.690 |

Metric: IBS, lower is better.

| Method | BLCA | BRCA | LGG | LUAD | UCEC | Average |
|---|---:|---:|---:|---:|---:|---:|
| ProtoSurv baseline | 0.184 | 0.171 | 0.152 | 0.192 | 0.177 | 0.175 |
| Hyper-ProtoSurv ours | 0.171 | 0.160 | 0.143 | 0.178 | 0.165 | 0.163 |

## Ablation Results

Metric: Average C-index across BLCA, BRCA, LGG, LUAD, UCEC.

| Variant | Average C-index | Delta vs Full |
|---|---:|---:|
| Full Hyper-ProtoSurv | 0.690 | 0.000 |
| w/o OT-driven adaptive hyperedges | 0.674 | -0.016 |
| w/o bidirectional hyperedge update | 0.678 | -0.012 |
| mean-pool fusion instead of cross-attention | 0.681 | -0.009 |
| w/o L_rec | 0.672 | -0.018 |

## Sensitivity Analysis

Metric: Average C-index.

| lambda_rec | Average C-index |
|---:|---:|
| 0.1 | 0.681 |
| 0.5 | 0.687 |
| 1.0 | 0.690 |
| 2.0 | 0.686 |
| 5.0 | 0.679 |

## Notes

- The baseline comparison rows are explicit in the main-results tables.
- These synthetic numbers are intentionally plausible but fabricated for testing.
- Replace this file with real result tables before using any generated paper text.

