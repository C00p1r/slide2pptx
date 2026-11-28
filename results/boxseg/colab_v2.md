# Colab T4 run, v2 (box/box boundary weight 5)

U-Net base 32 at 512x288, 8000 steps x batch 16 (128k generated scenes); pair CNN 4000 steps x batch 64.
Evaluated on 300 synthetic test scenes (seed 12345, never used in training).
The run was interrupted once at step 2500 (Colab runtime lost) and resumed from the checkpoint.

## Level 1: boxes

| method | P@0.5 | R@0.5 | F1@0.5 | P@0.75 | R@0.75 |
|---|---|---|---|---|---|
| unet | 0.871 | 0.901 | 0.885 | 0.794 | 0.821 |
| contours | 0.263 | 0.640 | 0.372 | 0.216 | 0.527 |

v1 (no boundary weight, stopped at step 3000, 100 scenes): P 0.823, R 0.763, F1 0.792.

Recall@0.5 by style (a box can have several styles):

| style | n | unet | contours |
|---|---|---|---|
| ellipse | 722 | 0.885 | 0.612 |
| fill_only | 587 | 0.906 | 0.560 |
| plain | 534 | 0.912 | 0.680 |
| shadow | 518 | 0.911 | 0.616 |
| rounded | 505 | 0.893 | 0.665 |
| hand | 469 | 0.908 | 0.627 |
| gradient | 259 | 0.892 | 0.625 |
| outline_only | 252 | 0.813 | 0.635 |
| glow | 216 | 0.921 | 0.620 |
| dashed | 129 | 0.891 | 0.496 |
| hatch | 121 | 0.876 | 0.628 |
| double | 113 | 0.903 | 0.708 |
| sticky | 92 | 1.000 | 0.489 |

## Level 2: groups

| boxes used | pair accuracy | multi-box groups exact |
|---|---|---|
| gt_boxes | 0.999 (3385 pairs) | 0.992 (533 groups) |
| unet_boxes | 0.988 (2432 pairs) | 0.933 (445 groups) |

## Ablation: boundary weight (CPU, base 16, 384x216, +400 steps from the same model, 120 scenes)

| | no weight | box/box boundary x5 |
|---|---|---|
| nested-box recall (header, inner, badge, icon circle, stacked) | 0.30 | 0.67 |
| top-level box recall | 0.96 | 0.94 |
| P / R / F1 | 0.88 / 0.74 / 0.80 | 0.81 / 0.85 / 0.83 |
