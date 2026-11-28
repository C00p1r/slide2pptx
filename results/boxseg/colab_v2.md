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

## Held-out styles (the main generalisation test)
Model `seg_no_hand` was trained with the same settings but never saw hand-drawn outlines or sticky
notes. Both models are tested only on boxes that are hand-drawn or sticky (300 scenes, 561 boxes).

| model | R@0.5 | R@0.75 | hand R@0.5 | sticky R@0.5 |
|---|---|---|---|---|
| full (saw all styles) | 0.919 | 0.820 | 0.908 | 1.000 |
| **held out (never saw hand / sticky)** | **0.908** | **0.805** | **0.896** | **1.000** |
| contours | 0.614 | 0.480 | 0.627 | 0.489 |

Recall drops by about one point when the style is never seen, so the U-Net detects these
boxes from their closed shape, not from having learned the style.
Precision is not meaningful in this table: false positives are counted over all boxes in the
scene while true positives only count the hand/sticky ones.

## Real NotebookLM slides (first look, no labels yet)
`pred_real/` on Drive. Observations from a few slides:
- the inside map finds closed regions well on unseen real styles (neon HUD frames, sticky notes,
  table cells, gauges);
- boundaries between touching cells (tables, grids) are weak, so neighbouring cells often merge
  into one large box: the generator has no tables/grids of touching cells yet;
- illustrations and photos (spine drawing, gauge pictures) are also marked as regions, since
  the generator draws no pictures or icons yet.

## Real NotebookLM slides: the 20 hand-labelled test slides
`python -m boxseg.eval_real runs/boxseg/seg.pt data/real/yolo --size 768x432`. Labels are the
same 392 boxes used for the YOLO real-slide mAP. The U-Net has no classes, so this is
class-agnostic recall at IoU 0.5 per labelled class. "matches any label" is the share of predicted
boxes that match some labelled box (a rough precision).

| method | size | shape R (106) | picture R (12) | icon R (68) | shape+picture+icon R | matches any label |
|---|---|---|---|---|---|---|
| U-Net v2 | 512x288 | 0.528 | 0.500 | 0.250 | 0.409 | 0.335 of 233 |
| **U-Net v2** | **768x432** | **0.604** | **0.583** | 0.324 | **0.484** | 0.305 of 308 |
| contours | 768x432 | 0.500 | 0.250 | 0.471 | 0.462 | 0.061 of 2081 |
| YOLO R=0.5 (class-aware, for reference) | 1280 | 0.226 | 0.250 | 0.368 | | P 0.70 / 0.36 / 0.48 |

The YOLO row is its per-class recall from `results/detector/ratio_r0.50_real.md`; YOLO must also get
the class right, so it is a stricter test than class-agnostic matching. Text and titles are not
regions (recall 0.10 and 0.12, as intended).
