# CPU smoke run (pipeline check, not the real result)

U-Net base 16 at 384x216, 1500 steps x batch 8 (~12k scenes, ~45 min on 4 CPU cores); pair CNN 1500 steps x batch 32. 100 test scenes. The Colab run (base 32, 512x288, 8000 steps) replaces this.

## Level 1: boxes

| method | P@0.5 | R@0.5 | F1@0.5 | P@0.75 | R@0.75 |
|---|---|---|---|---|---|
| unet | 0.860 | 0.754 | 0.804 | 0.796 | 0.697 |
| contours | 0.310 | 0.659 | 0.421 | 0.257 | 0.546 |

Recall@0.5 by style (a box can have several styles):

| style | n | unet | contours |
|---|---|---|---|
| ellipse | 203 | 0.690 | 0.621 |
| fill_only | 176 | 0.739 | 0.557 |
| rounded | 170 | 0.753 | 0.700 |
| plain | 152 | 0.803 | 0.704 |
| shadow | 151 | 0.834 | 0.675 |
| hand | 125 | 0.816 | 0.632 |
| gradient | 92 | 0.728 | 0.783 |
| outline_only | 87 | 0.828 | 0.540 |
| glow | 57 | 0.754 | 0.684 |
| dashed | 37 | 0.622 | 0.541 |
| hatch | 32 | 0.812 | 0.719 |
| double | 26 | 0.846 | 0.808 |
| sticky | 23 | 1.000 | 0.391 |

## Level 2: groups

| boxes used | pair accuracy | multi-box groups exact |
|---|---|---|
| gt_boxes | 0.983 (903 pairs) | 0.953 (127 groups) |
| unet_boxes | 0.982 (492 pairs) | 0.925 (40 groups) |
