# Experiment plan: how much random layout should the training data have?

## Question
The generator can make two kinds of slides: **template** slides (9 realistic layouts:
title, bullets, cards, process...) and **random** slides (`--random-ratio`): the same
element types scattered anywhere without overlap (domain randomisation).
If a share **R** of the training slides is random, how does R change detection
quality on real NotebookLM slides?

## Why it is interesting
- Templates teach the model what slides usually look like, but real slides may not
  match our 9 layouts: the model can learn "title is at the top" instead of what a
  title looks like.
- Random slides force it to judge each element by its appearance, but they look less
  like real slides.
- **Expectation:** a mix (R around 0.2–0.5) beats both ends on real slides; R = 1 may
  hurt `title` most, because position is a strong real cue it never sees.
  The result is useful either way: it is the sim-to-real story of the report.

## Design
| | |
|---|---|
| Variable | R ∈ {0, 0.2, 0.5, 0.8, 1.0} |
| Data | Two pools generated once: **TPL** (R = 0, seed 0) and **RAND** (R = 1, seed 1), 2,500 slides each = 2,000 train / 250 val / 250 test. `detector/mix.py` builds each R by taking 2,000 train slides as (1 − R) TPL + R RAND (val the same way). Every R has the same size, and the sets are nested, so only the mix changes. |
| Model | yolo11n, imgsz 1280, 30 epochs, batch 16, seed 0, sim-to-real augmentation on |
| Fairness | No early stopping (`--patience 0`); evaluate `last.pt`, so no model is picked using its own synthetic val set |
| Test sets | TPL test (250), RAND test (250), real NotebookLM slides (30–50, hand-labelled, `docs/labelling.md`) |
| Metrics | mAP@0.5 and mAP@0.5:0.95, overall and per class |
| Output | `results/detector/ratio_sweep.md` (tables) and `ratio_sweep.png` (mAP vs R, one line per test set) |

Why two pools instead of generating each R separately: one render pass instead of
five, exact ratios (`--random-ratio` flips a coin per slide), and the same template
slides appear across ratios, which removes noise from different random draws.

Why yolo11n: five runs must fit in free Colab time. The comparison between R values is what
matters, not the absolute numbers. Phase 2 checks that the winner still wins with the
final model.

## How to read the results
- **TPL test:** expected to drop as R grows (less in-distribution data).
- **RAND test:** expected to rise as R grows.
- **Real:** the answer to the question. Choose R by this line.
- **Noise:** the real set is small (~40 slides). Treat differences under about 0.02
  mAP@0.5:0.95 as a tie. If time allows, re-run R = 0 and the best R with `seed 1`
  to see how much results move between seeds.
- Look at per-class rows: title and icon are the most likely to react to R.

## Steps (all in `notebooks/train_yolo.ipynb`, section "Experiment")
| Step | Cell | Time (estimate, free Colab) |
|---|---|---|
| 1. Generate both pools into Drive | E2 | ~35 min CPU, once |
| 2. Build the five mixes | E3 | ~1 min, every session |
| 3. Train one model per R (skips finished runs, resumes interrupted ones) | E4 | ~30–45 min per R on a T4, ~3–4 h total: plan 2 sessions |
| 4. Evaluate on TPL, RAND (and real, once the CVAT export is on Drive) | E5 | ~1 min per model |
| 5. Table + plot | E6 | seconds |
| 6. Phase 2: retrain R = 0 and best R with yolo11s, 4,000 train slides, 40 epochs | last cell | ~3 h each |

The time per epoch is my estimate. Check the first epoch's time and adjust
`SWEEP_EPOCHS` if needed (keep it the same for every R).

Real labels are not needed to start: steps 1–5 run now on the synthetic test sets, and
E5 adds the real results later without retraining.

## Schedule
- **Week 2 (Nov 25 – Dec 1):** steps 1–5 on synthetic; label real slides in parallel.
- **By Dec 1:** real column filled in, best R chosen.
- **Week 3 (Dec 2 – 8):** Phase 2 runs; the best full-scale model becomes the demo model.
