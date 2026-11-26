# Trace log

A dated record of progress, problems we hit, and how we solved them.
Newest entries at the top. Format: **Done** / **Issues & fixes** / **Next**.

---

## 2026-11-27 — Using box relations: shapes found behind detected text

**Done**
- Cooper asked whether the model understands box relations (text usually sits on a shape).
  It does not: YOLO predicts every box on its own. On the 20 real slides (R=0.5 base),
  34 of 98 shape-with-text pairs had **only the text found**. Shapes with no text inside
  are almost never found (1/35).
- `detector/containers.py` (`find_cards`): for each detected text box, grow the region of
  the colour just outside it and keep it as a `shape` (conf 0.30) if it contains the text,
  is card-sized, mostly filled and has a colour edge. Synthetic check with true text boxes
  as seeds: finds 91% of text-holding shapes, and 92% of what it adds is correct.
  `evalpred --cards` / `predict --cards` use it.
- Real slides, P / R / F1 at conf 0.25:

| model (test slides) | | mAP@0.5:0.95 | P | R | F1 | shape mAP@0.5:0.95 | shape recall | only-text pairs |
|---|---|---|---|---|---|---|---|---|
| R=0.5 base (20) | raw | 0.333 | 0.818 | 0.446 | 0.578 | 0.206 | 0.24 | 34 |
| | + cards | **0.347** | 0.821 | **0.480** | **0.605** | **0.275** | **0.36** | 21 |
| fold 0 mix (10) | raw | 0.449 | 0.704 | 0.650 | 0.676 | 0.344 | 0.42 | 21 |
| | + cards | 0.455 | 0.701 | 0.678 | 0.689 | 0.375 | 0.52 | 16 |
| fold 1 mix (10) | raw | 0.304 | 0.555 | 0.555 | 0.555 | 0.287 | 0.40 | 12 |
| | + cards | 0.309 | 0.560 | 0.579 | 0.569 | 0.310 | 0.50 | 6 |

- Recall goes up with no loss in precision: 13 of the 15 cards it added on the base model
  are correct (IoU 0.78-0.98). With `--post` as well, F1 is best (0.607 / 0.705 / 0.585).

**Issues & fixes**
- *Crash on Cooper's laptop*: `cv2.imread` cannot open Chinese file names on Windows.
  **Fix:** `cv2.imdecode(np.fromfile(...))`.
- *Wrong cards (2)*: a light gradient with a circuit pattern taken as a 37%-of-slide card, and
  a striped block that was really an icon.
- *Missed cards*: outline-only cards whose fill equals the background (white boxes on grid
  paper), since step 2 skips them by design. Also cards whose text YOLO missed, which leaves
  no seed. Possible next step: an outline mode (edges + rectangular contours around text).

---

## 2026-11-27 — Low recall: diagnosis and inference-time fixes

**Done**
- Ranked the usual fixes for high precision / low recall: inference (lower conf, larger
  imgsz, TTA), data (more real labels, pseudo-labels, copy-paste / closer synthetic
  styles), and model (bigger model, unfreeze more, P2 head). Tried the inference ones first,
  since they cost no training. R=0.5 base, 20 slides; P/R/F1 at conf 0.25:

| setting | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 |
|---|---|---|---|---|---|
| **1280** | **0.460** | **0.333** | 0.818 | 0.446 | **0.578** |
| 1600 | 0.357 | 0.223 | 0.642 | 0.398 | 0.491 |
| 1920 | 0.298 | 0.198 | 0.580 | 0.380 | 0.459 |
| 1280 + TTA | 0.423 | 0.294 | 0.713 | 0.474 | 0.570 |
| 1600 + TTA | 0.464 | 0.332 | 0.634 | 0.495 | 0.556 |

- None helps. Larger inputs hurt (title 0.554 -> 0.117 at 1920), because the model was
  trained at 1280. TTA buys a little recall but loses more precision, also on the
  fine-tuned models (fold 1 F1 0.555 -> 0.444). Lowering conf from 0.25 to 0.05 only moves
  recall 0.446 -> 0.495. **Keep 1280, no TTA, conf 0.25.**
- **Why boxes are missed** (`evalpred`, base, conf 0.25): of 218 misses, 194 have **no box
  near them at all**, 20 have the wrong class and 3 are loose. Recall by class: title 0.71,
  text 0.60, shape 0.24, picture 0.08, icon 0.34. Recall by size: small 0.37, medium 0.43,
  large 0.51, so size is not the cause.
- After few-shot (mix), recall is fold 0 0.67 / fold 1 0.56, and text recall 0.90 / 0.68.
  Shape stays at ~0.40: most shape misses are still "nothing there".

**Conclusion**
- The model has not learned these styles. Inference tricks cannot recover them; only data can.
  Real labels already work for text. Shapes (sticky notes, hand-drawn frames, neon outlines,
  highlighter strokes) need many more examples, real or synthetic.
- Pseudo-labelling can only add boxes the model already finds at some confidence. Lowering
  conf from 0.25 to 0.05 recovers only ~20 of the 218 missed boxes, so its upside here is limited.

---

## 2026-11-27 — Overlapping boxes: post-processing tested

**Done**
- Cooper saw many overlapping boxes and suggested keeping only the most confident box when
  IoU is high. Checked where overlaps come from on the 20 labelled slides (R=0.5, conf >= 0.25):
  same-class duplicates 0 (YOLO's per-class NMS already removes them). Cross-class pairs
  with IoU > 0.5: 13, mostly text on a card. The hand labels have 27 such pairs, so most are correct.
- Class-agnostic NMS **hurts**: mAP@0.5 0.454 -> 0.433 (it deletes cards and their text).
- `detector/postprocess.py`: merge only within {title, text} and within {shape, picture,
  icon}, plus drop a text box lying inside another text box. `detector/evalpred.py`
  scores saved predictions with and without it; `predict.py --post` applies it.

| model (test slides) | | mAP@0.5:0.95 | P @0.25 | R @0.25 | F1 @0.25 | overlapping pairs |
|---|---|---|---|---|---|---|
| R=0.5 base (20) | raw | 0.333 | 0.818 | 0.446 | 0.578 | 4 |
| | post | 0.327 | 0.833 | 0.444 | 0.579 | 1 |
| fold 0 mix (10) | raw | 0.449 | 0.704 | 0.650 | 0.676 | 5 |
| | post | 0.443 | 0.747 | 0.645 | 0.692 | 0 |
| fold 1 mix (10) | raw | 0.304 | 0.555 | 0.555 | 0.555 | 7 |
| | post | 0.290 | 0.581 | 0.550 | 0.565 | 0 |

- Post-processing removes nearly all remaining overlaps and raises F1 a little (more for the
  fine-tuned models), but costs 0.006-0.014 mAP. Other thresholds made no difference (< 0.001).
  **Decision:** keep it opt-in (`--post`) for the boxes -> pptx step. Report mAP without it.

**Issues & fixes**
- *Title boxes lost in title/text merges* (fold 1 title 0.430 -> 0.371): the more confident
  `text` box wins over the `title`. Possible fix: prefer `title` when merging.
- The scorer uses predict's single-label NMS, so its raw mAP differs slightly from YOLO's
  val (0.333 vs 0.317). Only compare raw vs post within it.

**Next**
- Recall, not overlap, is the bottleneck (0.45 base, 0.55-0.65 fine-tuned).

---

## 2026-11-27 — Few-shot results

**Done**
- `detector/fewshot.py` finished on Cooper's laptop: 2 folds by deck, ~10 train /
  10 test slides each, 1 h 19 min on CPU. Mean over folds:

| model | mAP@0.5 | mAP@0.5:0.95 | P | R | title | text | shape | picture | icon |
|---|---|---|---|---|---|---|---|---|---|
| base (R=0.5, no fine-tune) | 0.454 | 0.324 | 0.578 | 0.429 | 0.543 | 0.510 | 0.226 | 0.071 | 0.269 |
| real (real slides only) | 0.481 | 0.353 | 0.734 | 0.451 | 0.629 | 0.522 | 0.317 | 0.054 | 0.245 |
| **mix** (real + synthetic) | **0.497** | **0.370** | **0.810** | 0.451 | 0.590 | **0.569** | 0.316 | 0.058 | **0.317** |

  Per fold mAP@0.5:0.95: fold 0 base 0.421 / real 0.415 / mix 0.434,
  fold 1 base 0.226 / real 0.292 / mix 0.306.
- **mix is best and beats base on both folds.** The gain is in **precision**
  (0.58 -> 0.81), not recall (0.43 -> 0.45). Ten real slides teach the model what is
  *not* an element, but not where the elements it misses are.
- real alone slightly drops on fold 0 and loses icon AP, so mixing in synthetic slides
  is what keeps the old skills.

**Issues & fixes**
- *Folds differ more than models* (base 0.42 vs 0.23 mAP@0.5:0.95). Fold 1 holds the
  dark, textured, illustrated decks (cyberpunk, funk, Edgerunners). Treat gaps under
  ~0.02 as ties.
- Hyperparameters (freeze 10, lr 5e-4, 20 epochs) were fixed in advance, not tuned:
  there is no validation split among 20 slides. Tuning needs more labels or nested CV.

**Next**
- Recall is the bottleneck. Options: more real labels for training, a lower confidence
  threshold for the pptx step, or generator styles that match the missed elements.

---

## 2026-11-26 (night) — Labelling rules settled, few-shot experiment started

**Done**
- **Labelling workflow.** Started with CVAT (pre-labels as a zip), but uploading,
  exporting and the class setup took too long. Replaced it with `tools/labeler.py`:
  a local page (Python standard library only) that opens each slide with the
  R=0.5 model's boxes already drawn. Fixing a slide takes key presses (`1`-`5` class,
  `Del`, `D` save + next) instead of drawing every box. 20 slides, 392 boxes took
  Cooper about 1.5 h (21:05-22:30), including settling the rules below.
- **Choosing which slides to label.** `--sample 20` takes slides round-robin over the 19
  decks with a fixed seed, so the test set covers every deck and is reproducible.
- **Ranking models without labels.** Before any labels existed, `detector/compare_preds.py`
  ran all 5 models on the 237 slides and measured how well they agree with each other
  (F1 0.44-0.63) plus side-by-side images. The low agreement was the first sign of a
  large sim-to-real gap, and the reason to hand-label at all.
- **Labelling rules decided while labelling** (full text in `docs/labelling.md`):
  1. Text first: every text block must be right, since it becomes editable text.
  2. One test for `shape`: a basic outline (rectangle, circle, bar, line, arrow) filled
     mostly in one flat colour. Glow, transparency, shadows, torn edges and sketchy
     hand-drawn outlines still count (neon rings, sticky notes, highlighter strokes,
     hand-drawn frames). Text on top does not change the class.
  3. Hatching, shading or several colours inside fail that test: small -> `icon`
     (paper clip, brush-stroke arrow, lightbulb), large -> `picture` (photo, record
     illustration). Nothing inside a picture is boxed, including words printed on it.
  4. No box for backdrops and textures (grunge wall, open book behind notes, grid
     paper, circuit pattern) or for regions too layered to rebuild.
  5. Anything you can type is `text`, even large or stylised (`A → aV*`, `⊢`).
- **Cooper chose few-shot over generator changes** to close the gap.
  `detector/fewshot.py` splits the 20 slides into 2 folds **by deck**. Each fold
  fine-tunes the R=0.5 model on the other ~10 slides and tests on its own. It compares
  `base` (no fine-tune), `real` (real slides only) and `mix` (real + an equal number of
  synthetic slides, against forgetting) on the same held-out slides. Settings: backbone
  frozen, AdamW lr 5e-4, 20 epochs, real slides repeated to ~40 per epoch.
  First number: fold 0 base = mAP@0.5 0.516 / mAP@0.5:0.95 0.421. Run in progress.

**Issues & fixes**
- *Hand-drawn frames were first labelled `icon`.* `icon` in the synthetic data means
  small square pictograms, so the mismatch would count as detector error. **Fix:**
  rule 2 above. A check lists icons wider than 10% of the slide: 28 found, 3 clearly
  wrong (Edgerunners_05, OS_12), sent back for fixing.
- *An earlier rule ("a drawn diagram is one picture") hid the diagram's text*,
  against text-first. **Fix:** diagrams made of basic shapes are labelled part by part;
  only painted illustrations are one `picture`.
- *Colab Pro has no background execution.* The runtime survived Chrome sleeping, but
  the page lost its connection, and reloading it discarded unsaved outputs. Later,
  reconnecting found no free T4 ("no backend with GPU available"). **Fix:** run long
  jobs from the Colab terminal so they don't depend on the page. Smaller jobs
  (evaluation, few-shot) run on Cooper's laptop.
- *The laptop had the slides but not the repo.* **Fix:** cloned it next to the decks,
  rebuilt the 237 images with `tools/extract_slides.py` (same names as on Colab), and
  copied predictions and weights from Google Drive for desktop (`G:\我的雲端硬碟`).
  Packages go in a `.venv` in the clone, so the global Python stays unchanged.
- *The laptop venv has CPU-only torch* (about 45 s per epoch for 10 slides at 1280).
  It has an RTX 3050 4 GB, so installing CUDA torch would speed up later runs.

**Next**
- Finish the few-shot run and report base vs real vs mix.
- Fix the 3 oversized icon labels and re-run the real-slide evaluation.
- Label more slides from unseen decks if the few-shot gain is promising, since ~10
  test slides per fold is noisy.

---

## 2026-11-26 — First real-slide mAP (20 hand-labelled NotebookLM slides)

**Done**
- Cooper labelled 20 slides spread over the 19 decks with `tools/labeler.py`
  (392 boxes: title 17, text 189, shape 106, picture 12, icon 68).
- Evaluated the 5 ratio-sweep models (last.pt, 1280 px) on them, on CPU on Cooper's PC.

| R | real mAP@0.5 | real mAP@0.5:0.95 | text mAP@0.5:0.95 | shape | icon |
|---|---|---|---|---|---|
| 0.00 | 0.251 | 0.160 | 0.276 | 0.077 | 0.079 |
| 0.20 | 0.286 | 0.199 | 0.370 | 0.114 | 0.103 |
| 0.50 | **0.454** | **0.317** | **0.496** | **0.212** | **0.278** |
| 0.80 | 0.259 | 0.170 | 0.317 | 0.099 | 0.150 |
| 1.00 | 0.227 | 0.144 | 0.249 | 0.050 | 0.184 |

- R=0.5 is best on every class. Synthetic scores are ~0.95+, so the sim-to-real gap is large.
- Main loss is **recall**: text R=0.62 at best, shape 0.07-0.23. Precision is 0.5-0.8.

**Issues & fixes**
- 28 icon boxes are wider than 10% of the slide. At least 3 (Edgerunners_05, OS_12) look
  like picture/shape. To review, then re-run the evaluation.
- Only 12 pictures in the set, so picture numbers are noise.

**Next**
- Make synthetic slides look more like NotebookLM: hand-drawn and glowing shapes, sticky
  notes, highlighter strokes, dark textured backgrounds, illustrated objects.
- Retrain at R=0.5 and compare on these 20 slides.

---

## 2026-11-26 — Sweep trained on Colab, real slides checked, labelling tool

**Done**
- Ran the full ratio sweep on Colab (yolo11n, 1280 px, 30 epochs, 2,500 slides per pool).
  mAP@0.5:0.95 template / random: R=0 0.960 / 0.756, R=0.2 0.960 / 0.962,
  R=0.5 0.955 / 0.970, R=0.8 0.952 / 0.975, R=1 0.701 / 0.975.
- Ran all 5 models on 237 unlabelled NotebookLM slides (19 decks).
  `detector/compare_preds.py` reports boxes per slide, confidence and how much the
  models agree with each other, plus side-by-side images.
- `tools/labeler.py`: a local box editor that starts from predictions and writes YOLO txt
  for `real_labels.py build`. It replaces CVAT as the default in `docs/labelling.md`.
- Labelling rules for NotebookLM slides: a drawn diagram is one `picture`; a region too
  complex to rebuild gets no box (background); text blocks matter most.

**Issues & fixes**
- *Real slides look very different from synthetic ones*: only 7-11 boxes per slide, and
  the models agree on just 44-63% of boxes (F1), while all score above 0.95 mAP on synthetic data.
  Small text inside drawn diagrams is mostly missed. **Fix:** hand-label 20 slides for a
  real mAP. The diagram rule above settles what counts as a box.
- *Ultralytics nested relative run paths* (`runs/detect/runs/...`). **Fix:** absolute `project` paths.

**Next**
- Label 20 slides, build `data/real/yolo`, evaluate all 5 models on real slides.

---

## 2026-11-19 — Random-ratio experiment set up

**Done**
- Plan in `docs/experiments.md`: train the same model with R ∈ {0, 0.2, 0.5, 0.8, 1}
  share of random-layout slides and test on template, random and real slides.
- `detector/mix.py` builds every R from two generated pools (templates only, random
  only), so all sets have the same size and only the mix changes.
- `detector/sweep.py` writes the results table and the mAP-vs-R plot.
- Notebook section E1–E6 (+ Phase 2) replaces the earlier three-set cells. It skips
  finished runs and resumes interrupted ones, so it survives Colab disconnects.
- Tested the whole section here on tiny pools (60 slides, 1 epoch, CPU): mixing,
  training, skip-on-rerun, evaluation, table and plot all work.

**Issues & fixes**
- *`--random-ratio` flips a coin per slide*, so a generated set is only roughly R and
  each R would need its own render. **Fix:** mix two pools by file list (exact R, one
  render pass).
- *Pools with the same seed share themes slide by slide* (theme is drawn first from the
  per-slide seed). **Fix:** seed 0 for templates, seed 1 for random.
- *Early stopping and best.pt use each set's own synthetic val*, which differs by R.
  **Fix:** `--patience 0` and evaluate `last.pt`.

**Next**
- Generate pools and run the sweep on Colab; add real results once labels exist.

---

## 2026-11-18 (afternoon) — More icons, random-layout option

**Done**
- New `iconlist` layout (rows of icon + heading + text, common in AI decks).
- Icons added to stats (50% of slides), process steps (40%, replacing the
  number) and bullet slides (decorative icon in free space); cards now have
  icons 85% of the time instead of 60%.

- `--random-ratio R`: a share R of slides use a new `random` layout that scatters
  every element type at random positions without overlaps (domain
  randomisation). Default 0, which reproduces earlier datasets exactly.
  Planned experiment: train on templates only vs random only vs 80/20 mix and
  compare scores on the real NotebookLM slides. Samples:
  `docs/img/random_layout_samples.png`.

**Issues & fixes**
- *Too few icons to measure icon accuracy* (reported by the YOLO thread): in 300
  slides icons were 81 of 2201 boxes (3.7%), only 6 in the test split.
  **Fix:** the changes above. Same 300-slide check now gives 329 of 2633 boxes
  (12.5%), 30 in the test split.
- *Big numbers like "$157M" split across two lines.* The text-fit check only
  counted wrapped lines, so a single word wider than its box "fit", and
  PowerPoint then broke it mid-word. **Fix:** a line wider than the box now
  counts as not fitting, so the font shrinks or the text is dropped.

---

## 2026-11-18 — Element detector (YOLO): training and evaluation code

**Done**
- `detector/`: `train.py` (slide-tuned YOLO training), `augment.py` (sim-to-real
  augmentation), `evaluate.py` (per-class mAP@0.5 and mAP@0.5:0.95 table as .md/.json),
  `predict.py` (drawn previews + JSON boxes for the later pipeline steps).
- `notebooks/train_yolo.ipynb`: Colab T4 training, resume, synthetic and real evaluation,
  and the no-augmentation ablation.
- Real test set labelling: `docs/labelling.md` (rules that match the generator, CVAT
  workflow) and `tools/real_labels.py` (pre-labels from the detector, then builds
  `data/real/yolo/` from the CVAT export).
- Smoke test on CPU: 300 synthetic slides, yolo11n, imgsz 640, 3 epochs →
  mAP@0.5 0.83 / mAP@0.5:0.95 0.64 on the 30-slide test split. This only shows the
  pipeline works end to end; real numbers come from the Colab run.
- `detector/compare.py` and a notebook section for the training-set experiment:
  templates only vs random layouts only (`--random-ratio 1`) vs 80/20 mix (`0.2`),
  same size and settings, compared on the real slides in one table.
- Round-trip test of the labelling tool: synthetic labels exported with a shuffled
  class order and rebuilt are byte-identical to the originals.

**Issues & fixes**
- *data.yaml has an absolute path*, so it breaks when the dataset moves to Drive or
  `/content`. **Fix:** train/evaluate write `data.local.yaml` next to it with the
  current folder, and check the class names match.
- *Runs saved to `runs/detect/runs/detect/...`*: Ultralytics prefixes a relative
  `project` with its own runs folder. **Fix:** pass an absolute path.
- *Flips and rotation make no sense for slides* (text has a direction, layout is upright).
  **Fix:** fliplr/degrees/shear off, mosaic 0.2 and closed for the last 15 epochs,
  stronger hue/saturation jitter for varied themes.
- *Synthetic renders are too clean* compared with NotebookLM images (upscaled,
  JPEG). **Fix:** extra Albumentations: JPEG quality 35–95, downscale, blur, noise,
  brightness/contrast. `--no-sim2real` turns it off for the ablation.
- *Labelling tools order classes differently.* **Fix:** `real_labels.py build` remaps
  class ids by name and stops on unknown names.
- *Icons are rare*: 81 of 2201 boxes (3.7%) in a 300-slide sample, so icon AP will be
  noisy. Flagged to the generator side.
- *Colab RAM*: `--cache ram` for 5k slides at 1280 needs ~11 GB, more than free
  Colab has. **Fix:** not cached; dataset is copied from Drive to local disk instead.

**Next**
- Generate 5k slides, train yolo11s at 1280 on Colab T4, report synthetic test mAP.
- Label the 30–50 NotebookLM slides in CVAT, report real mAP, then run the ablation.

---

## 2026-11-18 — Topic chosen, synthetic slide generator v1

**Done**
- Chose the project after comparing several ideas (sports ball tracking, video
  style transfer, virtual try-on, soccer analytics). Picked slide image → editable
  PPTX: original, unlimited free labelled data, fits free Colab + RTX 3050.
- Wrote `PLAN.md` (4-week plan, due 2026-12-15).
- Built `slidegen/`, the synthetic slide generator:
  - 8 layout templates: title, bullets, text + picture, cards, hero, gallery,
    stats, process.
  - Random themes (light/dark, accent colours, fonts), English and Traditional
    Chinese text, emoji icons, procedural pictures and matplotlib charts.
  - Renders with LibreOffice → PDF → PNG at 1280×720.
  - Outputs YOLO labels, full JSON metadata (text, font, size, colour, lines),
    per-pixel instance masks, and a text-free "clean" version of every slide.
- Tested on 48 slides: labels line up with rendered elements (see
  `docs/img/synthetic_samples.png`). Speed ≈ 1.1 slides/s with 2 workers.

**Issues & fixes**
- *Tight text boxes.* The box we place a text frame in is bigger than the text
  LibreOffice actually draws, so frame boxes would be loose labels. **Fix:** render
  a third "id-map" deck where every element is a flat unique colour on black,
  rendered with anti-aliasing off, so every pixel maps exactly to one element.
  Tight boxes, text-line boxes, and segmentation masks all come from it.
- *Wrong backgrounds on some slides (dark background behind dark text).* Picture
  files were named using Python's `id()` of a builder object; after an object was
  freed, the next slide could reuse the same id and overwrite the earlier slide's
  images. **Fix:** name asset files with `uuid4`.
- *Rounded cards looked unnatural* (python-pptx default corner radius is huge).
  **Fix:** random radius between 3% and 15%.
- *Cards much taller than their text.* **Fix:** in single-row card layouts,
  shrink all cards to the tallest content 60% of the time.

- *NotebookLM only exports PDF or PPTX.* The PPTX holds each slide as one
  full-size picture. **Fix:** `tools/extract_slides.py` pulls those pictures out at
  original resolution and starts a tagging CSV.

**Next**
- Push the repo and run the Colab setup notebook end to end.
- Generate ~5k slides to Google Drive.
- Collect 30–50 NotebookLM slides for the real test set.
