# Trace log

A dated record of progress, problems we hit, and how we solved them.
Newest entries at the top. Format: **Done** / **Issues & fixes** / **Next**.

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
