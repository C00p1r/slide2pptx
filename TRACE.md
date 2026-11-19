# Trace log

A dated record of progress, problems we hit, and how we solved them.
Newest entries at the top. Format: **Done** / **Issues & fixes** / **Next**.

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
