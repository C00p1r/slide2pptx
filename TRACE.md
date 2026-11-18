# Trace log

A dated record of progress, problems we hit, and how we solved them.
Newest entries at the top. Format: **Done** / **Issues & fixes** / **Next**.

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
