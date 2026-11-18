# Trace log

A dated record of progress, problems we hit, and how we solved them.
Newest entries at the top. Format: **Done** / **Issues & fixes** / **Next**.

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

**Next**
- Push the repo and run the Colab setup notebook end to end.
- Generate ~5k slides to Google Drive.
- Collect 30–50 NotebookLM slides for the real test set.
