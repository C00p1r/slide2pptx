# Labelling the real NotebookLM test slides

The real slides are the **out-of-distribution test set**. They are never used for
training (except the optional few-shot experiment in PLAN.md), so their labels must
follow the same rules as the synthetic labels, or the mAP numbers will measure
label disagreement instead of detector quality.

## Tool: `tools/labeler.py` (local, no install, no account)
A small box editor that runs on your computer (Python standard library only). Boxes start
from a detector's predictions, so you fix them instead of drawing everything.

1. Collect slides: `python tools/extract_slides.py path/to/pptx_folder --out data/real/images`
2. Predictions to start from (any run, e.g. the R=0.5 model from the ratio sweep):
   ```bash
   python -m detector.predict runs/ratio/r0.50/weights/last.pt data/real/images --out results/pred_llm/r0.50
   ```
3. Label 20 slides spread over all decks (same 20 every time with the same seed):
   ```bash
   python tools/labeler.py --pred results/pred_llm/r0.50 --sample 20
   # opens http://localhost:8765, saves data/real/labels/<slide>.txt (YOLO) + classes.txt
   ```
   Drag to draw a box, click to select, drag the corners to resize, `1`-`5` set the class,
   `Del` deletes, `D` saves and goes to the next slide. A dashed box is a low-confidence
   prediction. Stop any time: a slide is done once its .txt exists, and the tool resumes there.
4. Build the test set and evaluate:
   ```bash
   python tools/real_labels.py build data/real/labels
   # -> data/real/yolo/{images,labels}/test + data.yaml, prints box counts per class
   python -m detector.evaluate runs/ratio/r0.50/weights/last.pt --data data/real/yolo/data.yaml --tag real
   ```

**Alternative: CVAT** ([app.cvat.ai](https://app.cvat.ai), free, browser). `python tools/real_labels.py prelabel <weights>`
writes `data/real/cvat_prelabels.zip` (upload with *Actions -> Upload annotations -> YOLO 1.1*); export
the corrected task as *YOLO 1.1* and pass that zip to `real_labels.py build`.

**Bias warning:** pre-labels can nudge you toward the model's mistakes. Check each
slide for *missing* boxes too, not only wrong ones, and redraw loose boxes tightly.

## Rules (match the synthetic generator)
| Class | Box it | Do not box |
|---|---|---|
| `title` | The main slide title, tight around the ink, all its lines in one box | Section labels or headings inside the body (those are `text`) |
| `text` | Every other block of text: subtitle, heading, paragraph, bullet list, caption, big number, label inside a shape. **One box per text block** (a bullet list is one box; a heading above a paragraph is a separate box when there is a visible gap) | Text that is part of a picture (words inside a chart or photo) |
| `shape` | Flat vector shapes: cards, panels, rounded boxes, arrows/chevrons, circles, accent bars and lines, footer bands | The slide background, gradients covering the whole slide |
| `picture` | Photos, illustrations, charts and diagrams drawn as one image | Small icons |
| `icon` | Small pictograms or emoji, roughly square, usually < 10% of slide width | Large illustrations (those are `picture`) |

- Boxes are tight around what is **visible** (ink for text, the shape's edges).
- Overlaps are expected: a card is a `shape`, and the text on it is separately `text`.
- When an element is truly ambiguous (e.g. an illustration that could be an icon), pick
  one, and note the slide in `data/real/real_slides.csv` (`hard_case` column) so we can
  discuss it in the report.

### NotebookLM slides (decided 2026-11-26)
- **Text first.** Getting every text block right matters most: these are what become
  editable text boxes in the .pptx. Check the slide for missed text before anything else.
- **Could PowerPoint rebuild it with a basic shape** (rectangle, rounded box, circle,
  line, arrow) plus a fill or border? Then it is `shape`, even with glow, gradient or
  transparency (e.g. neon rings, a semi-transparent card). A diagram built from such
  shapes plus text is labelled part by part: each shape `shape`, each label `text`.
- A **painted or drawn illustration** (shading, texture, 3D, hand-drawn) counts as
  **one `picture`** box. Do not box the parts inside it or its small words.
  A small single drawn object (a brush-stroke arrow, a stack of boxes) is `icon`.
- When a region is **too complex to rebuild** (many stacked or overlapping layers), treat it
  as background: **no box at all**. It stays in the background image of the .pptx.
- Text that sits on a sticky note or card *outside* a diagram is still `text`, and the
  note or card is a `shape`.
- **Formula or big stylised text** (e.g. `A → aV*`): if you could type it on a keyboard,
  it is one `text` box. Hand-drawn marks on it (a red circle, an underline swoosh) get no box.
- **Clip-art objects** (a thumbs-up, a scanner, a lightbulb) are `icon` when they are a
  single small object, roughly square and under about 10% of the slide width. A scene
  or a drawing with several parts is `picture`.
