# Labelling the real NotebookLM test slides

The real slides are the **out-of-distribution test set**. They are never used for
training (except the optional few-shot experiment in PLAN.md), so their labels must
follow the same rules as the synthetic labels, or the mAP numbers will measure
label disagreement instead of detector quality.

## Tool: CVAT (free, browser, keeps your slides private)
Roboflow also works, but its free plan makes projects public. Label Studio works too;
any tool is fine as long as it exports YOLO txt with class names.

1. Collect slides: `python tools/extract_slides.py path/to/pptx_folder --out data/real/images`
2. Pre-label with a trained detector, so you correct boxes instead of drawing all of them:
   ```bash
   python tools/real_labels.py prelabel runs/detect/slides/weights/best.pt
   # -> data/real/cvat_prelabels.zip
   ```
3. On [app.cvat.ai](https://app.cvat.ai): create a project with the 5 labels
   `title, text, shape, picture, icon` (rectangle), create a task, upload every
   image in `data/real/images`.
4. In the task: **Actions → Upload annotations → YOLO 1.1**, choose `cvat_prelabels.zip`.
5. Fix every box (rules below). Budget about 2–3 min per slide; 40 slides ≈ 1.5–2 h.
6. **Actions → Export task dataset → YOLO 1.1** (no images needed). Then:
   ```bash
   python tools/real_labels.py build path/to/export.zip
   # -> data/real/yolo/{images,labels}/test + data.yaml, prints box counts per class
   python -m detector.evaluate runs/detect/slides/weights/best.pt \
       --data data/real/yolo/data.yaml --tag real
   ```

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
