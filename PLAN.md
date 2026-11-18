# Project plan: Slide image → editable PPTX

**Due:** 2026-12-15 — live demo, presentation video, and slides.

## Goal
AI tools such as NotebookLM export slides as flat images that can't be edited.
This project turns a slide **image** back into an **editable .pptx**: real text
boxes, shapes, and pictures in the right places, with the right font size and colour.

**Pitch:** "A CNN pipeline that rebuilds editable PowerPoint slides from images,
trained only on synthetic slides and tested on real AI-generated slides."

## Pipeline
```
slide image ─► element detector (YOLO)          ─► boxes: title / text / shape / picture / icon
            ─► OCR (PaddleOCR)                  ─► the words in each text box
            ─► font-attribute CNN (ours)        ─► font size, colour, bold
            ─► text removal (inpainting)        ─► clean background
            ─► python-pptx rebuilder            ─► editable .pptx
```

## Data
- **Synthetic (train / val / in-distribution test):** random slides generated with
  python-pptx and rendered by LibreOffice. Labels are exact and free: element
  boxes, text, font, size, colour, per-pixel masks, and a text-free version of every
  slide (target for text removal). See `slidegen/`.
- **Real (out-of-distribution test):** 30–50 NotebookLM slides on mixed topics,
  labelled by hand. Never used for training (except the optional few-shot
  experiment, which uses ~10 of them and keeps the rest as test).

## Evaluation
| What | Metric |
|---|---|
| Element detection | mAP@0.5, mAP@0.5:0.95 per class |
| Text reading | character error rate (CER) |
| Font attributes | size error (px), colour error (ΔE), bold accuracy |
| Whole slide | render the rebuilt .pptx and compare with the input (SSIM / LPIPS) |
| **Key result** | synthetic test score vs real NotebookLM score (the sim-to-real gap), and how much augmentation closes it |

## Timeline
| Week | Dates | Work | Done when |
|---|---|---|---|
| 1 | Nov 18 – 24 | Slide generator, Colab setup, collect NotebookLM slides | 5k synthetic slides on Drive; 30–50 real slides collected |
| 2 | Nov 25 – Dec 1 | Train YOLO; hook up OCR; train font CNN; basic inpainting | Detector mAP on synthetic + real test; OCR + font working on crops |
| 3 | Dec 2 – 8 | Rebuilder, full evaluation, Gradio demo | Upload image → download .pptx works end to end |
| 4 | Dec 9 – 15 | Freeze code (Dec 10), slides, video, buffer | Submitted |

**Stretch (only if the pipeline works end to end by ~Dec 5):** our own U-Net that
labels every pixel as background / title / text / shape / picture / icon, compared
with YOLO. The generator already saves the per-pixel masks it needs.

## Compute
- Data generation: Colab CPU (LibreOffice) or any Linux machine.
- Training: free Colab T4; RTX 3050 laptop for small runs and the demo.

## Deliverables (Dec 15)
- [ ] GitHub repo with README, results table, before/after examples
- [ ] Gradio demo (local + optionally Hugging Face Spaces)
- [ ] Presentation slides
- [ ] Presentation video
