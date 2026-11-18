"""Synthetic slide generator: random slides -> rendered images + exact labels."""
CANVAS_W, CANVAS_H = 1280, 720          # 16:9 slide rendered at 96 dpi
EMU_PER_PX = 9525                       # 12192000 EMU / 1280 px
PX_PER_PT = 96 / 72
CLASSES = ["title", "text", "shape", "picture", "icon"]
