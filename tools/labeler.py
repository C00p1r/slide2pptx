"""Tiny local box editor for the real test slides. Standard library only.

    python tools/labeler.py --images data/real/images --pred results/pred_llm/r0.50 --sample 20

Opens http://localhost:8765 in your browser. Boxes start from the detector's predictions
(--pred, the <stem>.json files written by detector/predict.py), so you only fix them.
Each slide is saved as YOLO txt in --labels (default data/real/labels) together with
classes.txt, which is what `python tools/real_labels.py build data/real/labels` reads.

A slide counts as done once its .txt exists. Restarting the tool picks up where you left off,
and an existing .txt always wins over the predictions.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import CLASSES  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeler.html")


def deck_of(stem):
    return re.sub(r"[_-]\d+$", "", stem)


def pick(stems, n, seed):
    """n slides spread over decks: shuffle each deck, then take one per deck in turn."""
    if not n or n >= len(stems):
        return stems
    rng = random.Random(seed)
    decks = {}
    for s in stems:
        decks.setdefault(deck_of(s), []).append(s)
    queues = [rng.sample(v, len(v)) for _, v in sorted(decks.items())]
    out = []
    while len(out) < n:
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop())
    return sorted(out)


class App:
    def __init__(self, a):
        files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith(IMG_EXT))
        if not files:
            sys.exit(f"no images in {a.images}")
        self.files = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
        self.stems = pick(sorted(self.files), a.sample, a.seed)
        self.pred, self.labels, self.conf = a.pred, a.labels, a.conf
        os.makedirs(self.labels, exist_ok=True)
        with open(os.path.join(self.labels, "classes.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(CLASSES) + "\n")

    def label_path(self, stem):
        return os.path.join(self.labels, stem + ".txt")

    def listing(self):
        return {"classes": CLASSES,
                "slides": [{"stem": s, "done": os.path.exists(self.label_path(s))} for s in self.stems]}

    def boxes(self, stem):
        """Normalised [x, y, w, h] (top-left corner) from the saved label, else from predictions."""
        p = self.label_path(stem)
        if os.path.exists(p):
            out = []
            for line in open(p, encoding="utf-8"):
                k, cx, cy, w, h = line.split()[:5]
                cx, cy, w, h = map(float, (cx, cy, w, h))
                out.append({"cls": CLASSES[int(k)], "bbox": [cx - w / 2, cy - h / 2, w, h], "conf": 1})
            return {"source": "label", "norm": True, "boxes": out}
        if self.pred:
            p = os.path.join(self.pred, stem + ".json")
            if os.path.exists(p):
                dets = json.load(open(p, encoding="utf-8"))["detections"]
                return {"source": "pred", "norm": False,
                        "boxes": [d for d in dets if d["conf"] >= self.conf and d["cls"] in CLASSES]}
        return {"source": "none", "norm": True, "boxes": []}

    def save(self, stem, boxes):
        lines = []
        for b in boxes:
            x, y, w, h = (min(max(v, 0.0), 1.0) for v in b["bbox"])
            w, h = min(w, 1 - x), min(h, 1 - y)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{CLASSES.index(b['cls'])} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}\n")
        tmp = self.label_path(stem) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        os.replace(tmp, self.label_path(stem))
        return len(lines)


def handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def stem(self, prefix):
            s = unquote(urlparse(self.path).path[len(prefix):])
            return s if s in app.files else None

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self.send(200, open(HTML, "rb").read(), "text/html; charset=utf-8")
            if path == "/api/list":
                return self.send(200, app.listing())
            if path.startswith("/api/boxes/") and self.stem("/api/boxes/"):
                return self.send(200, app.boxes(self.stem("/api/boxes/")))
            if path.startswith("/img/") and self.stem("/img/"):
                f = app.files[self.stem("/img/")]
                return self.send(200, open(f, "rb").read(), MIME[os.path.splitext(f)[1].lower()])
            self.send(404, {"error": "not found"})

        def do_POST(self):
            s = self.stem("/api/save/") if self.path.startswith("/api/save/") else None
            if not s:
                return self.send(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                n = app.save(s, body.get("boxes", []))
            except (KeyError, ValueError, TypeError) as e:
                return self.send(400, {"error": str(e)})
            self.send(200, {"saved": n})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="data/real/images")
    ap.add_argument("--pred", default=None, help="folder of <stem>.json predictions to start from")
    ap.add_argument("--labels", default="data/real/labels")
    ap.add_argument("--sample", type=int, default=0, help="only label N slides, spread over decks (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25, help="drop predictions below this confidence")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    app = App(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(app))
    url = f"http://localhost:{a.port}/"
    done = sum(x["done"] for x in app.listing()["slides"])
    print(f"{len(app.stems)} slides ({done} done), labels -> {os.path.abspath(a.labels)}")
    print(f"open {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
