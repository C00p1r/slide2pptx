"""Render a .pptx to PNG pages with LibreOffice (pptx -> pdf) and poppler (pdf -> png)."""
import glob
import os
import subprocess

from . import CANVAS_H, CANVAS_W


def pptx_to_pdf(pptx_path, out_dir, profile_dir, timeout=600):
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["soffice", f"-env:UserInstallation=file://{os.path.abspath(profile_dir)}",
           "--headless", "--convert-to", "pdf", "--outdir", out_dir, pptx_path]
    subprocess.run(cmd, check=True, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = os.path.join(out_dir, os.path.splitext(os.path.basename(pptx_path))[0] + ".pdf")
    if not os.path.exists(pdf):
        raise RuntimeError(f"LibreOffice did not produce {pdf}")
    return pdf


def pdf_to_pngs(pdf, prefix, antialias=True):
    cmd = ["pdftoppm", "-png", "-scale-to-x", str(CANVAS_W), "-scale-to-y", str(CANVAS_H)]
    if not antialias:
        cmd += ["-aa", "no", "-aaVector", "no"]
    subprocess.run(cmd + [pdf, prefix], check=True)
    return sorted(glob.glob(prefix + "-*.png"))


def render(pptx_path, work_dir, profile_dir, antialias=True):
    pdf = pptx_to_pdf(pptx_path, work_dir, profile_dir)
    prefix = os.path.splitext(pdf)[0]
    return pdf_to_pngs(pdf, prefix, antialias)
