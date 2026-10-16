"""DocIR web app: scanned page images in, faithful vector PDFs out, each with
its fidelity score (ink precision / recall / F1 against the scan).

Run:   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
GPU:   install paddlepaddle-gpu; [runtime] device in docir_config.toml picks the card.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from docir.config import line_ocr_enabled, paddle_device_kwargs, preprocess_mode
from docir.pipeline import process_image, score_pdf_file

IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"]
RUNS_DIR = Path(os.environ.get("DOCIR_RUNS_DIR", Path.home() / ".docir_runs"))
SCORE_FIELDS = ["name", "precision", "recall", "f1", "stray_ink", "seconds", "error"]
PREVIEW_WIDTH_PX = 900


def _prepare_paddle_thread() -> None:
    """Paddle keeps its eager (dygraph) mode and device per thread, and
    Streamlit runs every script run in a fresh thread that starts in static
    mode with no device -- model loading then fails."""
    import paddle

    paddle.disable_static()
    device = paddle_device_kwargs().get("device")
    if device is None:
        has_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        device = "gpu" if has_gpu else "cpu"
    paddle.set_device(device)


@st.cache_resource
def _engines():
    """Models load once per server process and are shared by all sessions;
    the lock keeps two sessions from running inference at the same time."""
    from docir.ocr import PaddleLineOcrEngine, PaddleOcrVlEngine

    _prepare_paddle_thread()
    return (PaddleOcrVlEngine(), PaddleLineOcrEngine() if line_ocr_enabled() else None,
            threading.Lock())


@st.cache_data
def _device_summary() -> str:
    try:
        import paddle
    except ImportError:
        return "paddle not installed (install the `ocr` extra)"
    configured = paddle_device_kwargs().get("device", "auto")
    if not paddle.device.is_compiled_with_cuda():
        return f"CPU build of paddle {paddle.__version__} (device setting: {configured})"
    n = paddle.device.cuda.device_count()
    names = [paddle.device.cuda.get_device_name(i) for i in range(n)]
    return f"CUDA build, {n} GPU(s): {', '.join(names) or 'none found'} (device setting: {configured})"


def _safe_name(path: Path) -> str:
    return re.sub(r"[^\w.+-]+", "_", path.stem)


def convert(image: Path, pdf_dir: Path, work_dir: Path, preprocess: str) -> dict:
    """One page: analyze, render, score. Never raises -- errors go in the row."""
    vl, lines, lock = _engines()
    name = _safe_name(image)
    pdf, js = pdf_dir / f"{name}.pdf", work_dir / f"{name}_res.json"
    row = {"name": name, "pdf": str(pdf), "json": str(js), "error": ""}
    t0 = time.time()
    try:
        with lock:
            _prepare_paddle_thread()
            process_image(str(image), output_pdf=str(pdf), output_json=str(js), engine=vl,
                          line_engine=lines, preprocess=preprocess)
        r = score_pdf_file(str(js), str(pdf))
        row.update(precision=round(r.precision, 3), recall=round(r.recall, 3), f1=round(r.f1, 3),
                   stray_ink=r.stray_ink)
    except Exception as e:  # one bad page must not stop a batch
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    row["seconds"] = round(time.time() - t0, 1)
    return row


def write_scores(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_batch(images: list[Path], pdf_dir: Path, work_dir: Path, preprocess: str) -> list[dict]:
    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    bar = st.progress(0.0, text="Loading models...")
    status = st.empty()
    for i, image in enumerate(images, 1):
        bar.progress((i - 1) / len(images), text=f"[{i}/{len(images)}] {image.name}")
        row = convert(image, pdf_dir, work_dir, preprocess)
        rows.append(row)
        write_scores(rows, work_dir / "scores.csv")
        status.write(f"`{row['name']}`: " + (f"F1 {row['f1']:.3f}" if not row["error"] else f"failed ({row['error']})")
                     + f" in {row['seconds']} s")
    bar.progress(1.0, text=f"Done: {len(images)} page(s)")
    return rows


def _zip(rows: list[dict], work_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            if not r["error"] and Path(r["pdf"]).is_file():
                z.write(r["pdf"], f"pdf/{Path(r['pdf']).name}")
        scores = work_dir / "scores.csv"
        if scores.is_file():
            z.write(scores, "scores.csv")
    return buf.getvalue()


def _preview(row: dict) -> None:
    from docir.fidelity import rasterize_pdf

    src = json.loads(Path(row["json"]).read_text()).get("input_path")
    if not src or not Path(src).is_file():
        return
    source = Image.open(src).convert("RGB")
    rendered = rasterize_pdf(row["pdf"], source.width, source.height)
    scale = min(1.0, PREVIEW_WIDTH_PX / source.width)
    size = (round(source.width * scale), round(source.height * scale))
    left, right = st.columns(2)
    left.image(source.resize(size), caption="Scan (after straightening)")
    right.image(rendered.resize(size), caption="Rendered PDF")


def show_results(rows: list[dict], work_dir: Path) -> None:
    ok = [r for r in rows if not r["error"]]
    failed = [r for r in rows if r["error"]]
    st.subheader("Results")
    if ok:
        cols = st.columns(4)
        for col, key in zip(cols, ("precision", "recall", "f1")):
            col.metric(f"Mean {key}", f"{sum(r[key] for r in ok) / len(ok):.3f}")
        cols[3].metric("Pages converted", f"{len(ok)}/{len(rows)}")
    st.dataframe([{k: r.get(k) for k in SCORE_FIELDS} for r in sorted(rows, key=lambda r: r.get("f1", -1))],
                 width="stretch", hide_index=True)
    st.download_button("Download all PDFs + scores (zip)", _zip(rows, work_dir),
                       file_name=f"docir_{work_dir.parent.name if work_dir.name == '_docir' else work_dir.name}.zip",
                       mime="application/zip")
    for r in sorted(ok, key=lambda r: r["f1"]):
        with st.expander(f"{r['name']}   F1 {r['f1']:.3f}   (P {r['precision']:.3f}, R {r['recall']:.3f})"):
            st.download_button("Download PDF", Path(r["pdf"]).read_bytes(), file_name=Path(r["pdf"]).name,
                               mime="application/pdf", key=f"dl_{r['pdf']}")
            if st.checkbox("Show side-by-side preview", key=f"pv_{r['pdf']}"):
                _preview(r)
    for r in failed:
        st.error(f"{r['name']}: {r['error']}")


def main() -> None:
    st.set_page_config(page_title="DocIR", layout="wide")
    st.title("DocIR: scanned page to faithful PDF")
    with st.sidebar:
        st.caption(_device_summary())
        modes = ["auto", "off", "force"]
        preprocess = st.selectbox("Page straightening", modes, index=modes.index(preprocess_mode()),
                                  help="auto: rotate/deskew/unwarp only when detected; off: as-is; force: always unwarp")
        st.caption(f"Uploads and outputs are kept under `{RUNS_DIR}` (set DOCIR_RUNS_DIR to change).")

    upload_tab, folder_tab = st.tabs(["Upload images", "Folder on this server"])
    with upload_tab:
        files = st.file_uploader("Page images", type=IMAGE_TYPES, accept_multiple_files=True)
        if st.button("Convert", type="primary", disabled=not files):
            run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            images = []
            for f in files:
                path = run_dir / f.name
                path.write_bytes(f.getbuffer())
                images.append(path)
            st.session_state["result"] = (run_batch(images, run_dir, run_dir / "_docir", preprocess),
                                          run_dir / "_docir")
    with folder_tab:
        folder = st.text_input("Folder path on the server",
                               help="PDFs are written next to the images; layouts, logs and scores.csv go to <folder>/_docir")
        skip_done = st.checkbox("Skip images that already have a PDF", value=True)
        if st.button("Convert folder", type="primary", disabled=not folder):
            root = Path(folder).expanduser()
            if not root.is_dir():
                st.error(f"Not a folder: {root}")
            else:
                images = sorted(p for p in root.iterdir() if p.suffix.lower().lstrip(".") in IMAGE_TYPES
                                and not p.stem.endswith("_flat"))
                if skip_done:
                    images = [p for p in images if not (root / f"{_safe_name(p)}.pdf").is_file()]
                if not images:
                    st.info("No images left to convert.")
                else:
                    st.session_state["result"] = (run_batch(images, root, root / "_docir", preprocess),
                                                  root / "_docir")

    if "result" in st.session_state:
        show_results(*st.session_state["result"])


main()
