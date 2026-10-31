"""DocIR web app: scanned page images or scanned PDFs in, faithful vector PDFs
out, each with its fidelity score (ink precision / recall / F1 against the scan).

Run:   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
GPU:   install paddlepaddle-gpu; [runtime] device in docir_config.toml picks the card.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from docir.config import (app_public, keep_runs_hours, line_ocr_enabled, max_pages_per_document,
                          paddle_device_kwargs, preprocess_mode, vlm_backend_kwargs)
from docir.document import merge_pdfs
from docir.pipeline import convert_page, document_pages, score_pdf_file

IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"]
INPUT_TYPES = IMAGE_TYPES + ["pdf"]
RUNS_DIR = Path(os.environ.get("DOCIR_RUNS_DIR", Path.home() / ".docir_runs"))
SCORE_FIELDS = ["name", "pages", "precision", "recall", "f1", "stray_ink", "seconds", "error"]
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


def _output_pdf(source: Path, pdf_dir: Path) -> Path:
    """`<name>.pdf`, or `<name>_docir.pdf` for a PDF input so the scan is never overwritten."""
    suffix = "_docir" if source.suffix.lower() == ".pdf" else ""
    return pdf_dir / f"{_safe_name(source)}{suffix}.pdf"


def convert(source: Path, pdf_dir: Path, work_dir: Path, preprocess: str) -> dict:
    """One document (image or scanned PDF): every page analysed, rendered and
    scored, then joined into `<pdf_dir>/<name>.pdf`. Never raises -- a page
    that fails is kept as its scan and its error goes in the row."""
    vl, lines, lock = _engines()
    name = _safe_name(source)
    pdf = _output_pdf(source, pdf_dir)
    row = {"name": name, "pdf": str(pdf), "pages": 0, "page_rows": [], "error": ""}
    t0 = time.time()
    try:
        page_dir = work_dir / name
        page_dir.mkdir(parents=True, exist_ok=True)
        pages = []
        for image in document_pages(str(source), str(page_dir), max_pages=max_pages_per_document()):
            _prepare_paddle_thread()
            page = convert_page(image, str(page_dir), engine=vl, line_engine=lines,
                                preprocess=preprocess, model_lock=lock)
            if not page["error"]:
                r = score_pdf_file(page["json"], page["pdf"])
                page.update(precision=r.precision, recall=r.recall, f1=r.f1, stray_ink=r.stray_ink)
            pages.append(page)
        merge_pdfs([p["pdf"] for p in pages], str(pdf))
        scored = [p for p in pages if not p["error"]]
        row.update(pages=len(pages), page_rows=pages)
        if scored:
            row.update({k: round(sum(p[k] for p in scored) / len(scored), 3)
                        for k in ("precision", "recall", "f1")},
                       stray_ink=sum(p["stray_ink"] for p in scored))
        failed = [f"page {i}: {p['error']}" for i, p in enumerate(pages, 1) if p["error"]]
        if failed:
            row["error"] = "; ".join(failed) + (" (kept as scan)" if scored else "")
    except Exception as e:  # one bad document must not stop a batch
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    row["seconds"] = round(time.time() - t0, 1)
    return row


def write_scores(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_batch(inputs: list[Path], pdf_dir: Path, work_dir: Path, preprocess: str) -> list[dict]:
    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    bar = st.progress(0.0, text="Loading models...")
    status = st.empty()
    for i, source in enumerate(inputs, 1):
        bar.progress((i - 1) / len(inputs), text=f"[{i}/{len(inputs)}] {source.name}")
        row = convert(source, pdf_dir, work_dir, preprocess)
        rows.append(row)
        write_scores(rows, work_dir / "scores.csv")
        result = f"F1 {row['f1']:.3f}" if "f1" in row else "failed"
        status.write(f"`{row['name']}` ({row['pages']} page(s)): {result}"
                     + (f", {row['error']}" if row["error"] else "") + f" in {row['seconds']} s")
    bar.progress(1.0, text=f"Done: {len(inputs)} document(s)")
    return rows


def _zip(rows: list[dict], work_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            if Path(r["pdf"]).is_file():
                z.write(r["pdf"], f"pdf/{Path(r['pdf']).name}")
        scores = work_dir / "scores.csv"
        if scores.is_file():
            z.write(scores, "scores.csv")
    return buf.getvalue()


def _preview(page: dict) -> None:
    from docir.fidelity import rasterize_pdf

    src = json.loads(Path(page["json"]).read_text()).get("input_path")
    if not src or not Path(src).is_file():
        return
    source = Image.open(src).convert("RGB")
    rendered = rasterize_pdf(page["pdf"], source.width, source.height)
    scale = min(1.0, PREVIEW_WIDTH_PX / source.width)
    size = (round(source.width * scale), round(source.height * scale))
    left, right = st.columns(2)
    left.image(source.resize(size), caption="Scan (after straightening)")
    right.image(rendered.resize(size), caption="Rendered PDF")


def show_results(rows: list[dict], work_dir: Path) -> None:
    ok = [r for r in rows if "f1" in r]
    st.subheader("Results")
    if ok:
        cols = st.columns(4)
        for col, key in zip(cols, ("precision", "recall", "f1")):
            col.metric(f"Mean {key}", f"{sum(r[key] for r in ok) / len(ok):.3f}")
        pages = [p for r in rows for p in r.get("page_rows", [])]
        cols[3].metric("Pages converted", f"{sum(1 for p in pages if not p['error'])}/{len(pages)}")
    st.dataframe([{k: r.get(k) for k in SCORE_FIELDS} for r in sorted(rows, key=lambda r: r.get("f1", -1))],
                 width="stretch", hide_index=True)
    st.download_button("Download all PDFs + scores (zip)", _zip(rows, work_dir),
                       file_name=f"docir_{work_dir.parent.name if work_dir.name == '_docir' else work_dir.name}.zip",
                       mime="application/zip")
    for r in sorted(rows, key=lambda r: r.get("f1", -1)):
        if not Path(r["pdf"]).is_file():
            st.error(f"{r['name']}: {r['error']}")
            continue
        score = f"F1 {r['f1']:.3f}   (P {r['precision']:.3f}, R {r['recall']:.3f})" if "f1" in r else "not converted"
        with st.expander(f"{r['name']}   {r['pages']} page(s)   {score}"):
            if r["error"]:
                st.warning(r["error"])
            st.download_button("Download PDF", Path(r["pdf"]).read_bytes(), file_name=Path(r["pdf"]).name,
                               mime="application/pdf", key=f"dl_{r['pdf']}")
            scored = [(i, p) for i, p in enumerate(r["page_rows"], 1) if not p["error"]]
            if scored and st.checkbox("Show side-by-side preview", key=f"pv_{r['pdf']}"):
                i = st.selectbox("Page", [i for i, _ in scored], key=f"pg_{r['pdf']}",
                                 format_func=lambda n: f"page {n}") if len(scored) > 1 else scored[0][0]
                _preview(r["page_rows"][i - 1])


def _purge_old_runs() -> None:
    """Delete upload folders older than `[app] keep_runs_hours` (0 keeps everything)."""
    hours = keep_runs_hours()
    if hours <= 0 or not RUNS_DIR.is_dir():
        return
    cutoff = time.time() - hours * 3600
    for run in RUNS_DIR.iterdir():
        if run.is_dir() and run.stat().st_mtime < cutoff:
            shutil.rmtree(run, ignore_errors=True)


def _save_uploads(files, run_dir: Path) -> list[Path]:
    """Uploaded files under sanitised, unique names (client file names are not trusted)."""
    paths = []
    for f in files:
        name = Path(f.name).name
        path = run_dir / f"{_safe_name(Path(name))}{Path(name).suffix.lower()}"
        n = 1
        while path.exists():
            n += 1
            path = run_dir / f"{_safe_name(Path(name))}_{n}{Path(name).suffix.lower()}"
        path.write_bytes(f.getbuffer())
        paths.append(path)
    return paths


def main() -> None:
    st.set_page_config(page_title="DocIR", layout="wide")
    st.title("DocIR: scanned page to faithful PDF")
    public = app_public()
    with st.sidebar:
        if not public:
            st.caption(_device_summary())
            vlm = vlm_backend_kwargs()
            st.caption(f"VLM: {vlm['vl_rec_backend']}"
                       + (f" ({vlm['vl_rec_server_url']})" if "vl_rec_server_url" in vlm else ""))
        modes = ["auto", "off", "force"]
        preprocess = st.selectbox("Page straightening", modes, index=modes.index(preprocess_mode()),
                                  help="auto: rotate/deskew/unwarp only when detected; off: as-is; force: always unwarp")
        if not public:
            st.caption(f"Uploads and outputs are kept under `{RUNS_DIR}` (set DOCIR_RUNS_DIR to change).")
        elif keep_runs_hours() > 0:
            st.caption(f"Uploaded files and results are deleted after {keep_runs_hours():g} hours.")

    tabs = st.tabs(["Upload scans"] if public else ["Upload scans", "Folder on this server"])
    with tabs[0]:
        files = st.file_uploader(f"Page images or scanned PDFs (one output PDF per file, up to "
                                 f"{max_pages_per_document()} pages each)", type=INPUT_TYPES,
                                 accept_multiple_files=True)
        if st.button("Convert", type="primary", disabled=not files):
            _purge_old_runs()
            run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            inputs = _save_uploads(files, run_dir)
            st.session_state["result"] = (run_batch(inputs, run_dir, run_dir / "_docir", preprocess),
                                          run_dir / "_docir")
    if public:
        if "result" in st.session_state:
            show_results(*st.session_state["result"])
        return
    with tabs[1]:
        folder = st.text_input("Folder path on the server",
                               help="PDFs are written next to the inputs (<name>_docir.pdf for a PDF input); "
                                    "layouts, logs and scores.csv go to <folder>/_docir")
        skip_done = st.checkbox("Skip inputs that already have an output PDF", value=True)
        if st.button("Convert folder", type="primary", disabled=not folder):
            root = Path(folder).expanduser()
            if not root.is_dir():
                st.error(f"Not a folder: {root}")
            else:
                files = sorted(p for p in root.iterdir() if p.suffix.lower().lstrip(".") in INPUT_TYPES)
                image_stems = {p.stem for p in files if p.suffix.lower() != ".pdf"}
                # our own outputs (<image>.pdf, <pdf>_docir.pdf) and flattened pages are not inputs
                inputs = [p for p in files if not p.stem.endswith("_flat") and not (
                    p.suffix.lower() == ".pdf" and (p.stem.endswith("_docir") or p.stem in image_stems))]
                if skip_done:
                    inputs = [p for p in inputs if not _output_pdf(p, root).is_file()]
                if not inputs:
                    st.info("No inputs left to convert.")
                else:
                    st.session_state["result"] = (run_batch(inputs, root, root / "_docir", preprocess),
                                                  root / "_docir")

    if "result" in st.session_state:
        show_results(*st.session_state["result"])


main()
