"""Re-render every cached page with the current code and score it, in parallel.

    .docir_env/bin/python scripts/eval_corpus.py OUT_DIR [--only CATEGORY ...] [--workers N]

Writes OUT_DIR/scores.csv (one row per page) and OUT_DIR/blocks.csv (one row per
scored block), and prints a per-category summary.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
import traceback
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CORPORA = {
    "eval": os.path.join(REPO, "output", "*_res.json"),
    "resume": os.path.join(REPO, "output", "resume", "*_res.json"),
    "report": os.path.expanduser("~/Downloads/scanned_dataset/Report/_docir/*_res.json"),
    "scientific": os.path.expanduser("~/Downloads/scanned_dataset/Scientific/_docir/*_res.json"),
}
# its layout belongs to the flattened image (vl1_6_3_flat), which is scored on its own
SKIP = {"vl1_6_3"}


def _one(job: tuple[str, str, str]) -> dict:
    from docir.pipeline import render_json_file, score_pdf_file

    category, js, out_dir = job
    name = os.path.basename(js)[:-len("_res.json")]
    pdf = os.path.join(out_dir, category, f"{name}.pdf")
    row = {"category": category, "name": name, "json": js, "pdf": pdf, "error": ""}
    t0 = time.time()
    try:
        render_json_file(js, pdf)
        rep = score_pdf_file(js, pdf)
        row.update(precision=round(rep.precision, 4), recall=round(rep.recall, 4), f1=round(rep.f1, 4),
                   stray_ink=rep.stray_ink,
                   blocks=[(b.block_id, b.label, round(b.precision, 3), round(b.recall, 3), round(b.f1, 3),
                            b.source_ink) for b in rep.blocks])
    except Exception as e:  # one failing page must not stop the corpus
        row["error"] = f"{type(e).__name__}: {e}"
        row["trace"] = traceback.format_exc(limit=3)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def _kept(path: str, rerun: list[str]) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["category"] not in rerun]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--only", nargs="*", default=list(CORPORA))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = []
    for cat in args.only:
        os.makedirs(os.path.join(args.out_dir, cat), exist_ok=True)
        for js in sorted(glob.glob(CORPORA[cat])):
            if os.path.basename(js)[:-len("_res.json")] not in SKIP:
                jobs.append((cat, js, args.out_dir))
    rows = []
    with Pool(args.workers, maxtasksperchild=10) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["category"], r["name"]))
    page_fields = ["category", "name", "precision", "recall", "f1", "stray_ink", "seconds", "error", "json", "pdf"]
    # keep earlier results for categories this run did not touch
    rows += _kept(os.path.join(args.out_dir, "scores.csv"), args.only)
    blocks_kept = _kept(os.path.join(args.out_dir, "blocks.csv"), args.only)
    with open(os.path.join(args.out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=page_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "blocks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "name", "block_id", "label", "precision", "recall", "f1", "source_ink"])
        for r in rows:
            for b in r.get("blocks", []):
                w.writerow([r["category"], r["name"], *b])
        for b in blocks_kept:
            w.writerow(list(b.values()))
    print(f"\n{'category':<12}{'pages':>6}{'errors':>7}{'P':>7}{'R':>7}{'F1':>7}{'F1<0.7':>8}")
    for cat in args.only:
        ok = [r for r in rows if r["category"] == cat and not r["error"]]
        err = sum(1 for r in rows if r["category"] == cat and r["error"])
        if not ok:
            print(f"{cat:<12}{0:>6}{err:>7}")
            continue
        mean = lambda k: sum(r[k] for r in ok) / len(ok)  # noqa: E731
        low = sum(1 for r in ok if r["f1"] < 0.7)
        print(f"{cat:<12}{len(ok):>6}{err:>7}{mean('precision'):>7.3f}{mean('recall'):>7.3f}{mean('f1'):>7.3f}{low:>8}")
    for r in rows:
        if r["error"]:
            print("ERROR", r["category"], r["name"], r["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
