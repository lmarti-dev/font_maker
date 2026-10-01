"""
harvest.py - pull letterforms out of handwritten scans (built for raw, non-cursive hands).

  python harvest.py segment scan.pdf -o work
  python harvest.py build   work -o svg [--labels work/labels.txt] [--otf Font.otf --family "Name"]

segment : PDF/image -> ink mask -> candidate glyphs (+ numbered contact sheets and page overlays)
build   : labels.txt (ID=char) -> one SVG per character, baseline-aligned, via potrace;
          with --otf, also compiles them to a font through svg2otf.py (same folder)

Deps: opencv-python-headless numpy pymupdf potracer fonttools svgelements fire
"""

import json
import platform
import string
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


# --------------------------------------------------------------------------- input
def _pix_to_bgr(pix):
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 1:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if pix.n == 4:
        a = a[:, :, :3]
    return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)


def load_pages(path, dpi=300, pages=None):
    """Yield (page_number, BGR image). Scanned PDFs: use the embedded original, not a re-render."""
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"cannot read {p}")
        yield 1, img
        return
    import pymupdf

    doc = pymupdf.open(p)
    for i, page in enumerate(doc, 1):
        if pages and i not in pages:
            continue
        img = None
        imgs = page.get_images(full=True)
        if len(imgs) == 1 and page.rotation == 0:
            pix = pymupdf.Pixmap(doc, imgs[0][0])
            if pix.colorspace is not None and pix.colorspace.n not in (1, 3):
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            # only use it if it actually covers the page (judged by placement, not pixel count,
            # so scans below 300 dpi are still taken at their native resolution)
            rects = page.get_image_rects(imgs[0][0])
            if rects and rects[0].get_area() >= 0.5 * page.rect.get_area():
                img = _pix_to_bgr(pix)
        if img is None:
            img = _pix_to_bgr(page.get_pixmap(dpi=dpi, alpha=False))
        yield i, img


# --------------------------------------------------------------------------- ink mask
def ink_mask(bgr, thresh=None):
    """Dark ink -> 255. Uses max(B,G,R) so red crayon rules and blue stamps (bright in one
    channel) drop out while graphite / black ink (dark in all channels) stays."""
    v = bgr.max(axis=2)
    h, w = v.shape
    s = 4
    small = cv2.resize(v, (w // s, h // s), interpolation=cv2.INTER_AREA)
    k = max(3, int(0.04 * max(h, w) / s)) | 1
    bg = cv2.morphologyEx(
        small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    bg = cv2.GaussianBlur(bg, (k | 1, k | 1), 0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    norm = np.clip(v.astype(np.float32) / np.maximum(bg, 1), 0, 1.2)
    n8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    if thresh is None:
        t, _ = cv2.threshold(n8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(np.clip(t, 0.55 * 255, 0.85 * 255))
    else:
        t = thresh * 255
    return (n8 < t).astype(np.uint8) * 255, norm


# --------------------------------------------------------------------------- segmentation
def segment_page(mask, min_area=None):
    H, W = mask.shape
    min_area = min_area or max(12, int(8e-6 * H * W))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area:
            continue
        if x == 0 or y == 0 or x + w >= W or y + h >= H:  # page edge / shadow
            continue
        if w > 0.5 * W or h > 0.25 * H:  # stamp ring, folds
            continue
        comps.append({"ids": [i], "x": int(x), "y": int(y), "w": int(w), "h": int(h), "a": int(a)})
    if not comps:
        return [], lab
    big = [c for c in comps if c["a"] >= np.median([c["a"] for c in comps])]
    mh = float(np.median([c["h"] for c in big]))
    med_area = float(np.median([c["a"] for c in big]))

    # attach dots, accents, thin crossbars to the letter they belong to
    def small(c):
        return max(c["w"], c["h"]) < 0.55 * mh or (
            min(c["w"], c["h"]) < 0.22 * mh and max(c["w"], c["h"]) < 1.0 * mh
        )

    parents = [c for c in comps if not small(c)]
    kept = list(parents)
    for c in comps:
        if not small(c):
            continue
        best, bo = None, 0.0
        for p in parents:
            ov = min(c["x"] + c["w"], p["x"] + p["w"] + 0.1 * mh) - max(
                c["x"], p["x"] - 0.1 * mh
            )
            gap = max(p["y"] - (c["y"] + c["h"]), c["y"] - (p["y"] + p["h"]), 0)
            if ov / max(c["w"], 1) >= 0.5 and gap <= 0.5 * mh and ov > bo:
                best, bo = p, ov
        if best is None:
            if c["a"] >= 0.12 * med_area:  # stray mark: punctuation etc. (not dust)
                kept.append(c)
        else:
            best["ids"] += c["ids"]
    for c in kept:  # recompute merged bbox
        if len(c["ids"]) > 1:
            m = np.isin(lab, c["ids"])
            ys, xs = np.nonzero(m)
            c.update(
                x=int(xs.min()),
                y=int(ys.min()),
                w=int(xs.max() - xs.min() + 1),
                h=int(ys.max() - ys.min() + 1),
                a=int(m.sum()),
            )

    # rough reading order: track lines left to right (display only - baselines don't depend on it)
    lines = []
    for c in sorted(kept, key=lambda c: c["x"] + c["w"] / 2):
        cy = c["y"] + c["h"] / 2
        best, bd = None, 1.0 * mh
        for ln in lines:
            d = abs(cy - np.median([k["y"] + k["h"] / 2 for k in ln[-3:]]))
            if d < bd:
                best, bd = ln, d
        if best is None:
            lines.append([c])
        else:
            best.append(c)
    lines.sort(key=lambda ln: np.mean([k["y"] + k["h"] / 2 for k in ln]))
    # local baseline: median bottom of the nearest glyphs; descenders/ascenders are the minority
    cx = np.array([c["x"] + c["w"] / 2 for c in kept])
    cyv = np.array([c["y"] + c["h"] / 2 for c in kept])
    bot = np.array([c["y"] + c["h"] for c in kept])
    for j, c in enumerate(kept):
        near = np.where(
            (np.abs(cyv - cyv[j]) < 1.0 * mh) & (np.abs(cx - cx[j]) < 6 * mh)
        )[0]
        near = near[np.argsort(np.abs(cx[near] - cx[j]))[:9]]
        c["baseline"] = float(np.median(bot[near])) if len(near) >= 3 else float(bot[j])
    out = []
    for li, ln in enumerate(lines):
        ln.sort(key=lambda c: c["x"] + c["w"] / 2)
        for j, c in enumerate(ln):
            c["line"] = li
            c["order"] = j
            c["flags"] = [
                f
                for f, cond in (
                    ("wide", c["w"] > 1.7 * mh),
                    ("tall", c["h"] > 2.0 * mh),
                    ("merged", len(c["ids"]) > 1),
                )
                if cond
            ]
            out.append(c)
    return out, lab, mh


# --------------------------------------------------------------------------- outputs
def contact_sheets(glyphs, gdir, odir, pageno, cols=10, rows=8, cell=120):
    per = cols * rows
    for s in range(0, len(glyphs), per):
        sheet = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
        for k, g in enumerate(glyphs[s : s + per]):
            m = cv2.imread(str(gdir / f"{g['id']:05d}.png"), 0)
            f = min((cell - 24) / m.shape[1], (cell - 24) / m.shape[0])
            m = cv2.resize(255 - m, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            r, c = divmod(k, cols)
            y0, x0 = r * cell + 4, c * cell + (cell - m.shape[1]) // 2
            sheet[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]] = m[:, :, None]
            cv2.rectangle(
                sheet,
                (c * cell, r * cell),
                ((c + 1) * cell - 1, (r + 1) * cell - 1),
                (210, 210, 210),
                1,
            )
            cv2.putText(
                sheet,
                str(g["id"]),
                (c * cell + 4, (r + 1) * cell - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 200),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(odir / f"sheet_p{pageno}_{s // per + 1:02d}.png"), sheet)


def overlay(bgr, glyphs, path, maxw=2200):
    o = bgr.copy()
    for g in glyphs:
        col = (200, 80, 0) if g["line"] % 2 else (0, 140, 0)
        cv2.rectangle(o, (g["x"], g["y"]), (g["x"] + g["w"], g["y"] + g["h"]), col, 2)
        cv2.putText(
            o,
            str(g["id"]),
            (g["x"], g["y"] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 220),
            1,
            cv2.LINE_AA,
        )
    if o.shape[1] > maxw:
        o = cv2.resize(
            o,
            None,
            fx=maxw / o.shape[1],
            fy=maxw / o.shape[1],
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), o)


def segment(
    input: str,
    out: str = "work",
    pages: str | None = None,
    dpi: int = 300,
    thresh: float | None = None,
    min_area: int | None = None,
) -> None:
    """Extract candidate glyphs from a PDF or image.

    Args:
        input: Input PDF or image path.
        out: Output working directory.
        pages: Comma-separated 1-based PDF page numbers, e.g. "1,2,5".
        dpi: Render DPI for PDF pages that need rasterization.
        thresh: Optional ink threshold in the range 0-1.
        min_area: Minimum connected-component area in pixels.
    """
    # Fire's current argument conversion is value-based rather than driven by
    # annotations, so normalize the boundary values explicitly.
    input = str(input)
    out = str(out)
    pages = str(pages) if pages is not None else None
    dpi = int(dpi)
    thresh = float(thresh) if thresh is not None else None
    min_area = int(min_area) if min_area is not None else None

    out_path = Path(out)
    gdir = out_path / "glyphs"
    gdir.mkdir(parents=True, exist_ok=True)
    page_numbers = {int(x) for x in pages.split(",")} if pages else None
    meta, nid = [], 1
    for pno, bgr in load_pages(input, dpi, page_numbers):
        mask, _ = ink_mask(bgr, thresh)
        res = segment_page(mask, min_area)
        if not res[0]:
            print(f"page {pno}: nothing found")
            continue
        glyphs, lab, mh = res
        glyphs.sort(key=lambda g: (g["line"], g["order"]))
        for g in glyphs:
            g["id"], g["page"] = nid, pno
            nid += 1
            m = np.isin(lab, g["ids"]).astype(np.uint8) * 255
            pad = 3
            crop = m[
                max(g["y"] - pad, 0) : g["y"] + g["h"] + pad,
                max(g["x"] - pad, 0) : g["x"] + g["w"] + pad,
            ]
            cv2.imwrite(str(gdir / f"{g['id']:05d}.png"), crop)
            g["pad"] = pad
            del g["ids"]
        meta += glyphs
        contact_sheets(glyphs, gdir, out_path, pno)
        overlay(bgr, glyphs, out_path / f"overlay_p{pno}.jpg")
        cv2.imwrite(str(out_path / f"mask_p{pno}.png"), mask)
        nlines = max(g["line"] for g in glyphs) + 1
        print(
            f"page {pno}: {len(glyphs)} candidates, ~{nlines} text lines "
            f"(median glyph height {mh:.0f}px)"
        )
    (out_path / "candidates.json").write_text(json.dumps(meta, indent=1))
    lf = out_path / "labels.txt"
    if not lf.exists():
        lf.write_text(
            "# ID=char - one per line, e.g.  17=a   42=R "
            "(several IDs per char are fine: the most\n"
            "# typical becomes the main glyph, the rest go to alts/)\n"
        )
    print(
        f"{len(meta)} candidates -> {out_path}/   next: look at "
        f"sheet_*.png / overlay_*.jpg, fill in {lf}"
    )


# --------------------------------------------------------------------------- build
def trace_svg_path(mask, tx):
    import potrace

    plist = potrace.Bitmap(
        mask == 0
    ).trace(  # potracer: True = paper, so pass the inverse of the ink
        turdsize=4, alphamax=1.0, opticurve=True, opttolerance=0.4
    )
    f = lambda p: f"{tx(p.x, p.y)[0]:.1f},{tx(p.x, p.y)[1]:.1f}"
    parts = []
    for curve in plist:
        parts.append("M" + f(curve.start_point))
        for s in curve.segments:
            parts.append(
                ("L" + f(s.c) + "L" + f(s.end_point))
                if s.is_corner
                else ("C" + f(s.c1) + " " + f(s.c2) + " " + f(s.end_point))
            )
        parts.append("Z")
    return "".join(parts)


def glyph_filename(ch, naming):
    """a.svg, A.svg (or A_upper.svg), and AGL names for everything else (period.svg, eacute.svg)."""
    if ch.isascii() and ch.isalnum():
        return ch + ("_upper" if naming == "suffix" and ch.isupper() else "")
    from fontTools.agl import UV2AGL

    return UV2AGL.get(ord(ch)) or f"uni{ord(ch):04X}"


def compile_otf(svg_dir, mapping_path, otf, family, canvas, baseline):
    """Hand the SVG folder to svg2otf.build_font (sits next to this file)."""
    import importlib.util

    path = Path(__file__).with_name("svg2otf.py")
    spec = importlib.util.spec_from_file_location("svg2otf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_font(
        input=str(svg_dir),
        output=str(otf),
        mapping=str(mapping_path),
        upm=canvas,
        descent=canvas - baseline,
        family=family or Path(otf).stem,
        fill_gaps=False,
    )


Naming = Literal["plain", "suffix"]


def build_glyphs(
    work: str,
    out: str = "svg",
    labels: str | None = None,
    charset: str = string.ascii_letters + string.digits,
    naming: Naming = "plain",
    canvas: int = 1000,
    baseline: int = 800,
    space: int = 350,
    otf: str | None = None,
    family: str | None = None,
    cap: int = 700,
    side: int = 40,
) -> None:
    """Build baseline-aligned SVG glyphs and optionally compile an OTF.

    Args:
        work: Working directory produced by ``segment``.
        out: Output directory for SVGs and metadata.
        labels: Optional labels.txt path; defaults to ``work/labels.txt``.
        charset: Characters to export.
        naming: ``plain`` uses ``A.svg``; ``suffix`` uses ``A_upper.svg``.
        canvas: SVG canvas height, in font-em units.
        baseline: Baseline Y coordinate inside the canvas.
        space: Width of the generated space glyph.
        otf: Optional output OTF path.
        family: Optional font family name used when compiling the OTF.
        cap: Target cap height in font units.
        side: Side bearing in font units.
    """
    work = str(work)
    out = str(out)
    labels = str(labels) if labels is not None else None
    charset = str(charset)
    naming = str(naming)
    canvas = int(canvas)
    baseline = int(baseline)
    space = int(space)
    otf = str(otf) if otf is not None else None
    family = str(family) if family is not None else None
    cap = int(cap)
    side = int(side)

    if naming not in {"plain", "suffix"}:
        raise ValueError("naming must be either 'plain' or 'suffix'")

    work_path = Path(work)
    meta = {
        g["id"]: g
        for g in json.loads((work_path / "candidates.json").read_text())
    }
    label_map = {}
    labels_path = Path(labels) if labels else work_path / "labels.txt"
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        i, c = line.split("=", 1)
        if c.strip() and i.strip().isdigit() and int(i) in meta:
            label_map.setdefault(c.strip()[0], []).append(int(i))
    charset_set = set(charset)
    label_map = {c: v for c, v in label_map.items() if c in charset_set}

    # Scale: cap height -> cap units. Use labelled capitals if we have them,
    # else median height.
    caps = [
        meta[i]["h"]
        for c in "EHIKLMNTUVWXYZ"
        for i in label_map.get(c, [])
    ]
    ref = (
        float(np.median(caps))
        if caps
        else float(np.median([g["h"] for g in meta.values()]))
    )
    scale = cap / ref
    print(
        f"scale: {ref:.0f}px reference height -> {cap} units "
        f"(x{scale:.3f}); {len(label_map)} characters labelled"
    )
    if canvas < baseline:
        raise ValueError("--canvas must be >= --baseline")

    over = []
    for ch, ids in label_map.items():
        up = max(
            (meta[i]["baseline"] - meta[i]["y"]) * scale for i in ids
        )
        dn = max(
            (meta[i]["y"] + meta[i]["h"] - meta[i]["baseline"]) * scale
            for i in ids
        )
        if up > baseline or dn > canvas - baseline:
            over.append(
                f"{ch}(+{max(up - baseline, 0):.0f}/-"
                f"{max(dn - (canvas - baseline), 0):.0f})"
            )
    if over:
        print(
            f"note: {len(over)} glyphs exceed the em box (fine for a font, "
            f"they just overlap line spacing; lower --cap to shrink "
            f"everything): {' '.join(sorted(over))}"
        )

    out_path = Path(out)
    (out_path / "alts").mkdir(parents=True, exist_ok=True)
    metrics, mapping = {}, {}
    for ch, ids in sorted(label_map.items()):
        areas = [meta[i]["a"] for i in ids]
        med = float(np.median(areas))
        ids = sorted(
            ids, key=lambda i: abs(meta[i]["a"] - med)
        )  # most typical instance first
        for rank, i in enumerate(ids):
            g = meta[i]
            m = cv2.imread(
                str(work_path / "glyphs" / f"{i:05d}.png"), 0
            )
            m = cv2.copyMakeBorder(
                m, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0
            )
            upsample = 4  # upsample + soften -> smoother curves
            m = cv2.resize(
                m,
                None,
                fx=upsample,
                fy=upsample,
                interpolation=cv2.INTER_CUBIC,
            )
            m = (
                cv2.GaussianBlur(m, (0, 0), upsample * 0.6) > 127
            ).astype(np.uint8) * 255

            # crop-pixel -> page-pixel: crop origin = bbox - pad,
            # plus the 8px border added above.
            ox = g["x"] - g["pad"] - 8
            oy = g["y"] - g["pad"] - 8

            def tx(px, py, ox=ox, oy=oy, g=g):
                x = (px / upsample + ox - g["x"]) * scale + side
                y = (py / upsample + oy - g["baseline"]) * scale + baseline
                return x, y

            width = g["w"] * scale + 2 * side
            d = trace_svg_path(m, tx)
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
                f'height="{canvas}" viewBox="0 0 {width:.0f} {canvas}" '
                f'overflow="visible" data-baseline="{baseline}">'
                f'<path fill="#000" fill-rule="evenodd" d="{d}"/></svg>'
            )
            name = glyph_filename(ch, naming)
            fn = (
                out_path / f"{name}.svg"
                if rank == 0
                else out_path / f"alts/{name}.{rank}.svg"
            )
            fn.write_text(svg)
            if rank == 0:
                mapping[fn.name] = ch
                metrics[ch] = {
                    "file": fn.name,
                    "source_id": i,
                    "width": round(width),
                    "canvas": canvas,
                    "baseline": baseline,
                    "descent": round(
                        (g["y"] + g["h"] - g["baseline"]) * scale
                    ),
                }

    if not (out_path / "space.svg").exists():
        (out_path / "space.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{space}" '
            f'height="{canvas}" viewBox="0 0 {space} {canvas}" '
            f'data-baseline="{baseline}"></svg>'
        )
    mapping["space.svg"] = "space"
    (out_path / "mapping.json").write_text(
        json.dumps(mapping, indent=1, ensure_ascii=False)
    )
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=1))
    missing = [c for c in charset if c not in label_map]
    print(f"wrote {len(metrics)} glyph SVGs to {out_path}/")
    if missing:
        print("still missing:", "".join(missing))
    if naming == "plain" and platform.system() != "Linux":
        print(
            "WARNING: case-insensitive filesystem - A.svg overwrites a.svg. "
            "Re-run with --naming suffix."
        )
    if otf:
        compile_otf(
            out_path,
            out_path / "mapping.json",
            otf,
            family,
            canvas,
            baseline,
        )

