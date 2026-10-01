#!/usr/bin/env python3
"""
svg2otf.py — Compile a folder of per-glyph SVG files into a single OTF font.

The converter is deliberately aimed at messy / hand-drawn / traced SVGs.
It preserves cubic geometry and supports multiple subpaths per SVG.

Important implementation detail:
very complex glyphs can exceed practical per-charstring limits in some
CFF consumers. To avoid that, glyph contours are stored in CFF local
subroutines and the glyph charstring merely calls those subroutines.
Very long contours are additionally split into small continuation
subroutines. This keeps the individual glyph charstring small without
simplifying or throwing away artwork.

USAGE
-----
    python3 svg2otf.py ./glyphs \
        --output Sloppy-Regular.otf \
        --family "Sloppy Sans" \
        --style Regular \
        --upm 1000 \
        --descent 200 \
        --lsb 40 \
        --rsb 40

Dependencies:
    pip install fonttools svgelements fire
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import fire
from fontTools.agl import AGL2UV, UV2AGL
from fontTools.cffLib import SubrsIndex
from fontTools.fontBuilder import FontBuilder
from fontTools.misc.psCharStrings import T2CharString
from fontTools.pens.t2CharStringPen import T2CharStringPen
from svgelements import (
    SVG,
    Arc,
    Close,
    CubicBezier,
    Line,
    Move,
    QuadraticBezier,
    Shape,
)
from svgelements import (
    Path as SvgPath,
)

from font_maker.utils import ensure_dir

ROOT = Path(__file__).parent

UNI_RE = re.compile(r"^u(?:ni)?([0-9A-Fa-f]{4,6})$")

# A contour is split into continuation subroutines after this many
# charstring operators.  This is intentionally conservative.  It does
# not alter geometry; it only changes how the CFF program is stored.
DEFAULT_SUBR_COMMANDS = 250

# Sanity limit for coordinates after SVG transforms and scaling.
# This catches malformed/pathological SVGs rather than producing a
# gigantic font full of invisible glyphs.
MAX_FONT_COORDINATE = 10_000_000.0


# --------------------------------------------------------------------------
# Name / codepoint resolution
# --------------------------------------------------------------------------

def load_mapping(path: str | Path | None) -> dict[str, str]:
    """Load a JSON or CSV filename-to-glyph mapping."""
    if path is None:
        return {}

    mapping_path = Path(path)

    if mapping_path.suffix.lower() == ".json":
        with mapping_path.open(encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return {str(key): str(value) for key, value in data.items()}

    mapping: dict[str, str] = {}

    with mapping_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if rows and rows[0][:2] == ["filename", "glyph"]:
        rows = rows[1:]

    for row in rows:
        if len(row) < 2 or not row[0].strip():
            continue
        mapping[row[0].strip()] = row[1].strip()

    return mapping


def resolve_glyph_value(value: str) -> tuple[str, int | None]:
    """Resolve a glyph value to (glyph_name, Unicode codepoint)."""
    if value in AGL2UV:
        return value, AGL2UV[value]

    if len(value) == 1:
        codepoint = ord(value)
        glyph_name = UV2AGL.get(codepoint) or f"uni{codepoint:04X}"
        return glyph_name, codepoint

    match = UNI_RE.match(value)
    if match:
        codepoint = int(match.group(1), 16)
        return f"uni{codepoint:04X}", codepoint

    # Bare, unencoded glyph name, e.g. a ligature.
    return value, None


def resolve_file(
    svg_file: Path,
    mapping: dict[str, str],
) -> tuple[str, int | None]:
    """Resolve the glyph name/codepoint for an SVG file."""
    stem = svg_file.stem

    if svg_file.name in mapping:
        return resolve_glyph_value(mapping[svg_file.name])

    if stem in mapping:
        return resolve_glyph_value(mapping[stem])

    return resolve_glyph_value(stem)


def is_recognized_stem(stem: str) -> bool:
    """
    True if `stem` (a filename without extension) is something we can
    confidently resolve on its own: a single character, a standard AGL
    glyph name ('space', 'comma', ...), or a uniXXXX/uXXXXXX codepoint.
    Anything else (glyph_004, sketch_final, IMG_0012, ...) is a "leftover"
    -- meaningless as a name, only useful as raw artwork -- and is a
    candidate for --fill-gaps alphabetical assignment.
    """
    if stem in AGL2UV:
        return True
    if len(stem) == 1:
        return True
    if UNI_RE.match(stem):
        return True
    return False


# Default fill order for --fill-gaps: plain lowercase/uppercase/digits
# first (the overwhelming majority of "I don't care about order" decorative
# fonts just want a-z filled in), then a few common accented letters and
# punctuation marks. Override with --alphabet if you want something else.
DEFAULT_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "éèàöüä"
    ".,;:!?"
)


def resolve_all_files(
    svg_files: Sequence[Path],
    mapping: dict[str, str],
    *,
    fill_gaps: bool,
    alphabet: Sequence[str],
) -> tuple[list[tuple[Path, str, int | None]], list[str]]:
    """
    Resolve every SVG file to (svg_file, glyph_name, codepoint).

    Pass 1: anything with an explicit --mapping entry, or a filename we
    can confidently recognize on its own (single char / AGL name / uniXXXX),
    is resolved immediately.

    Pass 2 (only if fill_gaps): whatever is left over -- filenames that
    didn't match anything -- fill in, in `svg_files` order (i.e.
    alphabetically by filename), whichever slots in `alphabet` were *not*
    already claimed by pass 1, in `alphabet` order. Once every gap is
    filled, any further leftover files are ignored (reported, not built).

    Returns (resolved, ignored_filenames).
    """
    resolved: list[tuple[Path, str, int | None]] = []
    leftovers: list[Path] = []

    for svg_file in svg_files:
        stem = svg_file.stem
        has_explicit_mapping = svg_file.name in mapping or stem in mapping
        if has_explicit_mapping or is_recognized_stem(stem):
            glyph_name, codepoint = resolve_file(svg_file, mapping)
            resolved.append((svg_file, glyph_name, codepoint))
        else:
            leftovers.append(svg_file)

    ignored: list[str] = []

    if fill_gaps and leftovers:
        claimed_names = {name for _, name, _ in resolved}
        claimed_codepoints = {
            cp for _, _, cp in resolved if cp is not None
        }

        gap_values = []
        for value in alphabet:
            glyph_name, codepoint = resolve_glyph_value(value)
            already_claimed = (
                glyph_name in claimed_names
                or (codepoint is not None and codepoint in claimed_codepoints)
            )
            if not already_claimed:
                gap_values.append(value)

        for value, svg_file in zip(gap_values, leftovers):
            glyph_name, codepoint = resolve_glyph_value(value)
            resolved.append((svg_file, glyph_name, codepoint))

        used = min(len(gap_values), len(leftovers))
        ignored = [f.name for f in leftovers[used:]]

    elif leftovers:
        # fill_gaps is off: leftover files are simply not built, same as
        # any other file that failed to resolve.
        ignored = [f.name for f in leftovers]

    return resolved, ignored


# --------------------------------------------------------------------------
# SVG -> outline extraction
# --------------------------------------------------------------------------

def flatten_shapes(svg_root: SVG) -> Iterator[Shape]:
    """Yield every Shape element in the parsed SVG."""
    for element in svg_root.elements():
        if isinstance(element, Shape) and not isinstance(element, SVG):
            yield element


def svg_canvas_size(
    svg_root: SVG,
    override_height: float | None = None,
) -> tuple[float, float]:
    """Return (width, height) of the drawing canvas in SVG user units."""
    if override_height is not None:
        width = getattr(svg_root, "width", None) or override_height
        return float(width), float(override_height)

    width = getattr(svg_root, "width", None)
    height = getattr(svg_root, "height", None)

    if not width or not height:
        viewbox = getattr(svg_root, "viewbox", None)
        if viewbox is not None:
            width, height = viewbox.width, viewbox.height

    if not width or not height:
        raise ValueError(
            "Could not determine SVG canvas size "
            "(no width/height/viewBox)"
        )

    return float(width), float(height)


def svg_baseline(svg_root: SVG, default: float) -> float:
    """Baseline y from a data-baseline attribute on the <svg> root, else `default`."""
    raw = (getattr(svg_root, "values", None) or {}).get("data-baseline")
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _finite_point(
    x: float,
    y: float,
    *,
    svg_file: Path,
) -> tuple[float, float]:
    """Validate a transformed/scaled font coordinate."""
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(
            f"{svg_file.name}: non-finite outline coordinate "
            f"({x!r}, {y!r})"
        )

    if abs(x) > MAX_FONT_COORDINATE or abs(y) > MAX_FONT_COORDINATE:
        raise ValueError(
            f"{svg_file.name}: outline coordinate is suspiciously large "
            f"({x:.3g}, {y:.3g})"
        )

    return x, y


def _tx(
    pt: Any,
    *,
    baseline_y: float,
    scale: float,
    x_shift: float,
    svg_file: Path,
) -> tuple[float, float]:
    """Transform an SVG point into font coordinates."""
    x = (float(pt.x) + x_shift) * scale
    y = (baseline_y - float(pt.y)) * scale
    return _finite_point(x, y, svg_file=svg_file)


def iter_subpaths(svg_path: SvgPath) -> Iterator[list[Any]]:
    """
    Split an SvgPath into independent subpaths.

    Each yielded list begins with Move and may contain Line/Cubic/Quadratic/
    Arc/Close segments.  The existing SVG geometry is not approximated here.
    """
    current: list[Any] = []

    for segment in svg_path.segments():
        if isinstance(segment, Move):
            if current:
                yield current
            current = [segment]
        else:
            if current:
                current.append(segment)

    if current:
        yield current


def draw_subpath(
    segments: Sequence[Any],
    *,
    baseline_y: float,
    scale: float,
    x_shift: float,
    pen: T2CharStringPen,
    svg_file: Path,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Draw one subpath into a T2CharStringPen.

    Returns (start_point, end_point) in font coordinates.
    """
    if not segments or not isinstance(segments[0], Move):
        raise ValueError("subpath does not begin with Move")

    start = _tx(
        segments[0].end,
        baseline_y=baseline_y,
        scale=scale,
        x_shift=x_shift,
        svg_file=svg_file,
    )

    current = start
    pen.moveTo(start)

    for segment in segments[1:]:
        if isinstance(segment, Line):
            current = _tx(
                segment.end,
                baseline_y=baseline_y,
                scale=scale,
                x_shift=x_shift,
                svg_file=svg_file,
            )
            pen.lineTo(current)

        elif isinstance(segment, CubicBezier):
            control1 = _tx(
                segment.control1,
                baseline_y=baseline_y,
                scale=scale,
                x_shift=x_shift,
                svg_file=svg_file,
            )
            control2 = _tx(
                segment.control2,
                baseline_y=baseline_y,
                scale=scale,
                x_shift=x_shift,
                svg_file=svg_file,
            )
            current = _tx(
                segment.end,
                baseline_y=baseline_y,
                scale=scale,
                x_shift=x_shift,
                svg_file=svg_file,
            )
            pen.curveTo(control1, control2, current)

        elif isinstance(segment, QuadraticBezier):
            control = _tx(
                segment.control,
                baseline_y=baseline_y,
                scale=scale,
                x_shift=x_shift,
                svg_file=svg_file,
            )
            current = _tx(
                segment.end,
                baseline_y=baseline_y,
                scale=scale,
                x_shift=x_shift,
                svg_file=svg_file,
            )
            pen.qCurveTo(control, current)

        elif isinstance(segment, Arc):
            for cubic in segment.as_cubic_curves():
                control1 = _tx(
                    cubic.control1,
                    baseline_y=baseline_y,
                    scale=scale,
                    x_shift=x_shift,
                    svg_file=svg_file,
                )
                control2 = _tx(
                    cubic.control2,
                    baseline_y=baseline_y,
                    scale=scale,
                    x_shift=x_shift,
                    svg_file=svg_file,
                )
                current = _tx(
                    cubic.end,
                    baseline_y=baseline_y,
                    scale=scale,
                    x_shift=x_shift,
                    svg_file=svg_file,
                )
                pen.curveTo(control1, control2, current)

        elif isinstance(segment, Close):
            # Type 2 implicitly closes the current contour when a new
            # moveto occurs or when endchar is reached.  There is no need
            # for a separate closepath operator in the subroutine.
            current = start

    return start, current


def _split_program(
    program: Sequence[Any],
    max_commands: int,
) -> list[list[Any]]:
    """
    Split a T2 program into command groups.

    `program` consists of numeric operands followed by string operators.
    The final endchar is omitted from the returned groups.
    """
    groups: list[list[Any]] = []
    current: list[Any] = []
    command_count = 0

    for item in program:
        if isinstance(item, str):
            if item == "endchar":
                continue

            current.append(item)
            command_count += 1

            if command_count >= max_commands:
                groups.append(current)
                current = []
                command_count = 0
        else:
            current.append(item)

    if current:
        groups.append(current)

    return groups


def _subr_bias(count: int) -> int:
    """Return the CFF Type 2 local-subroutine bias."""
    if count < 1240:
        return 107
    if count < 33900:
        return 1131
    return 32768


def _make_subroutine(
    program: Sequence[Any],
    *,
    private: Any,
    global_subrs: Any,
) -> T2CharString:
    """Create a CFF local subroutine from a Type 2 program."""
    return T2CharString(
        program=[*program, "return"],
        private=private,
        globalSubrs=global_subrs,
    )


def build_cff_charstrings(
    *,
    svg_paths: Sequence[SvgPath],
    advance: float,
    canvas_h: float,
    baseline_y: float,
    scale: float,
    x_shift: float,
    svg_file: Path,
    private: Any,
    global_subrs: Any,
    subr_commands: int,
) -> tuple[T2CharString, list[T2CharString], int, int]:
    """
    Convert SVG paths into a glyph charstring plus local subroutines.

    The actual geometry lives in local subroutines. The glyph charstring
    contains only width, moves to contour starts, and callsubr operators.

    Returns:
        (glyph_charstring, local_subroutines, path_count, command_count)
    """
    raw_subr_programs: list[list[Any]] = []
    contour_starts_and_ends: list[
        tuple[tuple[float, float], tuple[float, float], list[list[Any]]]
    ] = []

    total_commands = 0

    for svg_path in svg_paths:
        for subpath in iter_subpaths(svg_path):
            pen = T2CharStringPen(
                None,
                None,
                roundTolerance=0.5,
            )

            start, end = draw_subpath(
                subpath,
                baseline_y=baseline_y,
                scale=scale,
                x_shift=x_shift,
                pen=pen,
                svg_file=svg_file,
            )

            program = pen.getCharString().program

            # Ignore a completely empty subpath.
            if not program:
                continue

            # T2CharStringPen emits the initial rmoveto. Remove it from
            # the subroutine because the glyph charstring needs to move
            # to the contour start itself. This also lets continuation
            # subroutines resume at the current point.
            if len(program) >= 3 and program[2] == "rmoveto":
                drawing_program = list(program[3:])
            else:
                drawing_program = list(program)

            chunks = _split_program(
                drawing_program,
                max_commands=max(1, subr_commands),
            )

            if not chunks:
                continue

            contour_starts_and_ends.append((start, end, chunks))
            raw_subr_programs.extend(chunks)

            total_commands += sum(
                1 for item in drawing_program if isinstance(item, str)
            )

    local_subrs = [
        _make_subroutine(
            program,
            private=private,
            global_subrs=global_subrs,
        )
        for program in raw_subr_programs
    ]

    bias = _subr_bias(len(local_subrs))

    # IMPORTANT: the optional leading glyph width in a Type 2 CharString is
    # encoded as one extra numeric operand in front of the very first
    # stack-clearing operator's own arguments -- it is never a standalone
    # operator call. `[advance, "hmoveto"]` would be wrong here: "hmoveto"
    # is a real drawing command (move the pen `dx` units along X), so that
    # would execute an actual move-by-`advance` before any contour is
    # drawn, permanently shifting every subsequent coordinate to the right
    # by the glyph's own advance width. Instead, prepend `advance` to the
    # first contour's own rmoveto operands (or, if the glyph has no
    # contours at all -- e.g. a blank/space glyph -- to a bare endchar).
    glyph_program: list[Any] = []
    current_x = 0.0
    current_y = 0.0
    subr_index = 0
    first_moveto = True

    for start, _end, chunks in contour_starts_and_ends:
        start_x, start_y = start

        # Move from the end of the previous contour to this contour's
        # start. Type 2 rmoveto uses relative coordinates.
        dx = start_x - current_x
        dy = start_y - current_y
        if first_moveto:
            glyph_program.extend([advance, dx, dy, "rmoveto"])
            first_moveto = False
        else:
            glyph_program.extend([dx, dy, "rmoveto"])

        for _chunk in chunks:
            glyph_program.extend([
                subr_index - bias,
                "callsubr",
            ])
            subr_index += 1

        # We deliberately use the known contour endpoint instead of
        # querying a private Pen state after closePath().
        current_x, current_y = _end

    if first_moveto:
        # No contours at all (e.g. a blank/space glyph): the width still
        # needs to be encoded, this time as the lone extra operand in
        # front of endchar.
        glyph_program.append(advance)

    glyph_program.append("endchar")

    glyph = T2CharString(
        program=glyph_program,
        private=private,
        globalSubrs=global_subrs,
    )

    return (
        glyph,
        local_subrs,
        len(contour_starts_and_ends),
        total_commands,
    )


# --------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------

def build_font(
    input: str,
    output: str|None ="output.otf",
    mapping: str | None = None,
    upm: int |None= None,
    descent: int = 0,
    family: str|None = None,
    style: str = "Regular",
    version: str = "1.0",
    scale: float = 1.0,
    svg_units_per_em: float | None = None,
    baseline: float | None = None,
    width_mode: Literal["canvas","bbox"] = "canvas",
    lsb: float = 40,
    rsb: float = 40,
    advance_width: float | None = None,
    subr_commands: int = DEFAULT_SUBR_COMMANDS,
    fill_gaps: bool = True,
    alphabet: str | None = None,
    fit_scale:bool = False
) -> None:
    """
    Compile a folder of per-glyph SVG files into an OTF font.

    Args:
        input: Folder containing a.svg, b.svg, comma.svg, etc.
        output: Output .otf path.
        mapping: Optional JSON or CSV filename-to-glyph mapping.
        upm: Units per em.
        descent: Distance from baseline to bottom of em square.
        family: Font family name.
        style: Style/subfamily name.
        version: Font version string.
        scale: SVG-unit-to-font-unit scale. Ignored when
            svg_units_per_em is set.
        svg_units_per_em: SVG canvas height representing one em.
        baseline: SVG y-coordinate of the baseline.
        width_mode: Either "bbox" or "canvas".
        lsb: Left side bearing in bbox mode.
        rsb: Right side bearing in bbox mode.
        advance_width: Force one fixed advance width for every glyph.
        subr_commands: Maximum number of Type 2 drawing operators stored
            in one CFF local subroutine. Lower values make glyph
            charstrings smaller; they do not simplify the artwork.
        fill_gaps: For decorative sets where filenames don't matter: after
            resolving every file we *can* recognize (single characters,
            AGL names like "comma", uniXXXX), assign the leftover,
            unrecognized files to whichever slots in `alphabet` are still
            unclaimed, in alphabet order, using leftover files in
            filename-sorted order. Leftover files beyond however many
            gaps exist are ignored. Explicit --mapping entries always
            count as "recognized" and are never overwritten by this.
        alphabet: Comma-separated list of glyph values (single characters
            and/or AGL names) defining the fill order for --fill-gaps.
            Defaults to a-z, A-Z, 0-9, a few accented letters, then basic
            punctuation.
    """
    if width_mode not in {"bbox", "canvas"}:
        raise ValueError(
            f"width_mode must be 'bbox' or 'canvas', got {width_mode!r}"
        )

   
    if descent < 0:
        raise ValueError("descent must be non-negative")

    if subr_commands <= 0:
        raise ValueError("subr_commands must be positive")

    in_dir = Path(input)
    svg_files = sorted(in_dir.glob("*.svg"))

    if not svg_files:
        sys.exit(f"No .svg files found in {in_dir}")

    glyph_mapping = load_mapping(mapping)

    if alphabet is None:
        alphabet_values = list(DEFAULT_ALPHABET)
    elif isinstance(alphabet, str):
        # python-fire only leaves this as a plain string if there was no
        # comma in the CLI value at all (e.g. a single glyph).
        alphabet_values = [v.strip() for v in alphabet.split(",") if v.strip()]
    else:
        # Fire auto-splits comma-containing CLI args into a tuple/list.
        alphabet_values = [str(v).strip() for v in alphabet if str(v).strip()]

    resolved_files, ignored_filenames = resolve_all_files(
        svg_files,
        glyph_mapping,
        fill_gaps=fill_gaps,
        alphabet=alphabet_values,
    )

    if ignored_filenames:
        note = "ignored" if fill_gaps else "unrecognized, skipped"
        print(
            f"{len(ignored_filenames)} file(s) {note} "
            f"(pass --mapping or rename to fill them in): "
            f"{', '.join(ignored_filenames)}",
            file=sys.stderr,
        )

    if family is None:
        family = Path(output).stem.capitalize()

    



    skipped: list[str] = list(ignored_filenames)

    tot = len(resolved_files)
    for jj, (svg_file, glyph_name, codepoint) in enumerate(resolved_files):

        print(f"{glyph_name} ({codepoint}) {jj}/{tot}")

        if glyph_name is None:
            skipped.append(svg_file.name)
            continue
        

        try:
            svg_root = SVG.parse(
                str(svg_file),
                reify=True,
            )

            canvas_w, canvas_h = svg_canvas_size(
                svg_root,
                svg_units_per_em,
            )

   
          

            baseline_y = (
                baseline
                if baseline is not None
                else svg_baseline(svg_root, canvas_h)
            )

            effective_scale = scale

            if svg_units_per_em is not None:
                # upm may still be None here (it defaults after the loop)
                effective_scale = (
                    upm if upm is not None else svg_units_per_em
                ) / svg_units_per_em

            shapes = list(flatten_shapes(svg_root))
            paths: list[SvgPath] = []

            min_x = min_y = float("inf")
            max_x = max_y = float("-inf")

            for shape in shapes:
                path = SvgPath(shape)
                path.reify()
                paths.append(path)

                bbox = path.bbox()
                if bbox is None:
                    continue

                x0, y0, x1, y1 = (
                    float(value)
                    for value in bbox
                )

                values = (x0, y0, x1, y1)

                if not all(math.isfinite(v) for v in values):
                    raise ValueError(
                        f"non-finite SVG path bounds: {bbox}"
                    )

                min_x = min(min_x, x0)
                max_x = max(max_x, x1)
                min_y = min(min_y, y0)
                max_y = max(max_y, y1)

            has_ink = min_x != float("inf")

            if width_mode == "bbox" and has_ink:
                ink_w = max_x - min_x

                # fit_scale: with a forced advance_width, shrink THIS glyph
                # (uniformly, so its height shrinks too) if it doesn't fit.
                if (
                    fit_scale
                    and advance_width is not None
                    and advance_width > lsb + rsb
                    and ink_w * effective_scale + lsb + rsb > advance_width
                ):
                    effective_scale *= (
                        (advance_width - lsb - rsb)
                        / (ink_w * effective_scale)
                    )

                advance = (
                    advance_width
                    if advance_width is not None
                    else round(ink_w * effective_scale + lsb + rsb)
                )
                x_shift = -min_x + lsb / effective_scale

            else:
                x_shift = 0.0

                if (
                    fit_scale
                    and advance_width is not None
                    and canvas_w * effective_scale > advance_width
                ):
                    effective_scale *= (
                        advance_width / (canvas_w * effective_scale)
                    )

                advance = (
                    advance_width
                    if advance_width is not None
                    else round(canvas_w * effective_scale)
                )

            if not math.isfinite(advance) or advance < 0:
                raise ValueError(
                    f"invalid advance width: {advance!r}"
                )

            # Build the font first so we can attach the local CFF subroutines
            # to its Private dictionary after setupCFF().
            #
            # We temporarily store the SVG paths and build their charstring
            # after setupCFF below.  The actual work is performed immediately
            # after the font builder is initialized outside this per-glyph
            # loop, so the implementation uses a deferred list.
            #
            # To keep the rest of the build straightforward, stash the
            # converted geometry parameters here.
            #
            # (The actual conversion happens below once CFF Private exists.)
            if "_pending" not in locals():
                _pending: list[
                    tuple[
                        str,
                        int | None,
                        list[SvgPath],
                        float,
                        float,
                        float,
                        float,
                        Path,
                    ]
                ] = []

            _pending.append(
                (
                    glyph_name,
                    codepoint,
                    paths,
                    advance,
                    canvas_h,
                    baseline_y,
                    effective_scale,
                    svg_file,
                )
            )


            # Store the shift on a side table.
            if "_pending_shifts" not in locals():
                _pending_shifts: dict[str, float] = {}

            _pending_shifts[glyph_name] = x_shift

        except Exception as exc:
            print(
                f"  ! failed to prepare {svg_file.name}: {exc}",
                file=sys.stderr,
            )
            skipped.append(svg_file.name)


    if "_pending" not in locals() or not _pending:
        sys.exit(
            "No glyphs were successfully prepared -- "
            "check filenames/mapping."
        )

    if svg_units_per_em is None:
                svg_units_per_em = canvas_h

    if upm is None:
        upm = round(_pending[0][4])  # canvas height of the first glyph

    order: list[str] = [".notdef"]
    cmap: dict[int, str] = {}
    charstrings: dict[str, T2CharString] = {}
    advance_widths: dict[str, float] = {".notdef": upm // 2}


    # .notdef: simple box.
    notdef_pen = T2CharStringPen(
        advance_widths[".notdef"],
        None,
    )

    margin = upm // 10
    box = upm // 2 - margin

    notdef_pen.moveTo((margin, 0))
    notdef_pen.lineTo((box, 0))
    notdef_pen.lineTo((box, upm - descent))
    notdef_pen.lineTo((margin, upm - descent))
    notdef_pen.closePath()

    charstrings[".notdef"] = notdef_pen.getCharString()

    # ---- assemble the font ----
    font_builder = FontBuilder(upm, isTTF=False)

    # setupCFF needs the glyph dictionary now, but the glyph charstrings
    # themselves can reference the Private local subroutines created after
    # setupCFF.
    font_builder.setupGlyphOrder([".notdef"])

    # Character map/order is finalized after conversion.
    font_builder.setupCharacterMap({})

    font_builder.setupCFF(
        family,
        {
            "FontName": family.replace(" ", ""),
            "FullName": f"{family} {style}".strip(),
            "Weight": style,
        },
        charstrings,
        {},
    )

    cff = font_builder.font["CFF "].cff
    top_dict = cff.topDictIndex[0]

    # A SubrsIndex is required here; assigning a plain Python list does not
    # produce a valid CFF Private dictionary.
    local_subrs = SubrsIndex(
        globalSubrs=cff.GlobalSubrs,
        private=top_dict.Private,
    )
    local_subrs.items = []
    top_dict.Private.Subrs = local_subrs

    built_count = 0
    total_subrs = 0
    largest_raw_program = 0
    largest_glyph_program = 0

    for (
        glyph_name,
        codepoint,
        paths,
        advance,
        canvas_h,
        baseline_y,
        effective_scale,
        svg_file,
    ) in _pending:
        x_shift = _pending_shifts[glyph_name]

        try:
            glyph_charstring, glyph_subrs, contour_count, command_count = (
                build_cff_charstrings(
                    svg_paths=paths,
                    advance=advance,
                    canvas_h=canvas_h,
                    baseline_y=baseline_y,
                    scale=effective_scale,
                    x_shift=x_shift,
                    svg_file=svg_file,
                    private=top_dict.Private,
                    global_subrs=cff.GlobalSubrs,
                    subr_commands=subr_commands,
                )
            )

            raw_size = sum(
                len(subr.program)
                for subr in glyph_subrs
            )

            largest_raw_program = max(
                largest_raw_program,
                raw_size,
            )

            largest_glyph_program = max(
                largest_glyph_program,
                len(glyph_charstring.program),
            )

            start_index = len(local_subrs.items)
            local_subrs.items.extend(glyph_subrs)

            # The subroutines are appended in exactly the same order used
            # when building the glyph charstring, so their indices remain
            # valid.  build_cff_charstrings used a bias based on the final
            # number of subroutines for that glyph, but the global index is
            # not yet known.  Therefore we need to rebuild the glyph program
            # with the final global local-subroutine indices.
            #
            # The clean solution is to use global subroutine numbering
            # across all glyphs and rebuild every call operand after all
            # subroutines have been collected.  For simplicity, we instead
            # keep each glyph's subroutines in a separate contiguous range
            # and rewrite its operands here.
            #
            # CFF local subr bias depends only on the total number of local
            # subroutines, so the final bias is known after all glyphs have
            # been collected.  We temporarily store the un-biased program.
            charstrings[glyph_name] = glyph_charstring

            if glyph_name not in order:
                order.append(glyph_name)

            if codepoint is not None:
                cmap[codepoint] = glyph_name

            advance_widths[glyph_name] = advance
            built_count += 1

            print(
                f"  {svg_file.name}: "
                f"{contour_count} contours, "
                f"{command_count} drawing ops, "
                f"{len(glyph_subrs)} CFF subrs",
                file=sys.stderr,
            )

        except Exception as exc:
            print(
                f"  ! failed to build glyph {glyph_name!r} "
                f"from {svg_file.name}: {exc}",
                file=sys.stderr,
            )
            continue

    if built_count == 0:
        sys.exit(
            "No glyphs were successfully built -- "
            "check filenames/mapping."
        )

    # The glyph programs created above use local subroutine indices starting
    # at zero for each glyph.  We need to rewrite them to their final global
    # positions.  Since each glyph's subroutines were appended contiguously,
    # walk the glyphs again and assign the final index range.
    #
    # Rebuild all glyph programs from the original geometry would be wasteful,
    # so collect the actual subroutine ranges by replaying the same conversion.
    #
    # To avoid this bookkeeping complexity, reconstruct the glyph charstrings
    # from their subroutine calls and assign ranges in glyph order.
    #
    # Each glyph's program has callsubr operands of the form
    #   local_index - local_bias
    # where local_bias was calculated using only that glyph's subr count.
    # Recover the local index, add the glyph's global start, then apply the
    # final global bias.
    global_subr_count = len(local_subrs.items)
    global_bias = _subr_bias(global_subr_count)

    global_start = 0

    for glyph_name in order:
        if glyph_name == ".notdef":
            continue

        charstring = charstrings[glyph_name]

        # Find how many calls this glyph has and recover its old bias from
        # the number of calls.  Every call operand was generated against the
        # number of subroutines belonging to that glyph.
        calls = [
            index
            for index, item in enumerate(charstring.program)
            if item == "callsubr"
        ]

        glyph_subr_count = len(calls)

        # A contour may have been split into several subroutines, so the
        # number of calls is exactly the number of local subroutines used.
        old_bias = _subr_bias(glyph_subr_count)

        program = list(charstring.program)

        for index in calls:
            operand = program[index - 1]
            local_index = int(operand) + old_bias
            global_index = global_start + local_index
            program[index - 1] = global_index - global_bias

        charstrings[glyph_name] = T2CharString(
            program=program,
            private=top_dict.Private,
            globalSubrs=cff.GlobalSubrs,
        )

        global_start += glyph_subr_count

    # The above relies on every successfully built glyph having all of its
    # subroutines appended.  Recompute the total from the glyph call counts.
    if global_start != global_subr_count:
        raise RuntimeError(
            "Internal CFF subroutine bookkeeping error: "
            f"used {global_start}, have {global_subr_count}"
        )

    # Re-install the finalized glyph dictionary.
    top_dict.CharStrings.charStrings = charstrings

    # CRITICAL: FontBuilder.setupCFF() snapshotted topDict.charset from
    # the glyph order *at the time it was called* (just [".notdef"],
    # since the real glyphs weren't built yet). That snapshot is what
    # actually controls which charstrings get written into the compiled
    # CFF INDEX -- it is a separate list from font_builder's own glyph
    # order tracking. Without updating it here, the saved font ends up
    # with maxp/hmtx/cmap all agreeing on `len(order)` glyphs while the
    # embedded CFF program only contains ".notdef". fontTools happily
    # writes that inconsistent file out (and even fails to *read* it
    # back correctly -- see the hmtx IndexError), and strict embedders/
    # subsetters like Typst's correctly reject it as "malformed font".
    top_dict.charset = order

    # Glyph order and cmap.
    font_builder.setupGlyphOrder(order)
    font_builder.setupCharacterMap(cmap)

    ascent = upm - descent

    font_builder.setupHorizontalMetrics(
        {
            name: (advance_widths[name], 0)
            for name in order
        }
    )

    font_builder.setupHorizontalHeader(
        ascent=ascent,
        descent=-descent,
    )

    name_strings = {
        "familyName": family,
        "styleName": style,
        "uniqueFontIdentifier": f"{family}-{style}:{version}",
        "fullName": f"{family} {style}".strip(),
        "psName": (
            family.replace(" ", "")
            + "-"
            + style.replace(" ", "")
        ),
        "version": f"Version {version}",
    }

    font_builder.setupNameTable(name_strings)

    font_builder.setupOS2(
        sTypoAscender=ascent,
        sTypoDescender=-descent,
        usWinAscent=ascent,
        usWinDescent=descent,
    )

    font_builder.setupPost()

    out_path = Path(output)
    ensure_dir(out_path)
    font_builder.save(str(out_path))

    print(
        f"Built {built_count} glyphs"
        + (
            f", skipped {len(skipped)}: {', '.join(skipped)}"
            if skipped
            else ""
        )
    )
    print(
        f"CFF local subroutines: {global_subr_count}; "
        f"largest raw glyph program: {largest_raw_program} items; "
        f"largest glyph charstring: {largest_glyph_program} items"
    )
    print(
        f"Saved {out_path} ({out_path.stat().st_size} bytes)"
    )



def typst_specimen(font_name:str,outdir:str,font_path:str):
    import typst  # lazy: only needed for the specimen PDF
    ensure_dir(outdir)
    with open(Path(ROOT,"specimen.typ"),"rb") as f:
        typst.compile(
                    f.read(),
                    output=Path(outdir, f"{font_name}_specimen.pdf"),
                    font_paths=[font_path],
                    sys_inputs={"font":font_name},
                )
        

if __name__ == "__main__":
    fire.Fire(build_font)