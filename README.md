# svg2otf

Turn a folder of per-letter SVGs into one `.otf` font, with no manual
clicking-around. Pure Python (`fontTools` + `svgelements`) — no FontForge
install needed.

## Install

```bash
pip install fonttools svgelements
```

## Quick start

```
glyphs/
  a.svg
  b.svg
  c.svg
  ...
  comma.svg
  space.svg
```

```bash
python3 svg2otf.py glyphs -o MyFont-Regular.otf \
    --family "My Font" --style Regular
```

## Naming your files

- `a.svg` → the letter **a**. Same for any single character.
- Standard punctuation names work too: `space.svg`, `comma.svg`,
  `period.svg`, `hyphen.svg`, `exclam.svg`, `question.svg`, `at.svg`,
  `ampersand.svg` — anything in the Adobe Glyph List.
- `uni0041.svg` also works if you'd rather use codepoints.
- **Uppercase on macOS/Windows:** since `A.svg` and `a.svg` collide on a
  case-insensitive filesystem, name the uppercase file something else
  (e.g. `Aup.svg`) and supply a `--mapping mapping.json` file:
  ```json
  { "Aup.svg": "A", "Bup.svg": "B" }
  ```
  (CSV works too: `filename,glyph` per line.)

## How coordinates map to the font

Each SVG's own canvas (its `viewBox`, or `width`/`height`) is the
drawing area for that glyph. By default:

- the **bottom** of the canvas = the baseline
- 1 SVG unit = 1 font design unit (`--scale 1.0`)

If you'd rather say "my canvas is 100 units tall and that means one
em", use `--svg-units-per-em 100` and the scale is computed for you.

Advance width defaults to **ink bounding box + side bearings**
(`--lsb`, `--rsb`, default 40/40 each) — same idea as Glyphr Studio's
"bearings: left/right" fields, just automatic. Use
`--width-mode canvas` if you'd rather every glyph's advance just be
its SVG canvas width, or `--advance-width 600` to force one fixed
width on everything (monospace-style).

## All settings

```
python3 svg2otf.py --help
```

Key ones:
- `--upm` — units per em (default 1000)
- `--descent` — baseline-to-bottom-of-em distance (default 200, so
  default ascent works out to 800)
- `--family`, `--style`, `--version`
- `--scale`, `--svg-units-per-em`, `--baseline`
- `--width-mode {bbox,canvas}`, `--lsb`, `--rsb`, `--advance-width`

## Notes on messy/hand-drawn source art

- Multiple `<path>`/`<rect>`/`<circle>`/etc. elements in one file are
  all merged into that glyph (e.g. a dot + stem for "i", or a rect and
  a circle unioned for a bowl shape) — see `test_glyphs/b.svg` for an
  example.
- Counters (holes, like inside "o") work automatically as long as your
  SVG editor exported the inner and outer contours with opposite
  winding direction (which Illustrator/Inkscape/Figma do by default
  for compound paths).
- Cubic Beziers are kept as exact cubic curves in the OTF (CFF
  outlines) — no lossy quadratic conversion, so sloppy/organic curves
  stay faithful to your source art.
- Open subpaths get auto-closed with a straight line back to their
  start.

## Empty/whitespace glyphs

A file with no drawable shapes (e.g. an empty `space.svg`) is valid —
it just produces an empty outline sized by the canvas width (or
`--advance-width` if you set one).
