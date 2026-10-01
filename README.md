# What is this?

Code to extract svgs from a png and create a font from the glyphs.

For a simple example, see the sample code below.

It assumes a folder structure with `project/font_name/raw` for the image file, `project/font_name/svg` for the svg, and so on.

```python


from pathlib import Path

from font_maker.harvest import build_glyphs, segment
from font_maker.svg2otf import build_font, typst_specimen


def cmd_build_font(input:str):
    ROOT = Path(__file__).parent

    project = Path(input).stem

    output = Path("projects",project,"otf", project + ".otf")
    build_font(
        input=Path("projects",input,"svg"),
        output=output,

        upm=300,
        svg_units_per_em=400,
        descent=20,
        advance_width=300,
        fit_scale=True
    )

    typst_specimen(Path(output).stem, Path(ROOT, f"projects/{project}/specimens"), ROOT)

def cmd_segment(png:str,project:str):
    segment(input=Path("raw",png),out=Path("projects",project,"raw"),thresh=0.9)

def cmd_build_glyphs(input:str):
    p = Path("projects",input)
    build_glyphs(
        work=Path(p,"raw"),
        out=Path(p,"svg")
    )

if __name__ == "__main__":
    cmd_build_glyphs("exam_1910")
    cmd_build_font("exam_1910")
    

    ```
