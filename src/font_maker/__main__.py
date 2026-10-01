import Fire

from font_maker.harvest import build_glyphs, segment
from font_maker.svg2otf import build_font


class CLI:
    """Command-line interface for handwritten glyph harvesting."""

    segment = staticmethod(segment)
    build_glyphs = staticmethod(build_glyphs)
    build_font = staticmethod(build_font)


if __name__ == '__main__':
    Fire.fire(CLI)
