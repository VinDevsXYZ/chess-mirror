"""Render the app icon (a knight facing its own reflection) into ChessMirror.icns."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

FONT = "/System/Library/Fonts/Apple Symbols.ttf"
KNIGHT = "♞"  # black chess knight


def render(size: int = 1024) -> Image.Image:
    pad = int(size * 0.09)
    radius = int(size * 0.2)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((pad, pad, size - pad, size - pad), radius, fill=255)

    # plate: vertical gradient
    grad = Image.new("RGBA", (size, size))
    top, bottom = (48, 61, 94), (17, 21, 36)
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / size
        gd.line((0, y, size, y), fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))

    # faint board squares on the lower half (own layer, blended)
    board = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bd = ImageDraw.Draw(board)
    cell = (size - 2 * pad) / 8
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 0:
                x0, y0 = pad + c * cell, pad + r * cell
                bd.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(255, 255, 255, 22))
    grad.alpha_composite(board)

    # knights facing each other across a mirror line
    font = ImageFont.truetype(FONT, int(size * 0.40))
    glyph = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(glyph)
    bbox = d.textbbox((0, 0), KNIGHT, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = size // 2 + int(size * 0.035) - bbox[0]
    y = int(size * 0.62) - h - bbox[1]
    d.text((x, y), KNIGHT, font=font, fill=(242, 193, 78, 255))
    reflection = ImageOps.mirror(glyph)
    reflection.putalpha(reflection.getchannel("A").point(lambda v: int(v * 0.38)))
    shadow = glyph.filter(ImageFilter.GaussianBlur(size * 0.01))
    shadow.putalpha(shadow.getchannel("A").point(lambda v: int(v * 0.45)))
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    layer.alpha_composite(reflection)
    layer.alpha_composite(shadow, (int(size * 0.006), int(size * 0.012)))
    layer.alpha_composite(glyph)
    ImageDraw.Draw(layer).line((size // 2, int(size * 0.2), size // 2, int(size * 0.66)),
                               fill=(255, 255, 255, 70), width=max(1, size // 200))
    grad.alpha_composite(layer)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)
    del w
    return img


def main(out: str) -> None:
    base = render(1024)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "ChessMirror.iconset"
        iconset.mkdir()
        for s in (16, 32, 128, 256, 512):
            base.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            base.resize((s * 2, s * 2), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", out], check=True)
    base.save(Path(out).with_suffix(".png"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "packaging/ChessMirror.icns")
