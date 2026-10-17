"""Recorta el logo y genera PNG/ICO de la app y la extensión."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PNG = REPO / "assets" / "ilsa-logo.png"
ICONS = REPO / "extension" / "icons"
ICO = REPO / "packaging" / "ilsa.ico"


def square_pad(im: Image.Image) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    alpha = im.split()[-1]
    bbox = alpha.point(lambda p: 255 if p > 8 else 0).getbbox()
    if not bbox:
        return im
    left, top, right, bottom = bbox
    pad = int(0.08 * max(right - left, bottom - top))
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(im.width, right + pad)
    bottom = min(im.height, bottom + pad)
    cropped = im.crop((left, top, right, bottom))
    side = max(cropped.width, cropped.height)
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2), cropped)
    return out


def main() -> None:
    if not PNG.is_file():
        raise SystemExit(f"Falta {PNG}")
    im = Image.open(PNG).convert("RGBA")
    cropped = square_pad(im)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    ICONS.mkdir(parents=True, exist_ok=True)
    master = cropped.resize((512, 512), Image.Resampling.LANCZOS)
    master.save(PNG, "PNG", optimize=True)
    for size in (16, 48, 128):
        icon = cropped.resize((size, size), Image.Resampling.LANCZOS)
        icon.save(ICONS / f"icon{size}.png", "PNG", optimize=True)
    ico_sizes = [(s, s) for s in (16, 32, 48, 64, 128, 256)]
    master.save(ICO, sizes=ico_sizes)
    print(f"OK {PNG}")
    print(f"OK {ICO}")
    print(f"OK {ICONS}")


if __name__ == "__main__":
    main()
