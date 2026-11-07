from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageTk
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]

def asset_path(name: str) -> Path:
    path = APP_DIR / "assets" / name
    if not path.is_file() and getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "assets" / name
    return path

def get_add_photo_icon(size: int = 20) -> ImageTk.PhotoImage:
    assets_path = asset_path("add_photo.png")
    if assets_path.is_file():
        try:
            with Image.open(assets_path) as img:
                return ImageTk.PhotoImage(img.resize((size, size), Image.Resampling.LANCZOS))
        except Exception:
            pass
    # Fallback clean dynamic PIL rendering
    scale = 4
    dim = size * scale
    img = Image.new("RGBA", (dim, dim), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (55, 55, 55, 255)
    d.rounded_rectangle(
        [scale * 1.5, scale * 2.5, scale * 17.5, scale * 18.5],
        radius=scale * 2.5,
        outline=color,
        width=int(scale * 1.8),
    )
    d.ellipse([scale * 5, scale * 5.5, scale * 8.5, scale * 9], fill=color)
    d.polygon(
        [
            (scale * 3.5, scale * 17),
            (scale * 9, scale * 10.5),
            (scale * 13, scale * 15),
            (scale * 16, scale * 17),
        ],
        fill=color,
    )
    d.ellipse(
        [scale * 12.5, scale * 12.5, scale * 21.5, scale * 21.5],
        fill=(255, 255, 255, 255),
        outline=color,
        width=int(scale * 1.6),
    )
    d.line([(scale * 17, scale * 14.5), (scale * 17, scale * 19.5)], fill=color, width=int(scale * 1.8))
    d.line([(scale * 14.5, scale * 17), (scale * 19.5, scale * 17)], fill=color, width=int(scale * 1.8))
    return ImageTk.PhotoImage(img.resize((size, size), Image.Resampling.LANCZOS))

