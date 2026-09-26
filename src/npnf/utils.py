import hashlib
import io
import string
from collections.abc import Iterable
from pathlib import Path

from matplotlib.axes import Axes
from PIL import Image


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fig2img(
    fig, dpi: int = 200, image_format: str = "jpg", is_transparent: bool = False
) -> Image.Image:
    with io.BytesIO() as buf:
        fig.savefig(
            buf,
            dpi=dpi,
            bbox_inches="tight",
            format=image_format,
            transparent=is_transparent,
        )
        buf.seek(0)
        with Image.open(buf) as img:
            return img.copy()


def add_panel_labels(
    panels: Iterable[Axes], offset_points: tuple[float, float] = (6, -6)
) -> None:
    """Write A, B, ... inside the top left corner of each panel, in order.

    ``offset_points`` moves the top left of each label from the top left corner
    of its panel.
    """
    for letter, panel in zip(string.ascii_uppercase, panels, strict=False):
        panel.annotate(
            letter,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=offset_points,
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize="large",
            fontweight="bold",
        )
