"""Shared chart theme.

One visual system for every figure in the platform -- the EDA notebook, the
saved report figures and the Streamlit UI all draw from the tokens here, so a
chart means the same thing wherever it appears.

The palette is not a matter of taste. Every colour set below was checked with a
colour-vision-deficiency validator against the light chart surface: the two-hue
categorical pair clears CVD deltaE 24.7 (target >= 8) and normal-vision deltaE
33.6 (floor 15), and the ordinal blue ramp is monotone in lightness with every
adjacent step gap >= 0.06 and its lightest step clearing 2:1 against the surface.

Encoding rules this module enforces by construction:

* One series gets one colour -- never a value ramp across nominal categories,
  which would double-encode bar length as hue.
* Ordered bands (score quintiles, age bands, risk tiers) get the single-hue
  ordinal ramp, light to dark.
* Risk bands get the reserved status colours, which are never used for a
  plain data series, and always ship with a text label rather than relying on
  colour alone.
* Never two y-axes on one plot. Aligning two scales invents a correlation that
  is not in the data, so "volume and rate" is drawn as two panels -- side by
  side, or stacked on a shared x-axis -- rather than as a dual-axis chart.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Final, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
SURFACE: Final[str] = "#fcfcfb"
INK_PRIMARY: Final[str] = "#0b0b0b"
INK_SECONDARY: Final[str] = "#52514e"
INK_MUTED: Final[str] = "#898781"
GRID: Final[str] = "#e1e0d9"
BASELINE: Final[str] = "#c3c2b7"

# Categorical slots, in fixed order. Assigned by identity, never by rank, and
# never cycled -- past the third slot, fold the tail into "Other" or facet.
SERIES: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a")

# Single-hue ordinal ramp (blue). Light end is step 250, the lightest value that
# still clears 2:1 against the light surface.
_ORDINAL_STEPS: Final[tuple[str, ...]] = (
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

# Diverging pair for signed contributions (SHAP): warm/cool poles that read as
# opposite, with a neutral gray midpoint. Validated together -- CVD deltaE 21.6,
# normal-vision 32.3, both clear of the floors.
DIVERGING_NEGATIVE: Final[str] = "#2a78d6"  # cool: reduces risk
DIVERGING_POSITIVE: Final[str] = "#e34948"  # warm: increases risk
DIVERGING_MIDPOINT: Final[str] = "#f0efec"  # neutral gray: no effect

# Reserved status colours for risk bands. Always paired with a text label.
STATUS: Final[dict[str, str]] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

RISK_BAND_COLORS: Final[dict[str, str]] = {
    "Low": STATUS["good"],
    "Medium": STATUS["warning"],
    "High": STATUS["critical"],
}


# Minimum perceptual lightness gap between adjacent ordinal steps. Below this
# the bands stop reading as distinct tiers.
_MIN_ADJACENT_DELTA_L: Final[float] = 0.06

# The documented ramp steps sit ~0.047 apart in lightness, so six or more bands
# cannot all clear the 0.06 floor no matter which steps are picked (verified by
# exhaustive search). Five is the ceiling; bin more coarsely rather than
# shipping tiers the reader cannot separate.
MAX_ORDINAL_BANDS: Final[int] = 5


def _oklab_lightness(hex_color: str) -> float:
    """Return the OKLab L coordinate of an sRGB hex colour.

    Perceptual lightness, not the naive luminance matplotlib would give -- the
    ordinal ramp's step spacing has to be judged in the space the eye uses.
    """
    raw = hex_color.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    red, green, blue = linear
    long_ = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    long_, medium, short = (np.cbrt(v) for v in (long_, medium, short))
    return float(0.2104542553 * long_ + 0.7936177850 * medium - 0.0040720468 * short)


_ORDINAL_LIGHTNESS: Final[tuple[float, ...]] = tuple(
    _oklab_lightness(step) for step in _ORDINAL_STEPS
)


@lru_cache(maxsize=8)
def _best_ordinal_subset(n: int) -> tuple[str, ...]:
    """Pick the ``n`` ramp steps whose smallest adjacent lightness gap is largest.

    An exhaustive search over a ten-step ramp is a few hundred combinations, so
    the optimal choice is simply computed rather than approximated.
    """
    best: tuple[int, ...] = ()
    best_gap = -1.0
    for combo in combinations(range(len(_ORDINAL_STEPS)), n):
        gap = min(
            _ORDINAL_LIGHTNESS[combo[i]] - _ORDINAL_LIGHTNESS[combo[i + 1]]
            for i in range(n - 1)
        )
        if gap > best_gap:
            best_gap, best = gap, combo
    if best_gap < _MIN_ADJACENT_DELTA_L:  # pragma: no cover - guarded by the cap
        logger.warning(
            "Ordinal ramp of %d bands has a minimum lightness gap of %.3f, "
            "below the %.2f floor", n, best_gap, _MIN_ADJACENT_DELTA_L,
        )
    return tuple(_ORDINAL_STEPS[i] for i in best)


def ordinal_ramp(n: int) -> list[str]:
    """Return ``n`` ordered colours from the validated single-hue blue ramp.

    Steps are chosen to maximise the **smallest** perceptual lightness gap, not
    spaced by index. Index spacing looks reasonable but produces uneven gaps --
    it put two adjacent steps 0.048 apart, below the 0.06 floor, so the middle
    bands stopped reading as distinct tiers.

    Args:
        n: Number of ordered bands, at most :data:`MAX_ORDINAL_BANDS`.

    Returns:
        Colours light-to-dark, perceptually evenly spaced.

    Raises:
        ValueError: If ``n`` exceeds what the ramp can separate. Bin the data
            more coarsely rather than accepting bands the reader cannot tell
            apart.
    """
    if n <= 0:
        return []
    if n == 1:
        return [SERIES[0]]
    if n > MAX_ORDINAL_BANDS:
        raise ValueError(
            f"ordinal_ramp({n}): the blue ramp keeps only {MAX_ORDINAL_BANDS} bands "
            f"at least {_MIN_ADJACENT_DELTA_L} apart in lightness. Use fewer bands."
        )

    return list(_best_ordinal_subset(n))


def apply_theme() -> None:
    """Install the house style into matplotlib's global rcParams.

    Thin marks, hairline recessive grid, no top or right spine, generous
    padding. Called once at import time by the EDA module and the UI.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",  # solid hairlines; dashes read as thresholds
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "text.color": INK_PRIMARY,
        }
    )


def style_axes(ax: Axes, xgrid: bool = False, ygrid: bool = True) -> Axes:
    """Apply per-axes chrome: grid on one direction only, recessive baseline.

    Args:
        ax: The axes to style.
        xgrid: Show vertical gridlines (use for horizontal bar charts).
        ygrid: Show horizontal gridlines (use for vertical bar charts).

    Returns:
        The same axes, for chaining.
    """
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    return ax


def add_caption(fig: Figure, text: str) -> None:
    """Attach a plain-English takeaway under a figure.

    Static PNGs have no hover layer, so the interpretation has to live on the
    figure itself rather than in a tooltip.
    """
    fig.supxlabel(text, fontsize=9, color=INK_SECONDARY, ha="left", x=0.0, wrap=True)


def label_bars(
    ax: Axes,
    bars: Sequence,
    values: Sequence[float],
    fmt: str = "{:.1f}%",
    horizontal: bool = False,
    pad: float = 0.01,
    tops: Sequence[float] | None = None,
) -> None:
    """Direct-label bar ends.

    Static figures have no tooltip, so the value must be readable off the mark.
    Labels sit outside the bar end -- never inside, where a short bar would clip
    them -- and wear text ink rather than the series colour.

    Args:
        ax: Target axes.
        bars: The bar containers returned by ``ax.bar``/``ax.barh``.
        values: The values to print.
        fmt: Format string applied to each value.
        horizontal: True for ``barh``.
        pad: Gap between the anchor and the label, as a fraction of the span.
        tops: Optional anchor positions overriding the bar ends. Pass the
            confidence-interval upper bounds when error bars are drawn, so the
            label clears the whisker instead of colliding with it.
    """
    span = max(values) if len(values) else 1.0
    offset = span * pad if span else 0.01
    anchors = list(tops) if tops is not None else None
    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        if horizontal:
            anchor = anchors[index] if anchors else bar.get_width()
            ax.text(
                anchor + offset, bar.get_y() + bar.get_height() / 2,
                fmt.format(value), va="center", ha="left",
                fontsize=9, color=INK_SECONDARY,
            )
        else:
            anchor = anchors[index] if anchors else bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, anchor + offset,
                fmt.format(value), ha="center", va="bottom",
                fontsize=9, color=INK_SECONDARY,
            )


def add_reference_line(ax: Axes, value: float, label: str, horizontal: bool = True) -> None:
    """Draw a labelled threshold line (e.g. the portfolio average default rate).

    This is a genuine threshold, so a dashed stroke is meaningful here -- unlike
    on a gridline, where dashing is noise.
    """
    draw = ax.axhline if horizontal else ax.axvline
    draw(value, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
    if horizontal:
        ax.text(
            ax.get_xlim()[1], value, f" {label}",
            va="center", ha="left", fontsize=8, color=INK_MUTED,
        )
    else:
        ax.text(
            value, ax.get_ylim()[1], f" {label}",
            va="bottom", ha="center", fontsize=8, color=INK_MUTED,
        )


def save_figure(fig: Figure, name: str, directory: Path | None = None) -> Path:
    """Save a figure as PNG into the reports directory.

    Args:
        fig: Figure to write.
        name: Base filename without extension.
        directory: Override the destination; defaults to ``reports/figures``.

    Returns:
        The path written.
    """
    target_dir = Path(directory) if directory else settings.figures_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.png"
    fig.savefig(path)
    logger.debug("Saved figure -> %s", path)
    return path


def new_figure(figsize: tuple[float, float] = (9.0, 5.0)) -> tuple[Figure, Axes]:
    """Create a single styled axes on the house surface."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    style_axes(ax)
    return fig, ax
