"""README figures, rendered in a light and a dark variant."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

import config  # noqa: E402

# The original project's LSTM pipeline (v1), measured once with the same block
# cross-validation on all six gestures and on the four shipped ones.
BASELINE_ACCURACY = {"all": 0.515, "enabled": 0.891}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
        "accent": "#2a78d6", "neutral": "#b9b8b2",
        "ramp": ["#f3f7fc", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f",
        "accent": "#3987e5", "neutral": "#5d5c57",
        "ramp": ["#202328", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"],
    },
}


def luminance(color):
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def style(ax, theme):
    ax.set_facecolor(theme["surface"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=theme["muted"], length=0, labelsize=10)


def confusion_figure(metrics, theme, path):
    names = [config.GESTURES[g]["name"] for g in metrics["gestures"]]
    counts = np.array(metrics["smoothed"]["confusion_matrix"], dtype=float)
    rates = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("ramp", theme["ramp"])

    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=theme["surface"])
    style(ax, theme)
    ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted", color=theme["muted"], fontsize=10, labelpad=8)
    ax.set_ylabel("Actual", color=theme["muted"], fontsize=10, labelpad=8)
    ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color=theme["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(names)):
        for j in range(len(names)):
            if counts[i, j] == 0:
                continue
            fill = cmap(rates[i, j])
            ink = "#0b0b0b" if luminance(fill) > 0.45 else "#ffffff"
            ax.text(j, i, f"{rates[i, j]:.0%}", ha="center", va="center", fontsize=9, color=ink)

    ax.set_title(f"Confusion matrix ({config.SMOOTHING_WINDOW}-frame average, cross-validated)",
                 color=theme["text"], fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)


def accuracy_figure(metrics, theme, path):
    groups = [
        (f"All {len(config.GESTURES)} signs", BASELINE_ACCURACY["all"], metrics["all_gestures"]["smoothed"]["accuracy"]),
        (f"{len(metrics['gestures'])} shipped signs", BASELINE_ACCURACY["enabled"], metrics["smoothed"]["accuracy"]),
    ]
    series = [("Original LSTM pipeline", theme["neutral"]), ("New pipeline", theme["accent"])]
    fig, ax = plt.subplots(figsize=(6.2, 2.6), facecolor=theme["surface"])
    style(ax, theme)
    height = 0.34
    y = np.arange(len(groups))[::-1]
    for s_index, (label, color) in enumerate(series):
        offsets = y + (0.5 - s_index) * (height + 0.04)
        values = [g[1 + s_index] for g in groups]
        ax.barh(offsets, values, height=height, color=color, label=label)
        for yi, value in zip(offsets, values):
            ax.text(value + 0.012, yi, f"{value:.1%}", va="center", fontsize=9.5, color=theme["text"])
    ax.set_yticks(y, [g[0] for g in groups], color=theme["text"])
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    ax.set_title("Cross-validated accuracy", color=theme["text"], fontsize=11, loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)


def save_all(metrics):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        confusion_figure(metrics, theme, config.FIGURES_DIR / f"confusion-matrix-{mode}.png")
        accuracy_figure(metrics, theme, config.FIGURES_DIR / f"accuracy-{mode}.png")
