from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from PIL import Image


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.sans-serif": ["Arial", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "paper_tables"
FIGS = PROJECT_ROOT / "paper" / "figures"

FIGURE_DPI = 300

FIGURE_LAYOUTS = {
    "1x1": {
        "figsize": (5, 4),
        "pixels_300dpi": (1500, 1200),
    },
    "1x2": {
        "figsize": (7.5, 3),
        "pixels_300dpi": (2250, 900),
    },
    "1x3": {
        "figsize": (9, 3),
        "pixels_300dpi": (2700, 900),
    },
    "2x2": {
        "figsize": (7.5, 6),
        "pixels_300dpi": (2250, 1800),
    },
    "3x3": {
        "figsize": (12, 10),
        "pixels_300dpi": (3600, 3000),
    },
}

COLORBLIND_COLORS = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
]

PAPER_COLORS = {
    "cause": "#0072B2",
    "context": "#E69F00",
    "removal": "#D55E00",
    "non_pair": "#BDBDBD",
}

SINGLE_FIGSIZE = FIGURE_LAYOUTS["1x1"]["figsize"]
ORIGINAL_BINARY_FIGSIZE = SINGLE_FIGSIZE
SINGLE_LABEL_SIZE = 12
SINGLE_TICK_SIZE = 12
SINGLE_LEGEND_SIZE = 11
SINGLE_ANNOTATION_SIZE = 12
BAR_EDGE_COLOR = "#4D4D4D"
BAR_EDGE_WIDTH = 0.2

ROLE_ORDER = ["emo_cause", "emo_context", "non_pair"]
ROLE_TICK_LABELS = {
    "emo_cause": "emo-cause",
    "emo_context": "emo-context",
    "non_pair": "non-pair",
}
ROLE_LABELS = {
    "emo_cause": "emo-cause",
    "emo_context": "emo-context",
    "non_pair": "non-pair",
}
ROLE_COLORS = {
    "emo_cause": PAPER_COLORS["cause"],
    "emo_context": PAPER_COLORS["context"],
    "non_pair": PAPER_COLORS["non_pair"],
}
ROLE_ANNOTATION_COLORS = {
    "emo_cause": "#005A8D",
    "emo_context": "#C58400",
    "non_pair": "#4D4D4D",
}
ROLE_MARKERS = {
    "emo_cause": "o",
    "emo_context": "s",
    "non_pair": "^",
}
SPLIT_ORDER = ["train", "valid", "test"]
DISTANCE_BUCKET_ORDER = ["0", "1", "2", "3", "4-5", "6-10", ">10"]
RWC_FUSION_CONFUSION_MODEL = "RWC-Fusion"
RWC_FUSION_CONFUSION_MODALITY = "tav"
CONFUSION_SEEDS = (42, 123, 456)
CONFUSION_FAMILY_ORDER = [
    "RoBERTa",
    "WavLM",
    "CLIP",
    "RC-Fusion",
    "RW-Fusion",
    "RWC-Fusion",
    "hilo",
    "m3hg",
    "mecpe",
]
CONFUSION_FAMILY_LABELS = {
    "RoBERTa": "RoBERTa",
    "WavLM": "WavLM",
    "CLIP": "CLIP",
    "RC-Fusion": "RC-Fusion",
    "RW-Fusion": "RW-Fusion",
    "RWC-Fusion": "RWC-Fusion",
    "hilo": "HiLo",
    "m3hg": "M3HG",
    "mecpe": "MECPE-2step",
}
CONFUSION_MODALITY_LABELS = {
    "t": "T",
    "a": "A",
    "v": "V",
    "ta": "T+A",
    "tv": "T+V",
    "tav": "T+A+V",
}
CONFUSION_ROLE_LABELS = {
    "emo_cause": "cause",
    "emo_context": "context",
    "non_pair": "non-pair",
}
STRUCTURAL_FAMILY_ORDER = ["RoBERTa", "RC-Fusion", "RW-Fusion", "RWC-Fusion"]
ORIGINAL_BINARY_SOURCE_ORDER = ["orig_non_pair", "orig_pair"]
ORIGINAL_BINARY_SOURCE_LABELS = {
    "orig_non_pair": "Original non-pair",
    "orig_pair": "Original pair",
}


def style_axis(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    ax.tick_params(axis="both", colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.grid(True, axis=grid_axis, color="gray", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)


def add_vertical_bar_outline(ax, bars, *, color: str = BAR_EDGE_COLOR, linewidth: float = BAR_EDGE_WIDTH) -> None:
    for bar in bars:
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        if height <= 0:
            continue
        x0, x1 = x, x + width
        y0, y1 = y, y + height
        ax.plot([x0, x0], [y0, y1], color=color, linewidth=linewidth, zorder=4, solid_capstyle="butt")
        ax.plot([x1, x1], [y0, y1], color=color, linewidth=linewidth, zorder=4, solid_capstyle="butt")
        ax.plot([x0, x1], [y1, y1], color=color, linewidth=linewidth, zorder=4, solid_capstyle="butt")


def add_horizontal_bar_outline(ax, bars, *, color: str = BAR_EDGE_COLOR, linewidth: float = BAR_EDGE_WIDTH) -> None:
    for bar in bars:
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        if width <= 0:
            continue
        x0, x1 = x, x + width
        y0, y1 = y, y + height
        ax.plot([x1, x1], [y0, y1], color=color, linewidth=linewidth, zorder=4, solid_capstyle="butt")
        ax.plot([x0, x1], [y0, y0], color=color, linewidth=linewidth, zorder=4, solid_capstyle="butt")
        ax.plot([x0, x1], [y1, y1], color=color, linewidth=linewidth, zorder=4, solid_capstyle="butt")


def add_gradient_dashed_vline(
    ax,
    x: float,
    y0: float,
    y1: float,
    start_color: str,
    end_color: str,
    *,
    linewidth: float = 1.2,
    segments: int = 36,
    dash_every: int = 2,
) -> None:
    y_edges = np.linspace(y0, y1, segments + 1)
    start = np.array(to_rgb(start_color))
    end = np.array(to_rgb(end_color))
    lines = []
    colors = []
    for idx in range(segments):
        if idx % dash_every != 0:
            continue
        frac = idx / max(segments - 1, 1)
        lines.append([(x, y_edges[idx]), (x, y_edges[idx + 1])])
        colors.append(start * (1.0 - frac) + end * frac)
    collection = LineCollection(lines, colors=colors, linewidths=linewidth, zorder=2)
    ax.add_collection(collection)


def save_figure(
    fig,
    figure_dir,
    name: str,
    *,
    tight_layout: bool = True,
    fixed_canvas: bool = False,
) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    save_kwargs = {"dpi": FIGURE_DPI}
    if not fixed_canvas:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(
        figure_dir / f"{name}.pdf",
        format="pdf",
        **save_kwargs,
    )
    fig.savefig(
        figure_dir / f"{name}.svg",
        format="svg",
        **save_kwargs,
    )
    plt.close(fig)


def save_binary_axis() -> None:
    df = pd.read_csv(DATA / "figure_binary_axis_role_geometry.csv")
    df = df.set_index("gold_role").loc[ROLE_ORDER].reset_index()
    role_scores = df.set_index("gold_role")["prob_pair_mean"].to_dict()

    y_positions = [0.72, 0.48, 0.24]
    fig, ax = plt.subplots(
        figsize=(SINGLE_FIGSIZE[0], SINGLE_FIGSIZE[1] - 0.5),
        dpi=FIGURE_DPI,
    )

    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5, linewidth=1.2)

    for y, (_, row) in zip(y_positions, df.iterrows()):
        role = row["gold_role"]
        x = row["prob_pair_mean"]
        color = ROLE_COLORS[role]
        ax.hlines(y=y, xmin=0, xmax=x, color=color, linewidth=2)
        ax.scatter(
            x,
            y,
            s=100,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        label_text = f"{row['prob_pair_mean']:.3f}"
        ax.text(
            x + 0.018,
            y,
            label_text,
            ha="left",
            va="center",
            fontsize=SINGLE_ANNOTATION_SIZE,
            color="black",
        )

    np_score = role_scores["non_pair"]
    ctx_score = role_scores["emo_context"]
    cause_score = role_scores["emo_cause"]
    np_ctx_gap = ctx_score - np_score
    ctx_cause_gap = cause_score - ctx_score
    gap_label_style = dict(
        ha="left",
        va="center",
        fontsize=SINGLE_ANNOTATION_SIZE,
        color="black",
    )
    add_gradient_dashed_vline(
        ax,
        np_score,
        y_positions[2],
        y_positions[1],
        ROLE_COLORS["non_pair"],
        ROLE_COLORS["emo_context"],
    )
    add_gradient_dashed_vline(
        ax,
        ctx_score,
        y_positions[1],
        y_positions[0],
        ROLE_COLORS["emo_context"],
        ROLE_COLORS["emo_cause"],
    )
    ax.text(
        np_score + 0.030,
        (y_positions[2] + y_positions[1]) / 2.0,
        f"| gap to context | = {np_ctx_gap:.3f}",
        **gap_label_style,
    )
    ax.text(
        ctx_score + 0.030,
        (y_positions[1] + y_positions[0]) / 2.0,
        f"| gap to cause | = {ctx_cause_gap:.3f}",
        **gap_label_style,
    )

    ax.set_yticks([])
    ax.set_ylabel("Pair role")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(0.12, 0.76)
    ax.set_xlabel("Mean binary p(pair)")
    style_axis(ax, grid_axis="x")
    ax.tick_params(axis="both", labelsize=SINGLE_TICK_SIZE)
    ax.xaxis.label.set_size(SINGLE_LABEL_SIZE)
    ax.yaxis.label.set_size(SINGLE_LABEL_SIZE)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ROLE_COLORS[role],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            label=ROLE_TICK_LABELS[role].lower(),
        )
        for role in ROLE_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=SINGLE_LEGEND_SIZE,
        borderpad=0.25,
        labelspacing=0.25,
        handletextpad=0.4,
        borderaxespad=0.8,
    )
    save_figure(fig, FIGS, "binary_axis_role_geometry", fixed_canvas=True)


def save_context_channel() -> None:
    df = pd.read_csv(DATA / "figure_context_channel_threeclass.csv")
    df = df.set_index("gold_role").loc[ROLE_ORDER].reset_index()
    bar_label_fontsize = SINGLE_ANNOTATION_SIZE - 0.5

    fig, ax = plt.subplots(
        figsize=(SINGLE_FIGSIZE[0], SINGLE_FIGSIZE[1] - 0.5),
        dpi=FIGURE_DPI,
    )

    x_positions = list(range(len(df)))
    width = 0.36
    prob_values = df["prob_emo_context_mean"].tolist()
    drop_values = df["delta_emo_context_after_removal_mean"].tolist()
    prob_bars = ax.bar(
        [x - width / 2 for x in x_positions],
        prob_values,
        width=width,
        color=PAPER_COLORS["cause"],
        label="Mean p(emo-context)",
    )
    drop_bars = ax.bar(
        [x + width / 2 for x in x_positions],
        drop_values,
        width=width,
        color=PAPER_COLORS["context"],
        label="Drop after removal",
    )
    add_vertical_bar_outline(ax, prob_bars)
    add_vertical_bar_outline(ax, drop_bars)

    ax.set_ylabel("Probability", fontsize=SINGLE_LABEL_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [ROLE_LABELS[r] for r in df["gold_role"]],
        rotation=0,
        ha="center",
        color="black",
        fontsize=SINGLE_TICK_SIZE,
    )
    ax.set_ylim(0, 0.35)
    ax.tick_params(axis="both", labelsize=SINGLE_TICK_SIZE)
    ax.bar_label(prob_bars, fmt="%.3f", fontsize=bar_label_fontsize, padding=3, color="black")
    ax.bar_label(drop_bars, fmt="%.3f", fontsize=bar_label_fontsize, padding=3, color="black")
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=SINGLE_LEGEND_SIZE,
    )
    style_axis(ax, grid_axis="y")

    save_figure(fig, FIGS, "context_channel_threeclass", fixed_canvas=True)


def save_binary_axis_removal_by_family() -> None:
    df = pd.read_csv(DATA / "binary_axis_and_removal_by_family_long.csv")
    df = df[df["family"].isin(STRUCTURAL_FAMILY_ORDER)].copy()
    df["family"] = pd.Categorical(df["family"], categories=STRUCTURAL_FAMILY_ORDER, ordered=True)
    df["gold_role"] = pd.Categorical(df["gold_role"], categories=ROLE_ORDER, ordered=True)
    df = df.sort_values(["family", "gold_role"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(SINGLE_FIGSIZE[0], 5),
        dpi=FIGURE_DPI,
        sharex=True,
    )
    x = np.arange(len(STRUCTURAL_FAMILY_ORDER))
    width = 0.22
    offsets = {
        "emo_cause": -width,
        "emo_context": 0.0,
        "non_pair": width,
    }
    metric_specs = [
        ("prob_pair_mean", "Mean binary p(pair)", "(a) Binary score"),
        ("delta_pair_mean", "Probability drop", "(b) Evidence removal"),
    ]

    for ax, (metric, ylabel, panel_title) in zip(axes, metric_specs):
        for role in ROLE_ORDER:
            role_df = df[df["gold_role"] == role].set_index("family").reindex(STRUCTURAL_FAMILY_ORDER)
            values = role_df[metric].astype(float).to_numpy()
            errors = role_df[metric.replace("_mean", "_std")].astype(float).to_numpy()
            bars = ax.bar(
                x + offsets[role],
                values,
                width=width,
                color=ROLE_COLORS[role],
                edgecolor="none",
                yerr=errors,
                error_kw={
                    "elinewidth": 0.7,
                    "ecolor": "black",
                    "capsize": 2.5,
                    "capthick": 0.7,
                },
                label=ROLE_LABELS[role],
                zorder=3,
            )
            add_vertical_bar_outline(ax, bars)
        ax.set_ylabel(ylabel, fontsize=SINGLE_LABEL_SIZE)
        ax.set_ylim(0.0, 0.46)
        ax.set_title(panel_title, fontsize=SINGLE_LABEL_SIZE, color="black", pad=4)
        ax.tick_params(axis="both", labelsize=SINGLE_TICK_SIZE, colors="black")
        style_axis(ax, grid_axis="y")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(STRUCTURAL_FAMILY_ORDER, color="black", fontsize=SINGLE_TICK_SIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=SINGLE_LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    save_figure(fig, FIGS, "appendix_binary_axis_removal_by_family", tight_layout=False, fixed_canvas=True)


def save_matched_contrast_by_family() -> None:
    df = pd.read_csv(DATA / "matched_pair_contrast_by_family_appendix.csv")
    families = STRUCTURAL_FAMILY_ORDER
    binary = (
        df[df["analysis"] == "binary_prob_pair"]
        .set_index("family")
        .reindex(families)["context_minus_non_pair_prob_pair_mean"]
        .astype(float)
    )
    three = (
        df[df["analysis"] == "three_prob_context"]
        .set_index("family")
        .reindex(families)["context_minus_non_pair_prob_context_mean"]
        .astype(float)
    )

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=FIGURE_DPI)
    x = np.arange(len(families))
    width = 0.32
    binary_bars = ax.bar(
        x - width / 2,
        binary.to_numpy(),
        width=width,
        color=PAPER_COLORS["cause"],
        edgecolor="none",
        label="Binary p(pair) gap",
        zorder=3,
    )
    three_bars = ax.bar(
        x + width / 2,
        three.to_numpy(),
        width=width,
        color=PAPER_COLORS["context"],
        edgecolor="none",
        label="Three-class p(emo-context) gap",
        zorder=3,
    )
    add_vertical_bar_outline(ax, binary_bars)
    add_vertical_bar_outline(ax, three_bars)
    for bars in (binary_bars, three_bars):
        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=2,
            fontsize=SINGLE_ANNOTATION_SIZE - 0.5,
            rotation=90,
            color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(families, color="black", fontsize=SINGLE_TICK_SIZE)
    ax.set_ylabel("Context - matched non-pair", fontsize=SINGLE_LABEL_SIZE)
    ax.set_ylim(0, 0.08)
    ax.tick_params(axis="both", labelsize=SINGLE_TICK_SIZE, colors="black")
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=SINGLE_LEGEND_SIZE,
    )
    style_axis(ax, grid_axis="y")
    save_figure(fig, FIGS, "matched_contrast_by_family", fixed_canvas=True)


def save_shortcut_conflict_by_family() -> None:
    df = pd.read_csv(DATA / "shortcut_conflict_subsets_by_family_appendix.csv")
    df = df[(df["source"] == "binary") & df["family"].isin(STRUCTURAL_FAMILY_ORDER)].copy()
    families = STRUCTURAL_FAMILY_ORDER
    subset_specs = [
        ("long_distance_emo_cause", "Long-distance cause", PAPER_COLORS["cause"]),
        ("source_nonpair_emo_context", "Original-negative context", PAPER_COLORS["context"]),
        ("local_non_pair", "Local non-pair", PAPER_COLORS["non_pair"]),
    ]

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=FIGURE_DPI)
    x = np.arange(len(families))
    width = 0.24
    offsets = [-width, 0.0, width]
    all_bars = []
    for offset, (subset, label, color) in zip(offsets, subset_specs):
        values = (
            df[df["conflict_subset"] == subset]
            .set_index("family")
            .reindex(families)["prob_pair_mean"]
            .astype(float)
            .to_numpy()
        )
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="none",
            label=label,
            zorder=3,
        )
        all_bars.append(bars)
        add_vertical_bar_outline(ax, bars)

    for bars in all_bars:
        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=2,
            fontsize=SINGLE_ANNOTATION_SIZE - 2.5,
            rotation=90,
            color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(families, color="black", fontsize=SINGLE_TICK_SIZE)
    ax.set_ylabel("Mean binary p(pair)", fontsize=SINGLE_LABEL_SIZE)
    ax.set_ylim(0, 0.38)
    ax.tick_params(axis="both", labelsize=SINGLE_TICK_SIZE, colors="black")
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=SINGLE_LEGEND_SIZE - 0.5,
        borderpad=0.3,
        labelspacing=0.35,
    )
    style_axis(ax, grid_axis="y")
    save_figure(fig, FIGS, "shortcut_conflict_by_family", fixed_canvas=True)


def save_shortcut_vs_evidence_stress() -> None:
    # Deprecated: retained only for reference.
    df = pd.read_csv(DATA / "shortcut_vs_evidence_stress_by_family_appendix.csv")
    df = df[(df["source"] == "binary") & (df["family"] == RWC_FUSION_CONFUSION_MODEL)].copy()
    subset_order = [
        "local_lexical_non_pair",
        "long_distance_emo_cause",
        "source_nonpair_emo_context",
    ]
    subset_labels = {
        "local_lexical_non_pair": "Local lexical\nnon-pair",
        "long_distance_emo_cause": "Long-distance\ncause",
        "source_nonpair_emo_context": "Original-negative\ncontext",
    }
    subset_colors = {
        "local_lexical_non_pair": PAPER_COLORS["removal"],
        "long_distance_emo_cause": PAPER_COLORS["removal"],
        "source_nonpair_emo_context": PAPER_COLORS["removal"],
    }
    df["stress_subset"] = pd.Categorical(df["stress_subset"], categories=subset_order, ordered=True)
    df = df.sort_values("stress_subset").set_index("stress_subset")

    fig, ax = plt.subplots(figsize=(SINGLE_FIGSIZE[0], SINGLE_FIGSIZE[1] - 1), dpi=FIGURE_DPI)
    y = np.arange(len(subset_order))
    xmax = 0.41
    for pos, subset in enumerate(subset_order):
        value = float(df.loc[subset, "prob_pair_mean"])
        err = float(df.loc[subset, "prob_pair_std"])
        color = subset_colors[subset]
        ax.errorbar(
            value,
            pos,
            xerr=err,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=4,
            capthick=1.2,
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=3,
        )
        ax.text(
            min(value + err + 0.014, xmax - 0.01),
            pos,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=SINGLE_ANNOTATION_SIZE,
            color="black",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([subset_labels[s] for s in subset_order], fontsize=SINGLE_TICK_SIZE, color="black")
    ax.set_xlabel("Mean binary p(pair)", fontsize=SINGLE_LABEL_SIZE)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.55, len(subset_order) - 0.45)
    ax.invert_yaxis()
    ax.tick_params(axis="both", colors="black", labelsize=SINGLE_TICK_SIZE)
    style_axis(ax, grid_axis="x")
    ax.grid(False, axis="y")
    fig.subplots_adjust(left=0.315, right=0.985, bottom=0.16, top=0.95)
    save_figure(fig, FIGS, "shortcut_vs_evidence_stress", tight_layout=False, fixed_canvas=True)


def _row_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    row_sums = matrix.sum(axis=1).replace(0, np.nan)
    return matrix.div(row_sums, axis=0).fillna(0.0) * 100.0


def _confusion_matrix_from_long(df: pd.DataFrame) -> pd.DataFrame:
    matrix = (
        df.pivot_table(index="gold", columns="pred", values="count", aggfunc="sum", fill_value=0)
        .reindex(index=ROLE_ORDER, columns=ROLE_ORDER, fill_value=0)
        .astype(float)
    )
    return matrix


def _pick_best_modality_by_family(df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for model_family in CONFUSION_FAMILY_ORDER:
        family_df = df[df["model_family"] == model_family].copy()
        if family_df.empty:
            raise ValueError(f"Missing main result rows for model family {model_family}.")
        best_row = family_df.sort_values(
            ["pair_role_macro_f1_mean", "pair_role_accuracy_mean", "modality"],
            ascending=[False, False, True],
        ).iloc[0]
        best_rows.append(best_row)
    return pd.DataFrame(best_rows)


def save_family_mean_confusion_matrices() -> None:
    results = pd.read_csv(DATA / "main_model_results_by_family.csv")
    best = _pick_best_modality_by_family(results)

    confusion = pd.read_csv(DATA / "pair_role_confusion_by_family_long.csv")
    role_labels = {
        "emo_cause": "cause",
        "emo_context": "context",
        "non_pair": "non",
    }

    panels: list[tuple[str, pd.DataFrame]] = []
    for _, row in best.iterrows():
        model_family = row["model_family"]
        modality = row["modality"]
        family_conf = confusion[
            (confusion["model_family"] == model_family) & (confusion["modality"] == modality)
        ].copy()
        if family_conf.empty:
            raise ValueError(f"Missing confusion rows for {model_family} ({modality}).")

        matrix = (
            family_conf.pivot_table(
                index="gold",
                columns="pred",
                values="rate_within_gold_mean",
                aggfunc="mean",
                fill_value=0.0,
            )
            .reindex(index=ROLE_ORDER, columns=ROLE_ORDER, fill_value=0.0)
            .astype(float)
            * 100.0
        )
        display_family = CONFUSION_FAMILY_LABELS.get(model_family, model_family)
        display_modality = CONFUSION_MODALITY_LABELS.get(modality, modality.upper())
        panels.append((f"{display_family} ({display_modality})", matrix))

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(FIGURE_LAYOUTS["3x3"]["figsize"][0], FIGURE_LAYOUTS["3x3"]["figsize"][1] - 1.2),
        dpi=FIGURE_DPI,
    )
    axes = axes.ravel()

    last_im = None
    for idx, (ax, (title, matrix)) in enumerate(zip(axes, panels)):
        last_im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=100, cmap="Blues", aspect="equal")
        ax.set_box_aspect(1)
        ax.set_xticks(range(len(ROLE_ORDER)))
        ax.set_yticks(range(len(ROLE_ORDER)))
        ax.set_title(f"({chr(ord('a') + idx)}) {title}", fontsize=13, color="black", pad=6)
        if idx >= 6:
            ax.set_xticklabels([role_labels[r] for r in ROLE_ORDER], color="black", rotation=0, fontsize=13)
        else:
            ax.set_xticklabels([])
        if idx % 3 == 0:
            ax.set_yticklabels([role_labels[r] for r in ROLE_ORDER], color="black", fontsize=13)
        else:
            ax.set_yticklabels([])
        if idx % 3 == 2:
            for row_idx, label in enumerate([role_labels[r] for r in ROLE_ORDER]):
                ax.text(
                    2.56,
                    row_idx,
                    label,
                    ha="left",
                    va="center",
                    color="black",
                    fontsize=13,
                    clip_on=False,
                )
        ax.tick_params(axis="both", colors="black", length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = float(matrix.iloc[i, j])
                text_color = "white" if value >= 50 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=13,
                )

    fig.supxlabel("Predicted role", color="black", fontsize=14, y=0.025)
    fig.supylabel("Gold role", color="black", fontsize=14)
    cax = fig.add_axes([0.955, 0.20, 0.018, 0.60])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Mean within-gold %", color="black", fontsize=14)
    cbar.ax.tick_params(colors="black", labelsize=13)
    fig.subplots_adjust(left=0.085, right=0.89, bottom=0.08, top=0.925, wspace=0.16, hspace=0.15)
    save_figure(fig, FIGS, "appendix_family_mean_confusion_matrices", tight_layout=False)


def save_original_binary_transition_heatmap() -> None:
    df = pd.read_csv(DATA / "original_binary_to_final_role_flow_long.csv")
    df = df[df["split"] == "overall"].copy()
    df = df[df["original_binary_role"].isin(ORIGINAL_BINARY_SOURCE_ORDER)]
    if df.empty:
        raise ValueError("Missing original binary transition rows for the appendix figure.")

    source_order = ["orig_pair", "orig_non_pair"]

    matrix = (
        df.pivot_table(
            index="original_binary_role",
            columns="final_role",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=source_order, columns=ROLE_ORDER, fill_value=0)
        .astype(float)
    )
    percent = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    fig, ax = plt.subplots(
        figsize=ORIGINAL_BINARY_FIGSIZE,
        dpi=FIGURE_DPI,
    )

    x_positions = np.arange(len(source_order))
    bar_width = 0.58
    colors = ROLE_COLORS
    legend_handles = [
        Patch(facecolor=colors["emo_cause"], edgecolor="none", label="emo-cause"),
        Patch(facecolor=colors["emo_context"], edgecolor="none", label="emo-context"),
        Patch(facecolor=colors["non_pair"], edgecolor="none", label="non-pair"),
    ]
    text_colors = {
        "emo_cause": "white",
        "emo_context": "white",
        "non_pair": "black",
    }
    bottoms = np.zeros(len(source_order))
    tiny_threshold = 7.0
    tiny_label_fontsize = SINGLE_ANNOTATION_SIZE

    for role in ROLE_ORDER:
        heights = percent[role].to_numpy()
        bars = ax.bar(
            x_positions,
            heights,
            width=bar_width,
            bottom=bottoms,
            color=colors[role],
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )
        add_vertical_bar_outline(ax, bars)
        for idx, bar in enumerate(bars):
            h = float(bar.get_height())
            base = float(bar.get_y())
            cx = bar.get_x() + bar.get_width() / 2.0
            cy = base + h / 2.0
            label = f"{h:.1f}%"
            if h < tiny_threshold:
                source = source_order[idx]
                if source == "orig_pair" and role == "non_pair":
                    ax.annotate(
                        label,
                        xy=(cx - bar_width * 0.10, 100.0),
                        xytext=(cx - 0.58, 103.2),
                        ha="center",
                        va="bottom",
                        fontsize=SINGLE_ANNOTATION_SIZE,
                        color="black",
                        arrowprops=dict(
                            arrowstyle="<-",
                            color="black",
                            lw=1.0,
                            shrinkA=2,
                            shrinkB=3,
                        ),
                        clip_on=False,
                    )
                else:
                    if source == "orig_non_pair":
                        x_text = cx + 0.43
                        y_text = 8.0 if role == "emo_cause" else 18.5
                        label_y_text = y_text - 2.4 if role == "emo_cause" else y_text
                        ha = "left"
                        if role == "emo_context":
                            xy = (cx + bar_width / 2.0, base + h)
                        else:
                            xy = (cx + bar_width / 2.0, max(base + h, 1.2))
                        annotation_color = ROLE_ANNOTATION_COLORS[role]
                    else:
                        y_shift = -7.0 if role == "emo_cause" else 7.0
                        x_text = cx - 0.56
                        y_text = max(cy + y_shift, 2.5)
                        label_y_text = y_text
                        ha = "right"
                        xy = (cx - bar_width / 2.0, cy)
                        annotation_color = colors[role]
                    if source == "orig_non_pair" and role == "emo_cause":
                        ax.annotate(
                            "",
                            xy=xy,
                            xytext=(x_text, y_text),
                            arrowprops=dict(
                                arrowstyle="<-",
                                color=annotation_color,
                                lw=1.0,
                                shrinkA=2,
                                shrinkB=3,
                            ),
                        )
                        ax.text(
                            x_text,
                            label_y_text,
                            label,
                            ha=ha,
                            va="center",
                            fontsize=tiny_label_fontsize,
                            color=annotation_color,
                        )
                    else:
                        ax.annotate(
                            label,
                            xy=xy,
                            xytext=(x_text, label_y_text),
                            ha=ha,
                            va="center",
                            fontsize=tiny_label_fontsize,
                            color=annotation_color,
                            arrowprops=dict(
                                arrowstyle="<-",
                                color=annotation_color,
                                lw=1.0,
                                shrinkA=2,
                                shrinkB=3,
                            ),
                        )
            else:
                ax.text(
                    cx,
                    cy,
                    label,
                    ha="center",
                    va="center",
                    fontsize=SINGLE_ANNOTATION_SIZE,
                    color=text_colors[role],
                )
        bottoms += heights

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Original pair", "Original non-pair"], color="black", fontsize=SINGLE_TICK_SIZE)
    ax.set_xlim(-0.85, 1.78)
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Within original label (%)", color="black", fontsize=SINGLE_LABEL_SIZE)
    ax.tick_params(axis="both", colors="black", length=5, width=1.0, labelsize=SINGLE_TICK_SIZE)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=SINGLE_LEGEND_SIZE,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    ax.grid(True, axis="y", color="gray", alpha=0.3, linewidth=0.8, zorder=0)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(
        fig,
        FIGS,
        "original_binary_to_final_role_mapping",
        tight_layout=False,
        fixed_canvas=True,
    )


def save_boundary_uncertainty_enrichment_figure() -> None:
    df = pd.read_csv(DATA / "rwc_fusion_uncertainty_enrichment.csv")
    selection_order = [
        "baseline",
        "Uncertainty union + disagreement",
        "Top 5% uncertainty per scored split",
        "Top 5% highest probability per scored split",
        "Top 5% lowest probability per scored split",
    ]
    row_map = {row["selection_label"]: row for _, row in df.iterrows()}
    required = [s for s in selection_order[1:] if s not in row_map]
    if required:
        raise ValueError(f"Missing rows for the RWC-Fusion uncertainty figure: {required}")

    baseline_rate = float(df["reference_context_rate"].iloc[0]) * 100.0
    plot_rows = [
        {
            "label": "All candidate\npairs",
            "rate": baseline_rate,
            "lift": 1.0,
            "color": PAPER_COLORS["non_pair"],
            "text_color": "black",
            "is_baseline": True,
        },
        {
            "label": "Boundary set\n+ disagreement",
            "rate": float(row_map["Uncertainty union + disagreement"]["emo_context_rate"]) * 100.0,
            "lift": float(row_map["Uncertainty union + disagreement"]["context_lift_vs_reference"]),
            "color": PAPER_COLORS["context"],
            "text_color": "black",
            "is_baseline": False,
        },
        {
            "label": "Top 5%\nmost uncertain",
            "rate": float(row_map["Top 5% uncertainty per scored split"]["emo_context_rate"]) * 100.0,
            "lift": float(row_map["Top 5% uncertainty per scored split"]["context_lift_vs_reference"]),
            "color": PAPER_COLORS["context"],
            "text_color": "black",
            "is_baseline": False,
        },
        {
            "label": "Top 5%\nhighest score",
            "rate": float(row_map["Top 5% highest probability per scored split"]["emo_context_rate"]) * 100.0,
            "lift": float(row_map["Top 5% highest probability per scored split"]["context_lift_vs_reference"]),
            "color": PAPER_COLORS["cause"],
            "text_color": "black",
            "is_baseline": False,
        },
        {
            "label": "Top 5%\nlowest score",
            "rate": float(row_map["Top 5% lowest probability per scored split"]["emo_context_rate"]) * 100.0,
            "lift": float(row_map["Top 5% lowest probability per scored split"]["context_lift_vs_reference"]),
            "color": PAPER_COLORS["non_pair"],
            "text_color": "black",
            "is_baseline": False,
        },
    ]
    plot_df = pd.DataFrame(plot_rows)
    plot_df["y"] = np.arange(len(plot_df))[::-1]

    fig, ax = plt.subplots(
        figsize=(SINGLE_FIGSIZE[0], SINGLE_FIGSIZE[1] - 0.5),
        dpi=FIGURE_DPI,
    )
    y = plot_df["y"].to_numpy()
    uncertainty_bars = ax.barh(
        y,
        plot_df["rate"],
        color=plot_df["color"],
        edgecolor="none",
        height=0.56,
        zorder=3,
    )
    add_horizontal_bar_outline(ax, uncertainty_bars)
    ax.axvline(baseline_rate, color="gray", linestyle="--", linewidth=1.0, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], color="black", fontsize=SINGLE_TICK_SIZE)
    ax.set_xlabel("emo-context rate in selected pairs (%)", color="black", fontsize=SINGLE_LABEL_SIZE)
    ax.set_xlim(0, max(plot_df["rate"].max(), baseline_rate) * 1.25)
    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    ax.tick_params(axis="x", labelsize=SINGLE_TICK_SIZE, colors="black")
    ax.tick_params(axis="y", colors="black")

    for yi, rate, lift, is_baseline in zip(
        plot_df["y"], plot_df["rate"], plot_df["lift"], plot_df["is_baseline"]
    ):
        if is_baseline:
            text = f"{rate:.1f}%  baseline"
        else:
            text = f"{rate:.1f}%  {lift:.2f}x"
        ax.text(
            rate + 0.22,
            yi,
            text,
            ha="left",
            va="center",
            fontsize=SINGLE_ANNOTATION_SIZE,
            color="black",
        )

    style_axis(ax, grid_axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, FIGS, "boundary_uncertainty_enrichment", tight_layout=False, fixed_canvas=True)


def save_source_binary_remap_delta() -> None:
    df = pd.read_csv(DATA / "source_binary_remap_summary.csv")
    strategies = [
        ("drop_context", "Drop context", PAPER_COLORS["removal"], "o", -0.22),
        ("context_to_positive", "Context -> pair", PAPER_COLORS["cause"], "s", 0.0),
        ("context_to_negative", "Context -> non-pair", PAPER_COLORS["non_pair"], "^", 0.22),
    ]
    models = (
        df[["model_order", "model_label", "modality"]]
        .drop_duplicates()
        .sort_values("model_order")
        .reset_index(drop=True)
    )
    y_base = np.arange(len(models))[::-1]
    y_lookup = dict(zip(models["model_label"], y_base))

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=FIGURE_DPI)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1.0, zorder=1)

    for strategy, label, color, marker, offset in strategies:
        part = df[df["strategy"] == strategy].copy()
        part = part.sort_values("model_order")
        xs = part["delta_f1_vs_original_mean"].astype(float).to_numpy() * 100.0
        xerr = part["delta_f1_vs_original_std"].astype(float).to_numpy() * 100.0
        ys = np.array([y_lookup[row["model_label"]] for _, row in part.iterrows()]) + offset
        ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            fmt=marker,
            markersize=6.5,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            capthick=0.8,
            linestyle="none",
            label=label,
            zorder=3,
        )

    y_labels = [row.model_label for row in models.itertuples(index=False)]
    ax.set_yticks(y_base)
    ax.set_yticklabels(y_labels, color="black", fontsize=SINGLE_TICK_SIZE)
    ax.set_xlabel("Pair F1 change vs. binary baseline (pt)", fontsize=SINGLE_LABEL_SIZE)
    ax.set_xlim(-3.0, 6.2)
    ax.set_ylim(-0.55, len(models) - 0.45)
    ax.tick_params(axis="both", labelsize=SINGLE_TICK_SIZE, colors="black")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=SINGLE_LEGEND_SIZE,
        borderpad=0.35,
        labelspacing=0.35,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    style_axis(ax, grid_axis="x")
    ax.grid(False, axis="y")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save_figure(fig, FIGS, "source_binary_remap_delta", tight_layout=False, fixed_canvas=True)


def save_role_distance_distributions() -> None:
    df = pd.read_csv(DATA / "distance_distribution_by_split_label.csv")
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    df["label"] = pd.Categorical(df["label"], categories=ROLE_ORDER, ordered=True)
    df["distance_bucket"] = pd.Categorical(
        df["distance_bucket"], categories=DISTANCE_BUCKET_ORDER, ordered=True
    )
    distance_text_size = 10.19
    distance_legend_size = 9.34

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_LAYOUTS["1x3"]["figsize"][0], FIGURE_LAYOUTS["1x3"]["figsize"][1] + 0.55),
        dpi=FIGURE_DPI,
        sharey=True,
    )

    for idx, (ax, split) in enumerate(zip(axes, SPLIT_ORDER)):
        split_df = df[df["split"] == split].copy()
        pivot = (
            split_df.pivot_table(
                index="distance_bucket",
                columns="label",
                values="pct_within_split_label",
                aggfunc="mean",
            )
            .reindex(index=DISTANCE_BUCKET_ORDER, columns=ROLE_ORDER, fill_value=0.0)
            .fillna(0.0)
            * 100.0
        )
        x = np.arange(len(DISTANCE_BUCKET_ORDER))
        for role in ROLE_ORDER:
            ax.plot(
                x,
                pivot[role].to_numpy(),
                color=ROLE_COLORS[role],
                marker=ROLE_MARKERS[role],
                linewidth=2,
                markersize=6,
                label=ROLE_LABELS[role],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(DISTANCE_BUCKET_ORDER, color="black", rotation=0)
        ax.set_ylim(0, 45)
        ax.set_yticks(np.arange(0, 46, 10))
        ax.set_title(
            f"({chr(ord('a') + idx)}) {split.capitalize()}",
            fontsize=distance_text_size,
            color="black",
            pad=8,
        )
        style_axis(ax, grid_axis="y")
        ax.tick_params(axis="both", colors="black", labelsize=distance_text_size)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=distance_legend_size,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.supxlabel("Absolute distance bucket", color="black", fontsize=distance_text_size, y=0.065)
    fig.supylabel("Within-role share (%)", color="black", fontsize=distance_text_size)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, FIGS, "appendix_role_distance_distributions", tight_layout=False)


def save_discovery_construction_wrapper() -> None:
    png_path = FIGS / "discovery_construction.png"
    if not png_path.exists():
        return

    image = Image.open(png_path).convert("RGBA")
    width, height = image.size
    upscale = 4
    image = image.resize((width * upscale, height * upscale), resample=Image.Resampling.LANCZOS)
    image = np.asarray(image)
    height, width = image.shape[:2]
    aspect = width / height
    fig, ax = plt.subplots(figsize=(2.5, 2.5 / aspect), dpi=FIGURE_DPI)
    ax.imshow(image)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    save_figure(fig, FIGS, "discovery_construction")


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    save_binary_axis()
    save_context_channel()
    save_binary_axis_removal_by_family()
    save_matched_contrast_by_family()
    save_shortcut_conflict_by_family()
    # Deprecated: shortcut_vs_evidence_stress is no longer generated.
    # save_shortcut_vs_evidence_stress()
    save_original_binary_transition_heatmap()
    save_boundary_uncertainty_enrichment_figure()
    save_source_binary_remap_delta()
    save_family_mean_confusion_matrices()
    save_role_distance_distributions()
    save_discovery_construction_wrapper()


if __name__ == "__main__":
    main()
