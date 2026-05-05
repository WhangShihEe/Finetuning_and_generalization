"""Visualise agreement between GPT intended sentiment, human ratings, and judge ratings.

Generates four plots:
  1. Bar chart: mean rated sentiment per intended level, grouped by rater
  2. Confusion-style heatmap: intended sentiment vs actual rated sentiment (per judge model)
  3. Per-tier breakdown: same as plot 1 but split by tier
  4. Topic mention comparison: bar chart for topic mention scores

Usage:
    # With judge ratings only (no human ratings yet)
    python testing_eval_consistency/plot_comparison.py

    # With human ratings too
    python testing_eval_consistency/plot_comparison.py \\
        --human_path testing_eval_consistency/human_ratings.json

    # Custom tiers
    python testing_eval_consistency/plot_comparison.py --tiers direct close distant

    # Custom paths
    python testing_eval_consistency/plot_comparison.py \\
        --judge_path testing_eval_consistency/judge_ratings.json \\
        --output_dir testing_eval_consistency/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
DEFAULT_JUDGE_PATH = HERE / "judge_ratings.json"
DEFAULT_HUMAN_PATH = HERE / "human_ratings.json"
DEFAULT_OUTPUT_DIR = HERE / "plots"

DEFAULT_TIERS = ["direct", "close"]
LEVELS = list(range(11))  # 0-10


# ── Data loading ──────────────────────────────────────────────────────────────


def load_judge_ratings(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def load_human_ratings(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


# ── Aggregation helpers ───────────────────────────────────────────────────────


def stats_by_level(
    records: list[dict[str, Any]],
    score_field: str,
    level_field: str = "intended_sentiment",
) -> dict[int, tuple[float, float, int] | None]:
    """Return {level: (mean, sem, n)} for levels with at least one value, else None."""
    buckets: dict[int, list[float]] = {lv: [] for lv in LEVELS}
    for rec in records:
        val = rec.get(score_field)
        lv = rec.get(level_field)
        if val is not None and lv is not None:
            buckets[lv].append(float(val))
    out: dict[int, tuple[float, float, int] | None] = {}
    for lv, vals in buckets.items():
        if not vals:
            out[lv] = None
        else:
            n = len(vals)
            mean = sum(vals) / n
            sem = float(np.std(vals)) / np.sqrt(n) if n > 1 else 0.0
            out[lv] = (mean, sem, n)
    return out


def _draw_bars(
    ax: plt.Axes,
    raters: list[tuple[str, dict[int, tuple[float, float, int] | None], str]],
    ylim_top: float,
    capsize: int = 3,
    annotation_fontsize: int = 6,
) -> None:
    n_raters = len(raters)
    bar_width = 0.7 / max(n_raters, 1)
    offsets = np.linspace(-(n_raters - 1) / 2, (n_raters - 1) / 2, n_raters) * bar_width

    for offset, (label, stats, color) in zip(offsets, raters):
        present = [lv for lv in LEVELS if stats.get(lv) is not None]
        xs = [lv + offset for lv in present]
        ys = [stats[lv][0] for lv in present]  # type: ignore[index]
        errs = [stats[lv][1] for lv in present]  # type: ignore[index]
        ns = [stats[lv][2] for lv in present]  # type: ignore[index]
        ax.bar(xs, ys, width=bar_width, label=label, color=color, alpha=0.85,
               yerr=errs, capsize=capsize, error_kw={"elinewidth": 1, "alpha": 0.7})
        for x, y, err, n_val in zip(xs, ys, errs, ns):
            label_y = min(y + err + 0.15, ylim_top - 0.2)
            ax.text(x, label_y, f"n={n_val}", ha="center", va="bottom",
                    fontsize=annotation_fontsize, color="#333333", rotation=90)


def confusion_matrix(
    records: list[dict[str, Any]],
    score_field: str,
) -> np.ndarray:
    mat = np.zeros((11, 11), dtype=int)
    for rec in records:
        intended = rec.get("intended_sentiment")
        rated = rec.get(score_field)
        if intended is not None and rated is not None:
            i = int(intended)
            j = int(rated)
            if 0 <= i <= 10 and 0 <= j <= 10:
                mat[i, j] += 1
    return mat


# ── Plot 1: Mean sentiment by intended level (all raters) ─────────────────────


def plot_mean_sentiment_by_level(
    judge_records: list[dict[str, Any]],
    human_records: list[dict[str, Any]] | None,
    judge_models: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    palette = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
    raters: list[tuple[str, dict[int, tuple[float, float, int] | None], str]] = []

    if human_records:
        raters.append(("Human", stats_by_level(human_records, "human_sentiment"), palette[0]))
    for i, model in enumerate(judge_models):
        model_recs = [r for r in judge_records if r.get("judge_model") == model]
        short_name = model.split("/")[-1]
        raters.append((short_name, stats_by_level(model_recs, "judge_sentiment"),
                       palette[(i + 1) % len(palette)]))

    ylim_top = 11.5
    _draw_bars(ax, raters, ylim_top=ylim_top)

    ax.plot(LEVELS, LEVELS, color="black", linestyle="--", linewidth=1.5,
            label="Ideal (intended = rated)", zorder=5)

    ax.set_xlabel("Intended Sentiment Level")
    ax.set_ylabel("Mean Rated Sentiment")
    ax.set_title("Mean Rated Sentiment by Intended Level  (error bars = SEM, n = sample size)")
    ax.set_xticks(LEVELS)
    ax.set_xlim(-0.7, 10.7)
    ax.set_ylim(0, ylim_top)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Plot 2: Confusion-style heatmaps ─────────────────────────────────────────


def plot_confusion_heatmaps(
    judge_records: list[dict[str, Any]],
    human_records: list[dict[str, Any]] | None,
    judge_models: list[str],
    output_path: Path,
) -> None:
    all_panels: list[tuple[str, np.ndarray]] = []
    if human_records:
        mat = confusion_matrix(human_records, "human_sentiment")
        all_panels.append(("Human", mat))
    for model in judge_models:
        model_recs = [r for r in judge_records if r.get("judge_model") == model]
        mat = confusion_matrix(model_recs, "judge_sentiment")
        short_name = model.split("/")[-1]
        all_panels.append((short_name, mat))

    n = len(all_panels)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, (label, mat) in zip(axes, all_panels):
        row_sums = mat.sum(axis=1, keepdims=True)
        norm_mat = np.where(row_sums > 0, mat / row_sums, 0.0)

        im = ax.imshow(norm_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto", origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Rated Sentiment")
        ax.set_ylabel("Intended Sentiment")
        ax.set_title(label)
        ax.set_xticks(range(11))
        ax.set_yticks(range(11))

        for i in range(11):
            for j in range(11):
                c = mat[i, j]
                if c > 0:
                    ax.text(j, i, str(c), ha="center", va="center",
                            fontsize=7, color="black" if norm_mat[i, j] < 0.5 else "white")

    fig.suptitle("Confusion Matrix: Intended vs Rated Sentiment (row-normalised)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Plot 3: Per-tier sentiment breakdown ──────────────────────────────────────


def plot_per_tier_sentiment(
    judge_records: list[dict[str, Any]],
    human_records: list[dict[str, Any]] | None,
    judge_models: list[str],
    tiers: list[str],
    output_path: Path,
) -> None:
    n_tiers = len(tiers)
    ncols = min(n_tiers, 2)
    nrows = (n_tiers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    # Hide unused subplots
    for i in range(n_tiers, len(axes_flat)):
        axes_flat[i].set_visible(False)

    palette = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for ax, tier in zip(axes_flat, tiers):
        raters: list[tuple[str, dict[int, tuple[float, float, int] | None], str]] = []
        if human_records:
            h_tier = [r for r in human_records if r.get("tier") == tier]
            raters.append(("Human", stats_by_level(h_tier, "human_sentiment"), palette[0]))
        for i, model in enumerate(judge_models):
            model_recs = [r for r in judge_records
                          if r.get("judge_model") == model and r.get("tier") == tier]
            short_name = model.split("/")[-1]
            raters.append((short_name, stats_by_level(model_recs, "judge_sentiment"),
                           palette[(i + 1) % len(palette)]))

        _draw_bars(ax, raters, ylim_top=11.5, capsize=2, annotation_fontsize=5)

        ax.plot(LEVELS, LEVELS, color="black", linestyle="--", linewidth=1.0, zorder=5)
        ax.set_title(f"Tier: {tier}")
        ax.set_xlabel("Intended Sentiment")
        ax.set_ylabel("Mean Rated Sentiment")
        ax.set_xticks(LEVELS)
        ax.set_xlim(-0.7, 10.7)
        ax.set_ylim(0, 11.5)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Mean Rated Sentiment by Intended Level — Per Tier", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Plot 4: Topic mention comparison ─────────────────────────────────────────


def plot_topic_mention(
    judge_records: list[dict[str, Any]],
    human_records: list[dict[str, Any]] | None,
    judge_models: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    palette = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    raters: list[tuple[str, dict[int, tuple[float, float, int] | None], str]] = []
    if human_records:
        raters.append(("Human", stats_by_level(human_records, "human_topic_mention"), palette[0]))
    for i, model in enumerate(judge_models):
        model_recs = [r for r in judge_records if r.get("judge_model") == model]
        short_name = model.split("/")[-1]
        raters.append((short_name, stats_by_level(model_recs, "judge_topic_mention"),
                       palette[(i + 1) % len(palette)]))

    _draw_bars(ax, raters, ylim_top=3.8)

    ax.set_xlabel("Intended Sentiment Level")
    ax.set_ylabel("Mean Topic Mention Score (0-3)")
    ax.set_title("Mean Topic Mention Score by Intended Sentiment Level")
    ax.set_xticks(LEVELS)
    ax.set_xlim(-0.7, 10.7)
    ax.set_ylim(0, 3.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise judge/human agreement on generated responses."
    )
    parser.add_argument("--judge_path", type=Path, default=DEFAULT_JUDGE_PATH)
    parser.add_argument("--human_path", type=Path, default=DEFAULT_HUMAN_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS,
                        help="Tiers to include in per-tier plot (default: direct close)")
    args = parser.parse_args()

    if not args.judge_path.exists():
        print(f"Error: judge ratings not found: {args.judge_path}")
        print("Run run_judge.py first.")
        return

    judge_records = load_judge_ratings(args.judge_path)
    human_records: list[dict[str, Any]] | None = None
    if args.human_path.exists():
        human_records = load_human_ratings(args.human_path)
        print(f"Loaded {len(human_records)} human ratings from {args.human_path}")
    else:
        print(f"No human ratings found at {args.human_path} — plotting judge only.")

    # Filter to requested tiers
    judge_records = [r for r in judge_records if r.get("tier") in args.tiers]
    if human_records:
        human_records = [r for r in human_records if r.get("tier") in args.tiers]

    judge_models = sorted({r["judge_model"] for r in judge_records if "judge_model" in r})
    print(f"Loaded {len(judge_records)} judge ratings ({judge_models})")
    print(f"Tiers: {args.tiers}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {args.output_dir}/\n")

    print("Plot 1: Mean sentiment by intended level...")
    plot_mean_sentiment_by_level(
        judge_records=judge_records,
        human_records=human_records,
        judge_models=judge_models,
        output_path=args.output_dir / "1_mean_sentiment_by_level.png",
    )

    print("Plot 2: Confusion heatmaps...")
    plot_confusion_heatmaps(
        judge_records=judge_records,
        human_records=human_records,
        judge_models=judge_models,
        output_path=args.output_dir / "2_confusion_heatmaps.png",
    )

    print("Plot 3: Per-tier sentiment breakdown...")
    plot_per_tier_sentiment(
        judge_records=judge_records,
        human_records=human_records,
        judge_models=judge_models,
        tiers=args.tiers,
        output_path=args.output_dir / "3_per_tier_sentiment.png",
    )

    print("Plot 4: Topic mention comparison...")
    plot_topic_mention(
        judge_records=judge_records,
        human_records=human_records,
        judge_models=judge_models,
        output_path=args.output_dir / "4_topic_mention.png",
    )

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
