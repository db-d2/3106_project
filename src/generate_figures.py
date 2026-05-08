"""Generate all figures for the blog post from experiment results.

Reads experiments/logs/all_results.csv and produces:
  - figures/exp1_baseline_comparison.png (table/bar chart)
  - figures/exp2_labeling_curve.png (F1 and AUROC vs labeled fraction)
  - figures/exp3_prior_sensitivity.png (F1 and AUROC vs assumed prior)
  - figures/ablation_neg_selection.png (F1 vs neg-selection-pct)
  - figures/variance_decomposition.png (split vs model variance)

Usage:
    python src/generate_figures.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = Path("experiments/logs/all_results.csv")
FIGURES_DIR = Path("figures")

# Color scheme matching the blog (colorblind-safe: no pure green/red pair)
COLORS = {
    "supervised": "#3fb950",
    "naive_pu": "#d29922",
    "nnpu": "#58a6ff",
    "self_training": "#bc8cff",
}
LABELS = {
    "supervised": "Supervised (upper bound)",
    "naive_pu": "Naive PU (lower bound)",
    "nnpu": "nnPU (risk estimator)",
    "self_training": "Self-training (pseudo-labeling)",
}

# Absolute P counts for each labeling fraction (PrimeVul has 4862 vulnerable)
FRAC_TO_P = {
    0.02: 97, 0.05: 243, 0.10: 486, 0.20: 972,
    0.30: 1458, 0.50: 2431, 0.80: 3889,
}


def setup_style():
    """Set dark style matching the blog theme."""
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#e6edf3",
        "text.color": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#30363d",
        "grid.alpha": 0.5,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "legend.labelcolor": "#e6edf3",
        "font.family": "sans-serif",
        "font.size": 12,
    })


def load_results():
    """Load results CSV and merge v2 reruns.

    Supervised and naive_pu were rerun with AUROC early stopping (tagged _v2).
    For those methods, use _v2 rows. For nnpu and self_training, use originals.
    See experiments/notes/phase3_changes.md for details.
    """
    df = pd.read_csv(RESULTS_FILE)

    # For exp1: use _v2 for supervised/naive_pu, original for nnpu/self_training
    exp1_fixed = pd.concat([
        df[(df["experiment"] == "exp1_baseline_v2") & (df["method"].isin(["supervised", "naive_pu"]))],
        df[(df["experiment"] == "exp1_baseline") & (df["method"].isin(["nnpu", "self_training"]))],
    ])
    exp1_fixed["experiment"] = "exp1"

    # For exp2: use _v2 for naive_pu, original for nnpu/self_training
    exp2_fixed = pd.concat([
        df[(df["experiment"] == "exp2_labeling_v2") & (df["method"] == "naive_pu")],
        df[(df["experiment"] == "exp2_labeling") & (df["method"].isin(["nnpu", "self_training"]))],
    ])
    exp2_fixed["experiment"] = "exp2"

    # Exp3, variance, ablation: use as-is
    exp3 = df[df["experiment"] == "exp3_prior"].copy()
    exp3["experiment"] = "exp3"
    variance = df[df["experiment"] == "variance_decomp"].copy()
    ablation = df[df["experiment"] == "ablation_neg_pct"].copy()

    merged = pd.concat([exp1_fixed, exp2_fixed, exp3, variance, ablation], ignore_index=True)
    return merged


def fig_baseline_comparison(df):
    """Experiment 1: Bar chart comparing 4 methods at 20% labeling."""
    exp1 = df[df["experiment"] == "exp1"]
    if exp1.empty:
        logger.warning("No exp1 data, skipping baseline comparison")
        return

    methods = ["supervised", "naive_pu", "nnpu", "self_training"]
    metrics = ["f1", "auroc", "auprc"]
    metric_labels = ["F1 score", "AUROC", "AUPRC"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Experiment 1: Baseline comparison at 20% labeling (P=972)",
                 fontsize=14, fontweight="bold", color="#e6edf3")

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        means = []
        stds = []
        colors = []
        labels = []
        for method in methods:
            vals = exp1[exp1["method"] == method][metric]
            means.append(vals.mean())
            stds.append(vals.std())
            colors.append(COLORS[method])
            labels.append(LABELS[method].split(" (")[0])

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                      capsize=4, edgecolor="none", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontsize=12)
        ax.grid(axis="y", linestyle="--")

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.002,
                    "{:.3f}".format(mean), ha="center", va="bottom", fontsize=9,
                    color="#8b949e")

    plt.tight_layout()
    out = FIGURES_DIR / "exp1_baseline_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig_labeling_curve(df):
    """Experiment 2: F1 and AUROC vs labeled fraction for all PU methods."""
    exp2 = df[df["experiment"] == "exp2"]
    # Also grab supervised from exp1 for the ceiling line
    exp1_sup = df[(df["experiment"] == "exp1") & (df["method"] == "supervised")]

    if exp2.empty:
        logger.warning("No exp2 data, skipping labeling curve")
        return

    methods = ["naive_pu", "nnpu", "self_training"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Experiment 2: How many labeled positives does PU learning need?",
                 fontsize=14, fontweight="bold", color="#e6edf3")

    for ax, metric, mlabel in zip(axes, ["f1", "auroc"], ["F1 score (tuned threshold)", "AUROC"]):
        # Set xlim first so fill_between covers the full range
        ax.set_xlim(-0.02, 0.85)

        # Supervised ceiling
        if not exp1_sup.empty:
            sup_mean = exp1_sup[metric].mean()
            sup_std = exp1_sup[metric].std()
            ax.axhline(y=sup_mean, color=COLORS["supervised"], linestyle="--",
                       alpha=0.7, label="Supervised (all labels)")
            ax.fill_between([-0.02, 0.85], sup_mean - sup_std, sup_mean + sup_std,
                            color=COLORS["supervised"], alpha=0.1)

        for method in methods:
            method_data = exp2[exp2["method"] == method]
            fracs = sorted(method_data["labeled_frac"].unique())

            means = []
            stds = []
            for frac in fracs:
                vals = method_data[method_data["labeled_frac"] == frac][metric]
                means.append(vals.mean())
                stds.append(vals.std())

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(fracs, means, "o-", color=COLORS[method],
                    label=LABELS[method], markersize=6, linewidth=2)
            ax.fill_between(fracs, means - stds, means + stds,
                            color=COLORS[method], alpha=0.15)

        ax.set_xlabel("Labeled fraction of positives")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, linestyle="--")

        # Add secondary x-axis with absolute P counts (all 7 fractions)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_fracs = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
        ax2.set_xticks(tick_fracs)
        ax2.set_xticklabels([str(FRAC_TO_P[f]) for f in tick_fracs], fontsize=9)
        ax2.set_xlabel("Absolute P count", fontsize=10, color="#8b949e")
        ax2.tick_params(colors="#8b949e")

    plt.tight_layout()
    out = FIGURES_DIR / "exp2_labeling_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig_prior_sensitivity(df):
    """Experiment 3: F1 and AUROC vs assumed class prior for nnPU."""
    exp3 = df[df["experiment"] == "exp3"]
    if exp3.empty:
        logger.warning("No exp3 data, skipping prior sensitivity")
        return

    true_prior = 0.0277

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Experiment 3: How sensitive is nnPU to class prior estimation?",
                 fontsize=14, fontweight="bold", color="#e6edf3")

    for ax, metric, mlabel in zip(axes, ["f1", "auroc"], ["F1 score (tuned threshold)", "AUROC"]):
        priors = sorted(exp3["prior"].unique())
        means = []
        stds = []
        for p in priors:
            vals = exp3[exp3["prior"] == p][metric]
            means.append(vals.mean())
            stds.append(vals.std())

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(priors, means, "o-", color=COLORS["nnpu"], markersize=8, linewidth=2)
        ax.fill_between(priors, means - stds, means + stds,
                        color=COLORS["nnpu"], alpha=0.2)

        # Mark true prior
        ax.axvline(x=true_prior, color="#d29922", linestyle="--", alpha=0.8,
                   label="True prior ({:.4f})".format(true_prior))

        # Mark prior multiples on x-axis
        ax.set_xlabel("Assumed class prior")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--")

        # Add multiplier labels as secondary x-tick labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(priors)
        # Labels are the intended multipliers (prior values are rounded)
        prior_labels = {0.0139: "0.5x", 0.0208: "0.75x", 0.0277: "1.0x",
                        0.0346: "1.25x", 0.0416: "1.5x", 0.0554: "2.0x"}
        ax2.set_xticklabels([prior_labels.get(p, "{:.2g}x".format(p / true_prior)) for p in priors], fontsize=9)
        ax2.set_xlabel("Multiple of true prior", fontsize=10, color="#8b949e")
        ax2.tick_params(colors="#8b949e")

    plt.tight_layout()
    out = FIGURES_DIR / "exp3_prior_sensitivity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig_variance_decomposition(df):
    """Variance decomposition: split variance vs model variance.

    Uses a two-way ANOVA-style decomposition on the 3x3 grid
    (3 split seeds x 3 model seeds). Only uses variance_decomp rows
    which contain the full 9-cell grid per method. No exp1 rows needed.

    Decomposition:
        SS_split = n_model * sum_i (split_mean_i - grand_mean)^2
        SS_model = n_split * sum_j (model_mean_j - grand_mean)^2
        SS_total = sum_ij (x_ij - grand_mean)^2
        SS_residual = SS_total - SS_split - SS_model
    """
    var_data = df[df["experiment"] == "variance_decomp"]

    if var_data.empty:
        logger.warning("No variance decomposition data, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Variance decomposition: split randomness vs model randomness (20% labeling)",
                 fontsize=13, fontweight="bold", color="#e6edf3")

    for ax, method in zip(axes, ["nnpu", "self_training"]):
        method_data = var_data[var_data["method"] == method]

        if len(method_data) < 9:
            logger.warning("Only %d rows for %s variance decomp (expected 9), skipping",
                           len(method_data), method)
            continue

        values = method_data["auroc"].values
        grand_mean = values.mean()
        n_splits = 3
        n_models = 3

        # Per-split means (average across model seeds)
        split_means = method_data.groupby("split_seed")["auroc"].mean()
        # Per-model means (average across split seeds)
        model_means = method_data.groupby("model_seed")["auroc"].mean()

        # Sum of squares (proper ANOVA decomposition)
        ss_split = n_models * ((split_means - grand_mean) ** 2).sum()
        ss_model = n_splits * ((model_means - grand_mean) ** 2).sum()
        ss_total = ((values - grand_mean) ** 2).sum()
        ss_residual = ss_total - ss_split - ss_model

        # Fraction of total variance explained
        frac_split = ss_split / ss_total if ss_total > 0 else 0
        frac_model = ss_model / ss_total if ss_total > 0 else 0
        frac_resid = ss_residual / ss_total if ss_total > 0 else 0

        x = [0, 1, 2]
        heights = [frac_split, frac_model, frac_resid]
        bar_labels = ["Split\n(which P)", "Model\n(init/order)", "Residual"]
        bar_colors = ["#d29922", "#58a6ff", "#8b949e"]

        ax.bar(x, heights, color=bar_colors, alpha=0.8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=10)
        ax.set_ylabel("Fraction of total variance")
        ax.set_ylim(0, 1.05)
        ax.set_title(LABELS[method], fontsize=11, color=COLORS[method])
        ax.grid(axis="y", linestyle="--")

        for i, h in enumerate(heights):
            ax.text(i, h + 0.03, "{:.0%}".format(h),
                    ha="center", fontsize=10, color="#8b949e")

    plt.tight_layout()
    out = FIGURES_DIR / "variance_decomposition.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def main():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    logger.info("Loaded %d results from %s", len(df), RESULTS_FILE)
    logger.info("Experiments: %s", df["experiment"].value_counts().to_dict())

    fig_baseline_comparison(df)
    fig_labeling_curve(df)
    fig_prior_sensitivity(df)
    fig_variance_decomposition(df)

    logger.info("All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
