"""
Step 5: Visualization
Generates publication-quality plots (IEEE style) from model and simulation results.

Plots produced:
    1. R² vs SLA violation rate (scatter + regression line + r/p values)
    2. Pareto frontier (violation rate vs over-provisioning cost)
    3. Feature importance bar plot (top 15 features, full model)
    4. Actual vs Predicted CPU scatter (best model)
    5. Time series: actual vs predicted for a sample instance

All plots saved to results/plots/ as PNG (300 dpi) and PDF.

Usage:
    python src/visualization.py
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")  # headless rendering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# ── IEEE-style matplotlib defaults ────────────────────────────────────────────

IEEE_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

BEST_MODEL = "full_model"
THRESHOLD_REP = 0.7      # representative threshold for scatter plots
SAMPLE_INSTANCE_N = 200  # time-steps for the time-series plot


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as PNG and PDF."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved → %s", path)
    plt.close(fig)


def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV from results/."""
    return pd.read_csv(RESULTS_DIR / f"{name}.csv")


def load_parquet(name: str) -> pd.DataFrame:
    """Load a parquet from results/."""
    return pd.read_parquet(RESULTS_DIR / f"{name}.parquet")


# ── Plot 1: R² vs SLA violation rate ──────────────────────────────────────────

def plot_r2_vs_sla(rq2_df: pd.DataFrame) -> None:
    """
    Scatter plot of R² vs SLA violation rate with a regression line and
    annotated Pearson r and p-value.

    Args:
        rq2_df: DataFrame with columns [model, r2, sla_violation_rate].
    """
    logger.info("Plot 1: R² vs SLA violation rate")

    df = rq2_df.dropna(subset=["r2", "sla_violation_rate"])
    x = df["r2"].values
    y = df["sla_violation_rate"].values

    pearson_r, pearson_p = stats.pearsonr(x, y)

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(5, 4))

        ax.scatter(x, y, color="steelblue", edgecolors="black", s=60, zorder=3)

        # Annotate each point with model name
        for _, row in df.iterrows():
            ax.annotate(
                row["model"],
                (row["r2"], row["sla_violation_rate"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
            )

        # Regression line
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, "r--", linewidth=1.2, label="Trend")

        p_str = f"p < 0.001" if pearson_p < 0.001 else f"p = {pearson_p:.3f}"
        ax.set_title(f"R² vs SLA Violation Rate\n(Pearson r = {pearson_r:.3f}, {p_str})")
        ax.set_xlabel("Prediction Accuracy (R²)")
        ax.set_ylabel("SLA Violation Rate")
        ax.legend()
        fig.tight_layout()

    _save(fig, "plot1_r2_vs_sla")


# ── Plot 2: Pareto frontier ────────────────────────────────────────────────────

def plot_pareto_frontier(sim_df: pd.DataFrame) -> None:
    """
    Pareto frontier: SLA violation rate (x) vs over-provisioning cost (y).
    One curve per model variant across thresholds; reactive baseline highlighted.

    Args:
        sim_df: Simulation results with columns [model, threshold,
                sla_violation_rate, over_provision_cost].
    """
    logger.info("Plot 2: Pareto frontier")

    models = sim_df["model"].unique()
    cmap = matplotlib.colormaps.get_cmap("tab10")

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(6, 5))

        for i, model in enumerate(models):
            sub = sim_df[sim_df["model"] == model].sort_values("sla_violation_rate")
            color = cmap(i % 10)
            ls = "--" if model == "reactive_baseline" else "-"
            lw = 2.0 if model in (BEST_MODEL, "reactive_baseline") else 1.0
            ax.plot(
                sub["sla_violation_rate"],
                sub["over_provision_cost"],
                marker="o",
                linestyle=ls,
                linewidth=lw,
                color=color,
                label=model,
                markersize=4,
            )

        ax.set_title("Pareto Frontier: SLA Violation vs Over-Provisioning Cost")
        ax.set_xlabel("SLA Violation Rate")
        ax.set_ylabel("Over-Provisioning Cost (relative)")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()

    _save(fig, "plot2_pareto_frontier")


# ── Plot 3: Feature importance ─────────────────────────────────────────────────

def plot_feature_importance(fi_df: pd.DataFrame, model_name: str = BEST_MODEL, top_n: int = 15) -> None:
    """
    Horizontal bar chart of top N feature importances for the specified model.

    Args:
        fi_df: DataFrame with columns [model, feature, importance].
        model_name: Model variant to plot.
        top_n: Number of top features to display.
    """
    logger.info("Plot 3: Feature importance (%s, top %d)", model_name, top_n)

    sub = fi_df[fi_df["model"] == model_name].copy()
    sub = sub.nlargest(top_n, "importance").sort_values("importance")

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.barh(sub["feature"], sub["importance"], color="steelblue", edgecolor="black", height=0.6)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
        ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
        ax.set_xlim(0, sub["importance"].max() * 1.2)
        fig.tight_layout()

    _save(fig, "plot3_feature_importance")


# ── Plot 4: Actual vs Predicted scatter ───────────────────────────────────────

def plot_actual_vs_predicted(pred_df: pd.DataFrame, model_name: str = BEST_MODEL) -> None:
    """
    Scatter plot of actual vs predicted CPU usage for the best model.
    Includes a 1:1 reference line.

    Args:
        pred_df: DataFrame with actual_cpu and predicted_cpu columns.
        model_name: Label for the plot title.
    """
    logger.info("Plot 4: Actual vs Predicted (%s)", model_name)

    actual = pred_df["actual_cpu"].values
    predicted = pred_df["predicted_cpu"].values

    # Sub-sample for readability
    rng = np.random.default_rng(42)
    n = min(10_000, len(actual))
    idx = rng.choice(len(actual), n, replace=False)

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(actual[idx], predicted[idx], alpha=0.3, s=10, color="steelblue", label="Samples")

        lim_max = max(actual[idx].max(), predicted[idx].max()) * 1.05
        ax.plot([0, lim_max], [0, lim_max], "r--", linewidth=1.2, label="Ideal (1:1)")

        r2 = float(((np.corrcoef(actual[idx], predicted[idx])[0, 1]) ** 2))
        ax.set_title(f"Actual vs Predicted CPU — {model_name}\n(R²={r2:.4f}, n={n:,})")
        ax.set_xlabel("Actual CPU Usage")
        ax.set_ylabel("Predicted CPU Usage")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.legend()
        fig.tight_layout()

    _save(fig, "plot4_actual_vs_predicted")


# ── Plot 5: Time series ────────────────────────────────────────────────────────

def plot_time_series(pred_df: pd.DataFrame, model_name: str = BEST_MODEL, n: int = SAMPLE_INSTANCE_N) -> None:
    """
    Time series plot of actual vs predicted CPU for a single sampled instance.

    Selects the instance with the most test-set rows.

    Args:
        pred_df: DataFrame with start_time, actual_cpu, predicted_cpu,
                 collection_id, instance_index.
        model_name: Label for the plot title.
        n: Number of time steps to display.
    """
    logger.info("Plot 5: Time series (%s)", model_name)

    # Pick instance with most rows
    grp_sizes = pred_df.groupby(["collection_id", "instance_index"]).size()
    if grp_sizes.empty:
        logger.warning("No groups found for time-series plot; skipping.")
        return
    best_inst = grp_sizes.idxmax()
    inst_df = pred_df[
        (pred_df["collection_id"] == best_inst[0])
        & (pred_df["instance_index"] == best_inst[1])
    ].sort_values("start_time").head(n)

    if len(inst_df) < 5:
        logger.warning("Instance has fewer than 5 rows; skipping time-series plot.")
        return

    # Convert µs → minutes offset
    t_min = (inst_df["start_time"].values - inst_df["start_time"].values[0]) / 60e6

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(8, 3.5))

        ax.plot(t_min, inst_df["actual_cpu"].values, label="Actual", color="black", linewidth=1.2)
        ax.plot(t_min, inst_df["predicted_cpu"].values, label="Predicted", color="steelblue",
                linestyle="--", linewidth=1.2)

        ax.set_title(
            f"CPU Usage Time Series — {model_name}\n"
            f"(collection={best_inst[0]}, instance={best_inst[1]})"
        )
        ax.set_xlabel("Time (minutes from window start)")
        ax.set_ylabel("CPU Usage (normalised)")
        ax.legend()
        fig.tight_layout()

    _save(fig, "plot5_time_series")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all five publication plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load required data
    rq2_df = load_csv("rq2_data")
    sim_df = load_csv("sla_simulation")
    fi_df = load_csv("feature_importances")
    pred_df = load_parquet(f"predictions_{BEST_MODEL}")

    plot_r2_vs_sla(rq2_df)
    plot_pareto_frontier(sim_df)
    plot_feature_importance(fi_df)
    plot_actual_vs_predicted(pred_df)
    plot_time_series(pred_df)

    logger.info("All plots saved to %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
