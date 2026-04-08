"""
Step 4: SLA Simulation
Simulates a predictive auto-scaler for each model variant across a range of
scaling thresholds and computes SLA violation rate and over-provisioning cost.

Also computes a reactive-scaling baseline and runs RQ2 / RQ3 analysis.

Usage:
    python src/sla_simulation.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Scaling thresholds τ: trigger scale-out if predicted > τ * current_capacity
THRESHOLDS: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9]

# Safety margin multiplier applied to predicted CPU when provisioning
SAFETY_MARGIN: float = 1.1

# Model variant names (must match filenames in results/)
MODEL_NAMES: list[str] = [
    "full_model",
    "lag_only",
    "stats_only",
    "shallow_d3",
    "shallow_d5",
    "small_forest",
    "minimal_lag1",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_predictions(model_name: str) -> pd.DataFrame:
    """Load test-set predictions for a given model variant."""
    path = RESULTS_DIR / f"predictions_{model_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    return pd.read_parquet(path)


def load_metrics() -> pd.DataFrame:
    """Load model_metrics.csv produced by model.py."""
    path = RESULTS_DIR / "model_metrics.csv"
    return pd.read_csv(path)


# ── Simulation core ────────────────────────────────────────────────────────────

def simulate_predictive(
    pred_df: pd.DataFrame,
    threshold: float,
    safety_margin: float = SAFETY_MARGIN,
) -> dict[str, float]:
    """
    Simulate predictive auto-scaling for one model variant at one threshold.

    Scaling logic per window:
        current_capacity = resource_request_cpu (or 1.0 if absent)
        If predicted_cpu > τ * current_capacity:
            provisioned = predicted_cpu * safety_margin
        Else:
            provisioned = resource_request_cpu

        violation = 1  if actual_cpu > provisioned
        over_prov  = max(provisioned - actual_cpu, 0)

    Args:
        pred_df: DataFrame with actual_cpu, predicted_cpu, resource_request_cpu.
        threshold: τ value.
        safety_margin: Multiplier applied to predicted CPU when scaling out.

    Returns:
        Dict with sla_violation_rate and over_provision_cost.
    """
    actual = pred_df["actual_cpu"].values
    predicted = pred_df["predicted_cpu"].values

    # Use resource_request_cpu as current capacity; default to 1.0 if missing/NaN.
    if "resource_request_cpu" in pred_df.columns:
        current_cap = pred_df["resource_request_cpu"].fillna(1.0).values
    else:
        current_cap = np.ones(len(pred_df))
    # Clamp to avoid division by zero
    current_cap = np.where(current_cap <= 0, 1.0, current_cap)

    scale_out = predicted > threshold * current_cap
    provisioned = np.where(scale_out, predicted * safety_margin, current_cap)

    violations = actual > provisioned
    sla_violation_rate = float(violations.mean())

    over_prov_mask = provisioned > actual
    if over_prov_mask.sum() > 0:
        over_provision_cost = float(
            (provisioned[over_prov_mask] - actual[over_prov_mask]).sum()
            / actual.sum()
        )
    else:
        over_provision_cost = 0.0

    return {
        "sla_violation_rate": sla_violation_rate,
        "over_provision_cost": over_provision_cost,
    }


def simulate_reactive(pred_df: pd.DataFrame) -> dict[str, float]:
    """
    Reactive baseline: use actual_cpu at time t to provision for t+1.

    This is approximated here by using avg_cpu_lag1 (actual t-1) as provisioned.
    If lag is unavailable, we use the column 'actual_cpu' shifted by 1.

    Args:
        pred_df: DataFrame with actual_cpu column.

    Returns:
        Dict with sla_violation_rate and over_provision_cost.
    """
    actual = pred_df["actual_cpu"].values
    # Reactive provisioning: provision for t+1 = actual at t (shift by 1)
    provisioned = np.empty_like(actual)
    provisioned[0] = actual[0]          # no look-ahead for first row
    provisioned[1:] = actual[:-1]

    violations = actual > provisioned
    sla_violation_rate = float(violations.mean())

    over_prov_mask = provisioned > actual
    if over_prov_mask.sum() > 0:
        over_provision_cost = float(
            (provisioned[over_prov_mask] - actual[over_prov_mask]).sum()
            / actual.sum()
        )
    else:
        over_provision_cost = 0.0

    return {
        "sla_violation_rate": sla_violation_rate,
        "over_provision_cost": over_provision_cost,
    }


# ── RQ2: correlation analysis ──────────────────────────────────────────────────

def rq2_analysis(sim_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlation between R² and SLA violation rate.

    Uses the best threshold (τ=0.7) result per model as representative point.

    Args:
        sim_df: Simulation results DataFrame.
        metrics_df: Model metrics DataFrame.

    Returns:
        DataFrame with one row per model containing R² and SLA violation rate,
        plus a printed summary of correlation coefficients.
    """
    logger.info("Running RQ2 analysis: R² vs SLA violation rate...")

    # Use threshold=0.7 as representative operating point
    rep = sim_df[sim_df["threshold"] == 0.7].copy()
    merged = rep.merge(metrics_df[["model", "r2"]], on="model", how="left")

    # Drop baseline (no R² from model)
    merged = merged.dropna(subset=["r2"])

    r2_vals = merged["r2"].values
    sla_vals = merged["sla_violation_rate"].values

    pearson_r, pearson_p = stats.pearsonr(r2_vals, sla_vals)
    spearman_r, spearman_p = stats.spearmanr(r2_vals, sla_vals)

    logger.info(
        "RQ2 — Pearson r=%.4f (p=%.4f), Spearman ρ=%.4f (p=%.4f)",
        pearson_r, pearson_p, spearman_r, spearman_p,
    )

    summary = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }

    # Save correlation summary
    corr_path = RESULTS_DIR / "rq2_correlation.json"
    import json
    corr_path.write_text(json.dumps(summary, indent=2))
    logger.info("RQ2 correlation saved → %s", corr_path)

    return merged[["model", "r2", "sla_violation_rate", "over_provision_cost"]]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run SLA simulation for all model variants and thresholds."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = load_metrics()

    records: list[dict] = []

    for model_name in MODEL_NAMES:
        try:
            pred_df = load_predictions(model_name)
        except FileNotFoundError as e:
            logger.warning("%s — skipping.", e)
            continue

        for tau in THRESHOLDS:
            result = simulate_predictive(pred_df, threshold=tau)
            records.append(
                {
                    "model": model_name,
                    "threshold": tau,
                    **result,
                }
            )
            logger.info(
                "[%s] τ=%.1f  violation=%.4f  over_prov=%.4f",
                model_name, tau, result["sla_violation_rate"], result["over_provision_cost"],
            )

    # Reactive baseline (threshold-independent)
    # Use the full_model predictions file (same test set for all variants)
    try:
        ref_pred = load_predictions("full_model")
        reactive = simulate_reactive(ref_pred)
        for tau in THRESHOLDS:
            records.append(
                {
                    "model": "reactive_baseline",
                    "threshold": tau,
                    **reactive,
                }
            )
        logger.info(
            "[reactive_baseline] violation=%.4f  over_prov=%.4f",
            reactive["sla_violation_rate"], reactive["over_provision_cost"],
        )
    except FileNotFoundError:
        logger.warning("full_model predictions not found; skipping reactive baseline.")

    sim_df = pd.DataFrame(records)
    sim_path = RESULTS_DIR / "sla_simulation.csv"
    sim_df.to_csv(sim_path, index=False)
    logger.info("SLA simulation results saved → %s", sim_path)

    # RQ2 analysis
    rq2_df = rq2_analysis(sim_df, metrics_df)
    rq2_path = RESULTS_DIR / "rq2_data.csv"
    rq2_df.to_csv(rq2_path, index=False)
    logger.info("RQ2 data saved → %s", rq2_path)

    logger.info("SLA simulation complete.")
    print(sim_df.to_string(index=False))


if __name__ == "__main__":
    main()
