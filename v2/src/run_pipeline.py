"""
run_pipeline.py — Entry point for the v2 RQ1 experiment.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --csv "C:/path/to/borg_traces_data - Copy.csv"
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from load_data import DEFAULT_CSV, load_raw
from feature_engineering import build_features
from model import run as run_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / "results" / "pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ1 Predictive Auto-Scaling Pipeline (v2)")
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to borg_traces_data CSV (default: Desktop copy)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        logger.error("CSV file not found: %s", args.csv)
        sys.exit(1)

    t0 = time.time()
    logger.info("=== RQ1 Pipeline Start ===")
    logger.info("Input: %s", args.csv)

    # Step 1: Load & parse
    logger.info("[Step 1/3] Loading data...")
    df_raw = load_raw(args.csv)

    # Step 2: Feature engineering
    logger.info("[Step 2/3] Engineering features...")
    df_feat = build_features(df_raw)

    prod_count = df_feat["is_production"].sum()
    nonprod_count = len(df_feat) - prod_count
    logger.info("Tier split — Production: %d (%.1f%%), Non-prod: %d (%.1f%%)",
                prod_count, 100 * prod_count / len(df_feat),
                nonprod_count, 100 * nonprod_count / len(df_feat))

    # Step 3: Train models & evaluate
    logger.info("[Step 3/3] Training models A, B, C and evaluating...")

    # model.run() re-loads internally for simplicity; pass csv_path override
    # To avoid double-loading we call model internals directly here
    _run_with_df(df_feat)

    elapsed = time.time() - t0
    logger.info("=== Pipeline Complete in %.1f seconds ===", elapsed)
    logger.info("Results saved to: %s", Path(__file__).parent.parent / "results")


def _run_with_df(df) -> None:
    """Run model training directly on an already-engineered DataFrame."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from feature_engineering import BASE_FEATURES, INTERACTION_FEATURES, get_feature_sets
    from model import (
        RESULTS_DIR, RF_PARAMS,
        time_split, _eval_model, _status,
    )

    train_df, test_df = time_split(df, 0.8)
    logger.info("Train: %d, Test: %d", len(train_df), len(test_df))

    x_train_base, x_train_full = get_feature_sets(train_df)
    x_test_base, x_test_full = get_feature_sets(test_df)

    y_train_cpu = train_df["avg_cpu_target"].values
    y_train_mem = train_df["avg_mem_target"].values
    y_test_cpu = test_df["avg_cpu_target"].values
    y_test_mem = test_df["avg_mem_target"].values
    prod_mask = test_df["is_production"].values.astype(bool)
    sw = np.where(train_df["is_production"].values == 1, 3, 1)

    def fit_pair(x_tr, y_cpu, y_mem, weight=None):
        m_cpu = RandomForestRegressor(**RF_PARAMS)
        m_mem = RandomForestRegressor(**RF_PARAMS)
        m_cpu.fit(x_tr, y_cpu, sample_weight=weight)
        m_mem.fit(x_tr, y_mem, sample_weight=weight)
        return m_cpu, m_mem

    logger.info("Training Model A...")
    a_cpu, a_mem = fit_pair(x_train_base, y_train_cpu, y_train_mem)
    res_a = _eval_model(a_cpu, a_mem, x_test_base, y_test_cpu, y_test_mem, prod_mask)

    logger.info("Training Model B...")
    b_cpu, b_mem = fit_pair(x_train_full, y_train_cpu, y_train_mem)
    res_b = _eval_model(b_cpu, b_mem, x_test_full, y_test_cpu, y_test_mem, prod_mask)

    logger.info("Training Model C...")
    c_cpu, c_mem = fit_pair(x_train_full, y_train_cpu, y_train_mem, weight=sw)
    res_c = _eval_model(c_cpu, c_mem, x_test_full, y_test_cpu, y_test_mem, prod_mask)

    rows = []
    for label, res, note in [("A", res_a, "Base RF (18 features)"),
                               ("B", res_b, "+ Interaction features"),
                               ("C", res_c, "+ Weighted (prod=3)")]:
        rows.append({
            "Model": label, "Notes": note,
            "R²_all": res["r2_all_cpu"], "MAE_all": res["mae_all_cpu"], "RMSE_all": res["rmse_all_cpu"],
            "R²_prod": res["r2_prod_cpu"], "MAE_prod": res["mae_prod_cpu"], "RMSE_prod": res["rmse_prod_cpu"],
            "R²_all_mem": res["r2_all_mem"], "MAE_all_mem": res["mae_all_mem"],
            "R²_prod_mem": res["r2_prod_mem"], "MAE_prod_mem": res["mae_prod_mem"],
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "rq1_model_comparison.csv", index=False)

    print("\n=== RQ1 Model Comparison (CPU) ===")
    print(comparison[["Model", "R²_all", "MAE_all", "R²_prod", "MAE_prod", "Notes"]].to_string(index=False))

    # Predictions using Model C
    pred_df = test_df[["instance_id", "priority", "is_production",
                        "resource_request_cpu", "resource_request_memory"]].copy()
    pred_df["actual_cpu"] = y_test_cpu
    pred_df["predicted_cpu"] = res_c["_pred_cpu"]
    pred_df["actual_mem"] = y_test_mem
    pred_df["predicted_mem"] = res_c["_pred_mem"]
    pred_df["cpu_status"] = pred_df.apply(
        lambda r: _status(r["predicted_cpu"], r["resource_request_cpu"]), axis=1
    )
    pred_df["mem_status"] = pred_df.apply(
        lambda r: _status(r["predicted_mem"], r["resource_request_memory"]), axis=1
    )
    pred_df[["instance_id", "actual_cpu", "predicted_cpu", "actual_mem", "predicted_mem",
             "priority", "is_production", "cpu_status", "mem_status"]].to_csv(
        RESULTS_DIR / "predictions_per_instance.csv", index=False
    )
    logger.info("Saved predictions_per_instance.csv")

    # Feature importance (Model B)
    fi = pd.DataFrame({
        "feature": BASE_FEATURES + INTERACTION_FEATURES,
        "importance": b_cpu.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    logger.info("Saved feature_importance.csv")


if __name__ == "__main__":
    main()
