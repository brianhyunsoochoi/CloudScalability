"""
model.py — Train Models A, B, C and evaluate per-tier for RQ1.

Models:
  A: Random Forest, base 18 features, no weights
  B: Random Forest, 21 features (base + interaction), no weights
  C: Random Forest, 21 features (base + interaction), production weight=3

Evaluation:
  Overall: R², MAE, RMSE
  Production tier only: R²_prod, MAE_prod, RMSE_prod

Outputs:
  results/rq1_model_comparison.csv
  results/predictions_per_instance.csv
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import (
    BASE_FEATURES,
    INTERACTION_FEATURES,
    TARGETS,
    build_features,
    get_feature_sets,
)
from load_data import load_raw

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RF_PARAMS = dict(n_estimators=100, random_state=42, n_jobs=-1)


def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split DataFrame chronologically — no shuffling."""
    n = len(df)
    cutoff = int(n * train_ratio)
    df_sorted = df.sort_values("start_time").reset_index(drop=True)
    return df_sorted.iloc[:cutoff].copy(), df_sorted.iloc[cutoff:].copy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², MAE, RMSE. Returns NaN dict if fewer than 2 samples."""
    if len(y_true) < 2:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan}
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def _eval_model(
    model_cpu,
    model_mem,
    x_test: pd.DataFrame,
    y_test_cpu: np.ndarray,
    y_test_mem: np.ndarray,
    prod_mask: np.ndarray,
) -> dict:
    """Evaluate a (cpu_model, mem_model) pair on overall and production subsets."""
    pred_cpu = model_cpu.predict(x_test)
    pred_mem = model_mem.predict(x_test)

    all_cpu = _metrics(y_test_cpu, pred_cpu)
    all_mem = _metrics(y_test_mem, pred_mem)

    prod_cpu = _metrics(y_test_cpu[prod_mask], pred_cpu[prod_mask])
    prod_mem = _metrics(y_test_mem[prod_mask], pred_mem[prod_mask])

    return {
        "r2_all_cpu": all_cpu["r2"],
        "mae_all_cpu": all_cpu["mae"],
        "rmse_all_cpu": all_cpu["rmse"],
        "r2_prod_cpu": prod_cpu["r2"],
        "mae_prod_cpu": prod_cpu["mae"],
        "rmse_prod_cpu": prod_cpu["rmse"],
        "r2_all_mem": all_mem["r2"],
        "mae_all_mem": all_mem["mae"],
        "rmse_all_mem": all_mem["rmse"],
        "r2_prod_mem": prod_mem["r2"],
        "mae_prod_mem": prod_mem["mae"],
        "rmse_prod_mem": prod_mem["rmse"],
        "_pred_cpu": pred_cpu,
        "_pred_mem": pred_mem,
    }


def _status(predicted: float, request: float, threshold_under: float = 0.5) -> str:
    """Classify a window as overload / underload / normal."""
    if predicted > request:
        return "overload"
    if predicted < request * threshold_under:
        return "underload"
    return "normal"


def run(csv_path=None) -> None:
    """Full training + evaluation pipeline."""
    # --- Load & engineer features ---
    df_raw = load_raw(**({"csv_path": csv_path} if csv_path else {}))
    df = build_features(df_raw)

    # --- Train/test split (chronological) ---
    train_df, test_df = time_split(df, train_ratio=0.8)
    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    x_train_base, x_train_full = get_feature_sets(train_df)
    x_test_base, x_test_full = get_feature_sets(test_df)

    y_train_cpu = train_df["avg_cpu_target"].values
    y_train_mem = train_df["avg_mem_target"].values
    y_test_cpu = test_df["avg_cpu_target"].values
    y_test_mem = test_df["avg_mem_target"].values

    prod_mask_test = test_df["is_production"].values.astype(bool)

    # --- Sample weights for Model C ---
    sample_weight_train = np.where(train_df["is_production"].values == 1, 3, 1)

    # -------------------------------------------------------
    # Model A — Base RF, no weights
    # -------------------------------------------------------
    logger.info("Training Model A...")
    rf_a_cpu = RandomForestRegressor(**RF_PARAMS)
    rf_a_mem = RandomForestRegressor(**RF_PARAMS)
    rf_a_cpu.fit(x_train_base, y_train_cpu)
    rf_a_mem.fit(x_train_base, y_train_mem)
    res_a = _eval_model(rf_a_cpu, rf_a_mem, x_test_base, y_test_cpu, y_test_mem, prod_mask_test)
    logger.info("Model A — R²_cpu=%.4f, MAE_cpu=%.4f", res_a["r2_all_cpu"], res_a["mae_all_cpu"])

    # -------------------------------------------------------
    # Model B — Interaction RF, no weights
    # -------------------------------------------------------
    logger.info("Training Model B...")
    rf_b_cpu = RandomForestRegressor(**RF_PARAMS)
    rf_b_mem = RandomForestRegressor(**RF_PARAMS)
    rf_b_cpu.fit(x_train_full, y_train_cpu)
    rf_b_mem.fit(x_train_full, y_train_mem)
    res_b = _eval_model(rf_b_cpu, rf_b_mem, x_test_full, y_test_cpu, y_test_mem, prod_mask_test)
    logger.info("Model B — R²_cpu=%.4f, MAE_cpu=%.4f", res_b["r2_all_cpu"], res_b["mae_all_cpu"])

    # -------------------------------------------------------
    # Model C — Interaction RF, production weight=3
    # -------------------------------------------------------
    logger.info("Training Model C...")
    rf_c_cpu = RandomForestRegressor(**RF_PARAMS)
    rf_c_mem = RandomForestRegressor(**RF_PARAMS)
    rf_c_cpu.fit(x_train_full, y_train_cpu, sample_weight=sample_weight_train)
    rf_c_mem.fit(x_train_full, y_train_mem, sample_weight=sample_weight_train)
    res_c = _eval_model(rf_c_cpu, rf_c_mem, x_test_full, y_test_cpu, y_test_mem, prod_mask_test)
    logger.info("Model C — R²_cpu=%.4f, MAE_cpu=%.4f", res_c["r2_all_cpu"], res_c["mae_all_cpu"])

    # -------------------------------------------------------
    # Summary table
    # -------------------------------------------------------
    comparison = pd.DataFrame([
        {
            "Model": "A",
            "Notes": "Base RF (18 features)",
            "R²_all": res_a["r2_all_cpu"],
            "MAE_all": res_a["mae_all_cpu"],
            "RMSE_all": res_a["rmse_all_cpu"],
            "R²_prod": res_a["r2_prod_cpu"],
            "MAE_prod": res_a["mae_prod_cpu"],
            "RMSE_prod": res_a["rmse_prod_cpu"],
            "R²_all_mem": res_a["r2_all_mem"],
            "MAE_all_mem": res_a["mae_all_mem"],
            "R²_prod_mem": res_a["r2_prod_mem"],
            "MAE_prod_mem": res_a["mae_prod_mem"],
        },
        {
            "Model": "B",
            "Notes": "+ Interaction features",
            "R²_all": res_b["r2_all_cpu"],
            "MAE_all": res_b["mae_all_cpu"],
            "RMSE_all": res_b["rmse_all_cpu"],
            "R²_prod": res_b["r2_prod_cpu"],
            "MAE_prod": res_b["mae_prod_cpu"],
            "RMSE_prod": res_b["rmse_prod_cpu"],
            "R²_all_mem": res_b["r2_all_mem"],
            "MAE_all_mem": res_b["mae_all_mem"],
            "R²_prod_mem": res_b["r2_prod_mem"],
            "MAE_prod_mem": res_b["mae_prod_mem"],
        },
        {
            "Model": "C",
            "Notes": "+ Weighted (prod=3)",
            "R²_all": res_c["r2_all_cpu"],
            "MAE_all": res_c["mae_all_cpu"],
            "RMSE_all": res_c["rmse_all_cpu"],
            "R²_prod": res_c["r2_prod_cpu"],
            "MAE_prod": res_c["mae_prod_cpu"],
            "RMSE_prod": res_c["rmse_prod_cpu"],
            "R²_all_mem": res_c["r2_all_mem"],
            "MAE_all_mem": res_c["mae_all_mem"],
            "R²_prod_mem": res_c["r2_prod_mem"],
            "MAE_prod_mem": res_c["mae_prod_mem"],
        },
    ])

    out_comparison = RESULTS_DIR / "rq1_model_comparison.csv"
    comparison.to_csv(out_comparison, index=False)
    logger.info("Saved model comparison → %s", out_comparison)

    print("\n=== RQ1 Model Comparison (CPU) ===")
    print(comparison[["Model", "R²_all", "MAE_all", "R²_prod", "MAE_prod", "Notes"]].to_string(index=False))

    # -------------------------------------------------------
    # predictions_per_instance.csv  (use best model = C)
    # -------------------------------------------------------
    logger.info("Building predictions_per_instance.csv using Model C predictions...")
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

    keep_cols = [
        "instance_id", "actual_cpu", "predicted_cpu",
        "actual_mem", "predicted_mem",
        "priority", "is_production",
        "cpu_status", "mem_status",
    ]
    out_pred = RESULTS_DIR / "predictions_per_instance.csv"
    pred_df[keep_cols].to_csv(out_pred, index=False)
    logger.info("Saved predictions → %s", out_pred)

    # Feature importance for Model B (full features, no weights)
    fi_df = pd.DataFrame({
        "feature": BASE_FEATURES + INTERACTION_FEATURES,
        "importance": rf_b_cpu.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    logger.info("Saved feature importance → %s", RESULTS_DIR / "feature_importance.csv")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
