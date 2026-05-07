"""
feature_engineering.py — Build base and interaction features for RQ1.

Feature set:
  Base (18): lag, rolling, percentile, hardware, context, temporal, tier
  Interaction (3): prod_x_hour, prod_x_dow, sc_x_std
  Targets: avg_cpu_target, avg_mem_target (t+1 values)
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_FEATURES = [
    "avg_cpu_lag1", "avg_cpu_lag2", "avg_cpu_lag3",
    "avg_mem_lag1",
    "max_cpu_lag1",
    "avg_cpu_roll_mean6", "avg_cpu_roll_std6",
    "cpu_p90", "cpu_p95",
    "cycles_per_instruction", "memory_accesses_per_instruction",
    "priority", "scheduling_class",
    "resource_request_cpu", "resource_request_memory",
    "hour_of_day", "day_of_week",
    "is_production",
]

INTERACTION_FEATURES = [
    "prod_x_hour",
    "prod_x_dow",
    "sc_x_std",
]

TARGETS = ["avg_cpu_target", "avg_mem_target"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features and targets from a parsed DataFrame.

    Lags and rolling stats are computed per-instance to avoid
    bleeding across different workloads.

    Args:
        df: Output of load_data.load_raw(), must contain:
            instance_id, avg_cpu, avg_mem, max_cpu,
            resource_request_cpu, resource_request_memory,
            cpu_p90, cpu_p95, priority, scheduling_class,
            cycles_per_instruction, memory_accesses_per_instruction,
            start_time, is_production

    Returns:
        DataFrame with base features, interaction features, and targets.
        Rows with NaN from lag creation are dropped.
    """
    logger.info("Building features for %d rows...", len(df))

    # Sort by instance then time for correct lag computation
    df = df.sort_values(["instance_id", "start_time"]).reset_index(drop=True)

    grp = df.groupby("instance_id", sort=False)

    # --- Lag features ---
    logger.info("Computing lag features...")
    df["avg_cpu_lag1"] = grp["avg_cpu"].shift(1)
    df["avg_cpu_lag2"] = grp["avg_cpu"].shift(2)
    df["avg_cpu_lag3"] = grp["avg_cpu"].shift(3)
    df["avg_mem_lag1"] = grp["avg_mem"].shift(1)
    df["max_cpu_lag1"] = grp["max_cpu"].shift(1)

    # --- Rolling features (6-window) ---
    logger.info("Computing rolling features...")
    df["avg_cpu_roll_mean6"] = (
        grp["avg_cpu"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    )
    df["avg_cpu_roll_std6"] = (
        grp["avg_cpu"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=2).std().fillna(0.0))
    )

    # --- Target: t+1 values ---
    logger.info("Computing targets (t+1)...")
    df["avg_cpu_target"] = grp["avg_cpu"].shift(-1)
    df["avg_mem_target"] = grp["avg_mem"].shift(-1)

    # --- Temporal features ---
    if pd.api.types.is_datetime64_any_dtype(df["start_time"]):
        df["hour_of_day"] = df["start_time"].dt.hour
        df["day_of_week"] = df["start_time"].dt.dayofweek
    else:
        logger.warning("start_time is not datetime; hour_of_day/day_of_week set to 0")
        df["hour_of_day"] = 0
        df["day_of_week"] = 0

    # --- Ensure required columns exist with fallback ---
    for col in ["cycles_per_instruction", "memory_accesses_per_instruction",
                "scheduling_class"]:
        if col not in df.columns:
            logger.warning("Column '%s' missing, filling with 0", col)
            df[col] = 0

    # --- Interaction features ---
    df["prod_x_hour"] = df["is_production"] * df["hour_of_day"]
    df["prod_x_dow"] = df["is_production"] * df["day_of_week"]
    df["sc_x_std"] = df["scheduling_class"] * df["avg_cpu_roll_std6"]

    # --- Drop rows missing targets or lag1 (edges of each instance's window) ---
    required = ["avg_cpu_lag1", "avg_cpu_target", "avg_mem_target"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    logger.info("Dropped %d rows with NaN targets/lags; %d rows remain", before - len(df), len(df))

    all_features = BASE_FEATURES + INTERACTION_FEATURES
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        logger.warning("Features missing from DataFrame: %s", missing)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def get_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_base, X_full) feature matrices.

    X_base: 18 base features
    X_full: 21 features (base + interaction)
    """
    # Fill remaining NaN in features with column median
    feat_df = df.copy()
    all_features = BASE_FEATURES + INTERACTION_FEATURES
    for col in all_features:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].fillna(feat_df[col].median())

    x_base = feat_df[BASE_FEATURES]
    x_full = feat_df[BASE_FEATURES + INTERACTION_FEATURES]
    return x_base, x_full


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from load_data import load_raw

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df_raw = load_raw()
    df_feat = build_features(df_raw)
    print(df_feat[BASE_FEATURES + TARGETS].head())
    print("Base features:", BASE_FEATURES)
    print("Interaction features:", INTERACTION_FEATURES)
