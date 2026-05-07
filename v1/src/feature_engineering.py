"""
Step 2: Feature Engineering
Joins instance_usage with instance_events context and derives lag, statistical,
temporal, and context features.  Output saved to data/processed/.

Usage:
    python src/feature_engineering.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Google Cluster Trace timestamps are in microseconds.
US_PER_SECOND = 1_000_000
# Each usage window is 5 minutes = 300 s.
WINDOW_US = 5 * 60 * US_PER_SECOND

# Timezone for cell A (America/New_York per documentation).
CELL_TIMEZONE = "America/New_York"


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_raw(name: str) -> pd.DataFrame:
    """Load a parquet file from data/raw/."""
    path = RAW_DIR / f"{name}.parquet"
    logger.info("Loading %s", path)
    df = pd.read_parquet(path)
    logger.info("%s shape: %s", name, df.shape)
    return df


# ── Instance-events context ────────────────────────────────────────────────────

def build_events_context(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse instance_events to one representative row per (collection_id,
    instance_index).

    priority, scheduling_class, and resource_request exist only in
    instance_events (not in instance_usage), so they are sourced exclusively
    here and later joined onto the usage DataFrame.

    Strategy: prefer the latest SCHEDULE event (type=1) so the resource_request
    reflects the actual scheduled allocation; fall back to the last event of
    any type.

    Returns:
        DataFrame indexed by (collection_id, instance_index) with context cols.
    """
    logger.info("Building events context from instance_events...")

    # These fields only exist in instance_events — NOT in instance_usage.
    keep_cols = [
        "collection_id",
        "instance_index",
        "time",
        "type",          # event type: SCHEDULE=1
        "priority",
        "scheduling_class",
        "resource_request_cpu",
        "resource_request_memory",
    ]
    avail = [c for c in keep_cols if c in events_df.columns]
    ctx = events_df[avail].copy()

    # Sort by time so "last" means most recent event.
    ctx = ctx.sort_values(["collection_id", "instance_index", "time"])

    # Prefer SCHEDULE events (type == 1) when available, else fall back to any.
    schedule_mask = ctx["type"] == 1 if "type" in ctx.columns else pd.Series(False, index=ctx.index)
    scheduled = ctx[schedule_mask]
    others = ctx[~schedule_mask]

    # Take the last SCHEDULE event per instance, then fill gaps with last-any.
    ctx_sched = (
        scheduled
        .drop_duplicates(subset=["collection_id", "instance_index"], keep="last")
    )
    ctx_any = (
        others
        .drop_duplicates(subset=["collection_id", "instance_index"], keep="last")
    )
    ctx_merged = pd.concat([ctx_sched, ctx_any], ignore_index=True)
    ctx_merged = ctx_merged.drop_duplicates(subset=["collection_id", "instance_index"], keep="first")

    result_cols = ["priority", "scheduling_class", "resource_request_cpu", "resource_request_memory"]
    result_cols = [c for c in result_cols if c in ctx_merged.columns]

    ctx_out = (
        ctx_merged[["collection_id", "instance_index"] + result_cols]
        .set_index(["collection_id", "instance_index"])
    )
    logger.info("Events context shape: %s", ctx_out.shape)
    return ctx_out


# ── Lag & rolling features ─────────────────────────────────────────────────────

def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling window features per (collection_id, instance_index).

    Lag features (previous windows):
        avg_cpu_lag1, avg_cpu_lag2, avg_cpu_lag3
        max_cpu_lag1
        avg_memory_lag1

    Rolling features (last 6 windows = 30 min):
        avg_cpu_roll_mean6, avg_cpu_roll_std6

    Also adds the target: avg_cpu_target (avg_cpu at t+1).

    Args:
        df: DataFrame sorted by (collection_id, instance_index, start_time).

    Returns:
        DataFrame with new feature columns appended.
    """
    logger.info("Adding lag and rolling features...")

    grp = df.groupby(["collection_id", "instance_index"], sort=False)

    # Lag features
    df["avg_cpu_lag1"] = grp["avg_cpu"].shift(1)
    df["avg_cpu_lag2"] = grp["avg_cpu"].shift(2)
    df["avg_cpu_lag3"] = grp["avg_cpu"].shift(3)
    df["max_cpu_lag1"] = grp["max_cpu"].shift(1)
    df["avg_memory_lag1"] = grp["avg_memory"].shift(1)

    # Rolling statistics (min_periods=1 so we keep partial windows)
    df["avg_cpu_roll_mean6"] = (
        grp["avg_cpu"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    )
    df["avg_cpu_roll_std6"] = (
        grp["avg_cpu"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=1).std())
    )

    # Target: avg_cpu at t+1
    df["avg_cpu_target"] = grp["avg_cpu"].shift(-1)

    return df


# ── CPU distribution percentiles ───────────────────────────────────────────────

def extract_cpu_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the 90th and 95th CPU usage percentiles from the distribution columns.

    Per the v3 schema (p.12):
      cpu_usage_distribution      — 11 coarsely-spaced percentiles:
                                    index 0=0th, 1=10th, ..., 9=90th, 10=100th
      tail_cpu_usage_distribution — 9 finely-spaced percentiles:
                                    index 0=91st, 1=92nd, ..., 4=95th, ..., 8=99th

    So:
      p90  → cpu_usage_distribution[9]
      p95  → tail_cpu_usage_distribution[4]

    Args:
        df: DataFrame containing cpu_usage_distribution and optionally
            tail_cpu_usage_distribution columns.

    Returns:
        DataFrame with cpu_p90 and cpu_p95 columns added.
    """
    logger.info("Extracting CPU percentile features...")

    def _safe_index(dist, idx: int) -> float:
        """Return dist[idx] or NaN if the array is absent or too short."""
        if dist is None or not hasattr(dist, "__len__") or len(dist) <= idx:
            return np.nan
        return float(list(dist)[idx])

    # p90: index 9 of cpu_usage_distribution (11-element array)
    df["cpu_p90"] = df["cpu_usage_distribution"].apply(lambda d: _safe_index(d, 9))

    # p95: index 4 of tail_cpu_usage_distribution (9-element array, 91–99th pct)
    if "tail_cpu_usage_distribution" in df.columns:
        df["cpu_p95"] = df["tail_cpu_usage_distribution"].apply(lambda d: _safe_index(d, 4))
    else:
        logger.warning("tail_cpu_usage_distribution not found; cpu_p95 will be NaN.")
        df["cpu_p95"] = np.nan

    return df


# ── Temporal features ──────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert start_time (microseconds) to a timezone-aware datetime and derive
    hour_of_day and day_of_week features.

    Args:
        df: DataFrame with start_time column in microseconds.

    Returns:
        DataFrame with hour_of_day and day_of_week columns added.
    """
    logger.info("Adding temporal features...")

    # The trace epoch is 2019-05-01 00:00:00 UTC (documented baseline).
    # Convert µs offset to absolute UTC timestamp, then localise.
    epoch_utc = pd.Timestamp("2019-05-01", tz="UTC")
    ts_utc = epoch_utc + pd.to_timedelta(df["start_time"], unit="us")
    ts_local = ts_utc.dt.tz_convert(CELL_TIMEZONE)

    df["hour_of_day"] = ts_local.dt.hour
    df["day_of_week"] = ts_local.dt.dayofweek  # Monday=0, Sunday=6
    return df


# ── Context join ───────────────────────────────────────────────────────────────

def join_events_context(usage_df: pd.DataFrame, ctx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join events context onto the usage DataFrame.

    Args:
        usage_df: Instance usage DataFrame.
        ctx_df: Events context indexed by (collection_id, instance_index).

    Returns:
        Merged DataFrame.
    """
    logger.info("Joining events context...")
    merged = usage_df.join(
        ctx_df,
        on=["collection_id", "instance_index"],
        how="left",
        rsuffix="_evt",
    )
    logger.info("After join shape: %s", merged.shape)
    return merged


# ── Final feature set ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Lag features
    "avg_cpu_lag1",
    "avg_cpu_lag2",
    "avg_cpu_lag3",
    "max_cpu_lag1",
    "avg_memory_lag1",
    # Rolling features
    "avg_cpu_roll_mean6",
    "avg_cpu_roll_std6",
    # CPU percentile features
    "cpu_p90",
    "cpu_p95",
    # Hardware/efficiency features
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    # Context features (from events)
    "priority",
    "scheduling_class",
    "collection_type",
    "resource_request_cpu",
    "resource_request_memory",
    # Temporal features
    "hour_of_day",
    "day_of_week",
]

TARGET_COL = "avg_cpu_target"

META_COLS = [
    "start_time",
    "end_time",
    "collection_id",
    "instance_index",
    "machine_id",
    # Keep raw actuals for SLA simulation
    "avg_cpu",
    "avg_memory",
    "max_cpu",
    "assigned_memory",
]


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select feature and meta columns, drop rows with NaN in features or target.

    Args:
        df: Fully-featured DataFrame.

    Returns:
        Clean DataFrame ready for modelling.
    """
    logger.info("Selecting columns and dropping NaN rows...")
    all_cols = META_COLS + FEATURE_COLS + [TARGET_COL]
    # Keep only columns that exist (graceful if some cols absent from data)
    existing = [c for c in all_cols if c in df.columns]
    df = df[existing].copy()

    before = len(df)
    df = df.dropna(subset=[c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns])
    logger.info("Dropped %d rows with NaN (%.1f%%)", before - len(df), 100 * (before - len(df)) / max(before, 1))
    logger.info("Final processed shape: %s", df.shape)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the feature engineering pipeline."""
    # Load raw data
    usage_df = load_raw("instance_usage")
    events_df = load_raw("instance_events")

    # Sort by instance and time (required for lag features)
    logger.info("Sorting by (collection_id, instance_index, start_time)...")
    usage_df = usage_df.sort_values(
        ["collection_id", "instance_index", "start_time"]
    ).reset_index(drop=True)

    # CPU distribution → percentiles
    usage_df = extract_cpu_percentiles(usage_df)

    # Temporal features
    usage_df = add_temporal_features(usage_df)

    # Lag + rolling features + target
    usage_df = add_lag_and_rolling_features(usage_df)

    # Context from events
    ctx_df = build_events_context(events_df)
    usage_df = join_events_context(usage_df, ctx_df)

    # Select, clean, save
    processed = select_and_clean(usage_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "features.parquet"
    processed.to_parquet(out_path, index=False)
    logger.info("Feature engineering complete. Saved → %s", out_path)


if __name__ == "__main__":
    main()
