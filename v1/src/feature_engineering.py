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
    Build a context lookup indexed by (collection_id, instance_index).

    The BigQuery extraction query already returns exactly one row per instance
    (latest SCHEDULE event, selected via ROW_NUMBER), so no further sorting or
    deduplication is needed here.

    Columns present: collection_id, instance_index, scheduling_class,
                     priority, resource_request_cpu, resource_request_memory.

    Returns:
        DataFrame indexed by (collection_id, instance_index).
    """
    logger.info("Building events context from instance_events...")

    ctx = events_df.set_index(["collection_id", "instance_index"])
    logger.info("Events context shape: %s", ctx.shape)
    return ctx


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

    df.join(on=...) 은 인덱스 dtype이 다르면 매칭 실패한다.
    pd.merge는 내부적으로 dtype을 통일하므로 더 안전하다.

    Args:
        usage_df: Instance usage DataFrame.
        ctx_df: Events context indexed by (collection_id, instance_index).

    Returns:
        Merged DataFrame.
    """
    logger.info("Joining events context...")

    # ctx_df 인덱스를 일반 컬럼으로 풀어서 merge
    ctx_reset = ctx_df.reset_index()   # collection_id, instance_index가 컬럼이 됨

    merged = usage_df.merge(
        ctx_reset,
        on=["collection_id", "instance_index"],
        how="left",
        suffixes=("", "_evt"),
    )

    matched = merged["priority"].notna().sum() if "priority" in merged.columns else 0
    logger.info(
        "After join shape: %s  (events matched: %d / %d rows, %.1f%%)",
        merged.shape, matched, len(merged), 100 * matched / max(len(merged), 1),
    )
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


# 필수 컬럼: NaN이면 row 삭제 (lag 피처 + target)
MANDATORY_COLS = [
    "avg_cpu_lag1",
    "avg_cpu_lag2",
    "avg_cpu_lag3",
    "max_cpu_lag1",
    "avg_memory_lag1",
    "avg_cpu_roll_mean6",
    TARGET_COL,
]

# 선택 컬럼: 원본이 sparse하거나 join miss 가능 → NaN이면 0으로 대체
FILLNA_ZERO_COLS = [
    "avg_cpu_roll_std6",        # 첫 윈도우 std는 0이 자연스러움
    "cpu_p90",                  # 분포 배열 없는 경우
    "cpu_p95",                  # tail_cpu_usage_distribution 없는 경우
    "cycles_per_instruction",   # trace에서 sparse (약 19% NaN)
    "memory_accesses_per_instruction",
    "priority",                 # events join miss 시
    "scheduling_class",
    "collection_type",
    "resource_request_cpu",
    "resource_request_memory",
]


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select feature and meta columns, then apply a two-tier NaN strategy:

    1. **필수 컬럼** (MANDATORY_COLS): NaN이면 row 삭제.
       lag/rolling 피처와 target이 없으면 학습에 쓸 수 없음.

    2. **선택 컬럼** (FILLNA_ZERO_COLS): NaN이면 0으로 대체.
       원본 trace에서 sparse하거나 events join이 실패한 row도 유지.

    Args:
        df: Fully-featured DataFrame.

    Returns:
        Clean DataFrame ready for modelling.
    """
    logger.info("Selecting columns and applying NaN strategy...")
    all_cols = META_COLS + FEATURE_COLS + [TARGET_COL]
    existing = [c for c in all_cols if c in df.columns]
    df = df[existing].copy()

    # Step 1: 선택 컬럼 fillna(0)
    fill_cols = [c for c in FILLNA_ZERO_COLS if c in df.columns]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(0)
        logger.info("Filled NaN with 0 for %d optional columns: %s", len(fill_cols), fill_cols)

    # Step 2: 필수 컬럼 dropna
    mandatory = [c for c in MANDATORY_COLS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=mandatory)
    dropped = before - len(df)
    logger.info(
        "Dropped %d rows with NaN in mandatory columns (%.1f%% of total)",
        dropped, 100 * dropped / max(before, 1),
    )
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
