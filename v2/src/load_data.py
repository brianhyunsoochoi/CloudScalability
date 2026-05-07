"""
load_data.py — Load and parse borg_traces_data CSV for v2 RQ1 pipeline.

Handles dict-string columns: average_usage, resource_request,
cpu_usage_distribution, tail_cpu_usage_distribution, maximum_usage.
"""
import ast
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = Path(r"C:\Users\hynbb\Desktop\borg_traces_data - Copy.csv")


def _parse_dict_col(series: pd.Series, key: str) -> pd.Series:
    """Parse a column containing dict-like strings and extract a single key.

    Handles formats like:
      {'cpus': 0.5, 'memory': 0.1}
      {cpus: 0.5, memory: 0.1}   (unquoted keys)
    """
    def _extract(val):
        if pd.isna(val) or val == "" or val == "{}":
            return np.nan
        try:
            d = ast.literal_eval(str(val))
            return float(d.get(key, np.nan))
        except Exception:
            # Fallback: regex
            m = re.search(rf"['\"]?{re.escape(key)}['\"]?\s*:\s*([\d.eE+\-]+)", str(val))
            return float(m.group(1)) if m else np.nan

    return series.apply(_extract)


def _parse_distribution(series: pd.Series, index: int) -> pd.Series:
    """Extract value at a specific index from a list-like string column."""
    def _extract(val):
        if pd.isna(val) or val in ("", "[]"):
            return np.nan
        try:
            lst = ast.literal_eval(str(val))
            if isinstance(lst, (list, tuple)) and len(lst) > index:
                return float(lst[index])
        except Exception:
            pass
        return np.nan

    return series.apply(_extract)


def load_raw(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load CSV and parse complex columns into usable numeric fields.

    Returns a DataFrame with added columns:
      avg_cpu, avg_mem, max_cpu,
      resource_request_cpu, resource_request_memory,
      cpu_p90, cpu_p95
    """
    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    logger.info("Columns: %s", df.columns.tolist())

    # --- average_usage ---
    if "average_usage" in df.columns:
        logger.info("Parsing average_usage...")
        df["avg_cpu"] = _parse_dict_col(df["average_usage"], "cpus")
        df["avg_mem"] = _parse_dict_col(df["average_usage"], "memory")
    else:
        logger.warning("Column 'average_usage' not found; setting avg_cpu/avg_mem to NaN")
        df["avg_cpu"] = np.nan
        df["avg_mem"] = np.nan

    # --- maximum_usage ---
    if "maximum_usage" in df.columns:
        logger.info("Parsing maximum_usage...")
        df["max_cpu"] = _parse_dict_col(df["maximum_usage"], "cpus")
    else:
        logger.warning("Column 'maximum_usage' not found; max_cpu will fallback to avg_cpu")
        df["max_cpu"] = df["avg_cpu"]

    # --- resource_request ---
    if "resource_request" in df.columns:
        logger.info("Parsing resource_request...")
        df["resource_request_cpu"] = _parse_dict_col(df["resource_request"], "cpus")
        df["resource_request_memory"] = _parse_dict_col(df["resource_request"], "memory")
    else:
        logger.warning("Column 'resource_request' not found")
        df["resource_request_cpu"] = np.nan
        df["resource_request_memory"] = np.nan

    # --- cpu_usage_distribution (p90 = index 9) ---
    if "cpu_usage_distribution" in df.columns:
        logger.info("Parsing cpu_usage_distribution...")
        df["cpu_p90"] = _parse_distribution(df["cpu_usage_distribution"], index=9)
    else:
        logger.warning("Column 'cpu_usage_distribution' not found; cpu_p90 set to NaN")
        df["cpu_p90"] = np.nan

    # --- tail_cpu_usage_distribution (p95 = index 0, i.e. first tail bucket) ---
    if "tail_cpu_usage_distribution" in df.columns:
        logger.info("Parsing tail_cpu_usage_distribution...")
        df["cpu_p95"] = _parse_distribution(df["tail_cpu_usage_distribution"], index=0)
    else:
        logger.warning("Column 'tail_cpu_usage_distribution' not found; cpu_p95 set to NaN")
        df["cpu_p95"] = np.nan

    # --- start_time: convert to datetime if numeric (microseconds) ---
    if "start_time" in df.columns:
        if pd.api.types.is_numeric_dtype(df["start_time"]):
            logger.info("Converting numeric start_time (microseconds) to datetime...")
            df["start_time"] = pd.to_datetime(df["start_time"], unit="us", utc=True)
        else:
            df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")

    # --- instance_id ---
    if "collection_id" in df.columns and "instance_index" in df.columns:
        df["instance_id"] = (
            df["collection_id"].astype(str) + "_" + df["instance_index"].astype(str)
        )
    elif "instance_id" not in df.columns:
        logger.warning("No instance_id derivable; using row index as fallback")
        df["instance_id"] = df.index.astype(str)

    # --- is_production ---
    if "priority" in df.columns:
        df["is_production"] = df["priority"].between(120, 359).astype(int)
    else:
        logger.warning("Column 'priority' not found")
        df["is_production"] = 0

    logger.info("Parsing complete. Final shape: %s", df.shape)
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df[["instance_id", "avg_cpu", "avg_mem", "max_cpu",
              "resource_request_cpu", "cpu_p90", "cpu_p95",
              "is_production"]].head(10))
    print(df.dtypes)
