"""
Step 1: Data Extraction
Extracts sampled data from Google BigQuery (Google Cluster Trace v3)
and saves to data/raw/.

Usage:
    export GCP_PROJECT_ID="lively-nimbus-492701-g0"
    python src/data_extraction.py
"""

import logging
import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET = "google.com:google-cluster-data.clusterdata_2019_a"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"

# Google Cluster Trace v3 timestamps are in microseconds.
# 7 days = 7 * 24 * 3600 * 1_000_000 µs
SEVEN_DAYS_US: int = 7 * 24 * 3600 * 1_000_000

SAMPLE_LIMIT = 1_000_000  # max rows for instance_usage


# ── Queries ────────────────────────────────────────────────────────────────────

INSTANCE_USAGE_QUERY = f"""
SELECT
    start_time,
    end_time,
    collection_id,
    instance_index,
    machine_id,
    alloc_collection_id,
    alloc_instance_index,
    collection_type,
    average_usage.cpus          AS avg_cpu,
    average_usage.memory        AS avg_memory,
    maximum_usage.cpus          AS max_cpu,
    maximum_usage.memory        AS max_memory,
    assigned_memory,
    cycles_per_instruction,
    memory_accesses_per_instruction,
    cpu_usage_distribution,
    tail_cpu_usage_distribution,
    sample_rate
FROM
    `{DATASET}.instance_usage`
WHERE
    start_time BETWEEN 0 AND {SEVEN_DAYS_US}
LIMIT {SAMPLE_LIMIT}
"""

# instance_events query is built dynamically after we know the collection_ids.
INSTANCE_EVENTS_QUERY_TEMPLATE = """
SELECT
    time,
    type,
    collection_id,
    instance_index,
    machine_id,
    priority,
    scheduling_class,
    collection_type,
    resource_request.cpus   AS resource_request_cpu,
    resource_request.memory AS resource_request_memory,
    alloc_collection_id
FROM
    `{dataset}.instance_events`
WHERE
    collection_id IN UNNEST(@collection_ids)
"""

MACHINE_EVENTS_QUERY = f"""
SELECT
    time,
    machine_id,
    type,
    capacity.cpus   AS capacity_cpu,
    capacity.memory AS capacity_memory
FROM
    `{DATASET}.machine_events`
WHERE
    type IN (0, 1)   -- 0=ADD, 1=UPDATE (capacity info)
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_project_id() -> str:
    """Return GCP project ID from environment variable."""
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise EnvironmentError(
            "GCP_PROJECT_ID environment variable is not set. "
            "Run: export GCP_PROJECT_ID='lively-nimbus-492701-g0'"
        )
    return project_id


def build_client(project_id: str) -> bigquery.Client:
    """Create and return a BigQuery client."""
    logger.info("Building BigQuery client for project: %s", project_id)
    return bigquery.Client(project=project_id)


def run_query(client: bigquery.Client, query: str, job_config: bigquery.QueryJobConfig | None = None) -> pd.DataFrame:
    """Execute a BigQuery query and return results as a DataFrame."""
    job = client.query(query, job_config=job_config)
    logger.info("Query job started: %s", job.job_id)
    df = job.to_dataframe(progress_bar_type=None)
    logger.info("Query returned %d rows", len(df))
    return df


# ── Extraction functions ───────────────────────────────────────────────────────

def extract_instance_usage(client: bigquery.Client) -> pd.DataFrame:
    """
    Extract instance_usage rows for cell A, first 7 days, up to SAMPLE_LIMIT rows.

    Returns:
        DataFrame with CPU/memory usage per instance per time window.
    """
    logger.info("Extracting instance_usage (limit=%d, 7-day filter)...", SAMPLE_LIMIT)
    df = run_query(client, INSTANCE_USAGE_QUERY)
    logger.info("instance_usage shape: %s", df.shape)
    return df


def extract_instance_events(client: bigquery.Client, collection_ids: list[int]) -> pd.DataFrame:
    """
    Extract instance_events for the collection_ids found in instance_usage.

    Args:
        client: BigQuery client.
        collection_ids: List of collection_id values to filter on.

    Returns:
        DataFrame with event metadata per instance.
    """
    logger.info(
        "Extracting instance_events for %d unique collection_ids...",
        len(collection_ids),
    )

    # BigQuery parameterised query with ARRAY parameter to avoid giant IN-lists.
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("collection_ids", "INT64", collection_ids),
        ]
    )
    query = INSTANCE_EVENTS_QUERY_TEMPLATE.format(dataset=DATASET)
    df = run_query(client, query, job_config=job_config)
    logger.info("instance_events shape: %s", df.shape)
    return df


def extract_machine_events(client: bigquery.Client) -> pd.DataFrame:
    """
    Extract machine_events (ADD and UPDATE types) for capacity information.

    Returns:
        DataFrame with machine capacity records.
    """
    logger.info("Extracting machine_events (ADD/UPDATE only)...")
    df = run_query(client, MACHINE_EVENTS_QUERY)
    logger.info("machine_events shape: %s", df.shape)
    return df


# ── Persistence ────────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, name: str) -> Path:
    """
    Save a DataFrame to data/raw/<name>.parquet.

    Args:
        df: DataFrame to persist.
        name: Base filename (without extension).

    Returns:
        Path to the saved file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %s → %s (%d rows)", name, path, len(df))
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full data extraction pipeline."""
    project_id = get_project_id()
    client = build_client(project_id)

    # 1. Instance usage
    usage_df = extract_instance_usage(client)
    save_parquet(usage_df, "instance_usage")

    # 2. Instance events — filtered to matching collection_ids
    collection_ids = usage_df["collection_id"].dropna().unique().tolist()
    collection_ids = [int(c) for c in collection_ids]
    events_df = extract_instance_events(client, collection_ids)
    save_parquet(events_df, "instance_events")

    # 3. Machine events
    machine_df = extract_machine_events(client)
    save_parquet(machine_df, "machine_events")

    logger.info("Data extraction complete. Files saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
