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
import time
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

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

# Temp dataset in the user's own project for the pairs upload
TEMP_DATASET = "tmp_pipeline"
TEMP_TABLE   = "usage_instances"


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

# instance_events query template.
# {temp_table} is filled at runtime with the fully-qualified temp table ID,
# e.g. `project.tmp_pipeline.usage_instances`.
#
# Strategy:
#   1. INNER JOIN instance_events with the temp table that holds the EXACT
#      (collection_id, instance_index) pairs from the downloaded usage data.
#   2. Prefer SCHEDULE events (type=1); fall back to any event type.
#      ROW_NUMBER orders by: SCHEDULE first, then most recent time.
#   3. Download only 1 row per instance — small result set.
INSTANCE_EVENTS_QUERY_TEMPLATE = """
WITH ranked_events AS (
    SELECT
        ie.collection_id,
        ie.instance_index,
        ie.scheduling_class,
        ie.priority,
        ie.resource_request.cpus   AS resource_request_cpu,
        ie.resource_request.memory AS resource_request_memory,
        ROW_NUMBER() OVER (
            PARTITION BY ie.collection_id, ie.instance_index
            ORDER BY
                CASE WHEN ie.type = 1 THEN 0 ELSE 1 END,  -- SCHEDULE first
                ie.time DESC                                -- then most recent
        ) AS rn
    FROM
        `{dataset}.instance_events` AS ie
    INNER JOIN
        `{temp_table}` AS ui
        ON  ie.collection_id  = ui.collection_id
        AND ie.instance_index = ui.instance_index
)
SELECT
    collection_id,
    instance_index,
    scheduling_class,
    priority,
    resource_request_cpu,
    resource_request_memory
FROM
    ranked_events
WHERE
    rn = 1
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


def run_query(
    client: bigquery.Client,
    query: str,
    job_config: bigquery.QueryJobConfig | None = None,
) -> pd.DataFrame:
    """Execute a BigQuery query and return results as a DataFrame."""
    job = client.query(query, job_config=job_config)
    logger.info("Query job started: %s", job.job_id)
    df = job.to_dataframe(progress_bar_type=None)
    logger.info("Query returned %d rows", len(df))
    return df


# ── Temp table helpers ─────────────────────────────────────────────────────────

def ensure_temp_dataset(client: bigquery.Client, project_id: str) -> None:
    """Create the temp dataset in the user's project if it does not exist."""
    dataset_ref = bigquery.DatasetReference(project_id, TEMP_DATASET)
    try:
        client.get_dataset(dataset_ref)
        logger.info("Temp dataset '%s' already exists.", TEMP_DATASET)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        logger.info("Created temp dataset '%s'.", TEMP_DATASET)


def upload_pairs_to_bq(
    client: bigquery.Client,
    usage_df: pd.DataFrame,
    project_id: str,
) -> str:
    """
    Upload the distinct (collection_id, instance_index) pairs from the
    downloaded usage DataFrame to a BigQuery temp table in the user's project.

    This ensures the events query joins against the EXACT same instance set
    that was downloaded — not a different random LIMIT sample.

    Args:
        client: BigQuery client.
        usage_df: Downloaded instance_usage DataFrame.
        project_id: GCP project ID for the temp table.

    Returns:
        Fully-qualified temp table ID, e.g.
        "project.tmp_pipeline.usage_instances".
    """
    ensure_temp_dataset(client, project_id)

    pairs = (
        usage_df[["collection_id", "instance_index"]]
        .drop_duplicates()
        .astype({"collection_id": "Int64", "instance_index": "Int64"})
        .reset_index(drop=True)
    )
    logger.info(
        "Uploading %d distinct (collection_id, instance_index) pairs to BQ temp table...",
        len(pairs),
    )

    temp_table_id = f"{project_id}.{TEMP_DATASET}.{TEMP_TABLE}"

    schema = [
        bigquery.SchemaField("collection_id",  "INT64"),
        bigquery.SchemaField("instance_index", "INT64"),
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(pairs, temp_table_id, job_config=job_config)
    job.result()  # wait for completion
    logger.info("Temp table ready: %s (%d rows)", temp_table_id, len(pairs))
    return temp_table_id


def delete_temp_table(client: bigquery.Client, temp_table_id: str) -> None:
    """Delete the temporary BigQuery table after the events query completes."""
    try:
        client.delete_table(temp_table_id)
        logger.info("Deleted temp table: %s", temp_table_id)
    except NotFound:
        logger.warning("Temp table not found (already deleted?): %s", temp_table_id)


# ── Extraction functions ───────────────────────────────────────────────────────

def extract_instance_usage(client: bigquery.Client) -> pd.DataFrame:
    """
    Extract instance_usage rows for cell A, first 7 days, up to SAMPLE_LIMIT.

    Returns:
        DataFrame with CPU/memory usage per instance per time window.
    """
    logger.info("Extracting instance_usage (limit=%d, 7-day filter)...", SAMPLE_LIMIT)
    df = run_query(client, INSTANCE_USAGE_QUERY)
    logger.info("instance_usage shape: %s", df.shape)
    return df


def extract_instance_events(
    client: bigquery.Client,
    temp_table_id: str,
) -> pd.DataFrame:
    """
    Extract the best available event per (collection_id, instance_index) for
    the EXACT instances present in the downloaded usage data.

    Uses a temp table (uploaded from the usage parquet) so the JOIN key set
    matches perfectly — avoiding the sampling mismatch that caused 0.5% hit
    rate when both sides used independent LIMIT clauses.

    Preference order within each instance:
        1. Most recent SCHEDULE event (type=1)  — actual allocation metadata
        2. Most recent event of any other type  — fallback for instances that
           were never scheduled (batch jobs, etc.)

    Args:
        client: BigQuery client.
        temp_table_id: Fully-qualified BQ table ID for the pairs upload.

    Returns:
        DataFrame with one row per instance: scheduling_class, priority,
        resource_request_cpu, resource_request_memory.
    """
    logger.info(
        "Extracting instance_events (temp table JOIN: %s)...", temp_table_id
    )
    query = INSTANCE_EVENTS_QUERY_TEMPLATE.format(
        dataset=DATASET,
        temp_table=temp_table_id,
    )
    df = run_query(client, query)
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

    # 1. Instance usage — download and save immediately
    usage_df = extract_instance_usage(client)
    save_parquet(usage_df, "instance_usage")

    # 2. Upload exact pairs to BQ temp table, then query events against them
    temp_table_id: str | None = None
    try:
        temp_table_id = upload_pairs_to_bq(client, usage_df, project_id)
        events_df = extract_instance_events(client, temp_table_id)
        save_parquet(events_df, "instance_events")
    finally:
        # Always clean up the temp table, even if the query failed
        if temp_table_id:
            delete_temp_table(client, temp_table_id)

    # 3. Machine events
    machine_df = extract_machine_events(client)
    save_parquet(machine_df, "machine_events")

    logger.info("Data extraction complete. Files saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
