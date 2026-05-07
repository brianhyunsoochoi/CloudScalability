"""
Pipeline Orchestrator
Runs all five steps in sequence:
    1. data_extraction
    2. feature_engineering
    3. model
    4. sla_simulation
    5. visualization

Usage:
    export GCP_PROJECT_ID="lively-nimbus-492701-g0"
    python src/run_pipeline.py

    # Skip extraction (use existing raw data):
    python src/run_pipeline.py --skip-extraction
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")

# Ensure src/ is on the path so the individual modules are importable.
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))


def _run_step(step_name: str, step_fn) -> None:
    """Execute a pipeline step, logging elapsed time."""
    logger.info("=" * 60)
    logger.info("STEP: %s", step_name)
    logger.info("=" * 60)
    t0 = time.perf_counter()
    step_fn()
    elapsed = time.perf_counter() - t0
    logger.info("DONE: %s (%.1f s)", step_name, elapsed)


def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the full auto-scaling research pipeline.")
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip BigQuery data extraction and use existing files in data/raw/.",
    )
    parser.add_argument(
        "--only",
        choices=["extraction", "features", "model", "simulation", "visualization"],
        default=None,
        help="Run only the specified step.",
    )
    args = parser.parse_args()

    # Import step modules lazily so GCP credentials are only needed if extraction runs.
    steps: list[tuple[str, object]] = []

    if args.only is None:
        if not args.skip_extraction:
            import data_extraction
            steps.append(("1. Data Extraction", data_extraction.main))
        import feature_engineering
        import model
        import sla_simulation
        import visualization
        steps += [
            ("2. Feature Engineering", feature_engineering.main),
            ("3. Model Training", model.main),
            ("4. SLA Simulation", sla_simulation.main),
            ("5. Visualization", visualization.main),
        ]
    else:
        mapping = {
            "extraction": ("1. Data Extraction", lambda: __import__("data_extraction").main()),
            "features": ("2. Feature Engineering", lambda: __import__("feature_engineering").main()),
            "model": ("3. Model Training", lambda: __import__("model").main()),
            "simulation": ("4. SLA Simulation", lambda: __import__("sla_simulation").main()),
            "visualization": ("5. Visualization", lambda: __import__("visualization").main()),
        }
        steps.append(mapping[args.only])

    t_pipeline = time.perf_counter()
    for name, fn in steps:
        _run_step(name, fn)

    total = time.perf_counter() - t_pipeline
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — total time: %.1f s", total)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
