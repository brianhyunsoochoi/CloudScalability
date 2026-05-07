"""
Step 3: Model Training
Trains Random Forest regressors with varying hyperparameters and feature subsets
to produce a range of R² values for RQ2 analysis.

Saves per-model metrics and test-set predictions to results/.

Usage:
    python src/model.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"

FEATURE_COLS = [
    "avg_cpu_lag1",
    "avg_cpu_lag2",
    "avg_cpu_lag3",
    "max_cpu_lag1",
    "avg_memory_lag1",
    "avg_cpu_roll_mean6",
    "avg_cpu_roll_std6",
    "cpu_p90",
    "cpu_p95",
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    "priority",
    "scheduling_class",
    "collection_type",
    "resource_request_cpu",
    "resource_request_memory",
    "hour_of_day",
    "day_of_week",
]

TARGET_COL = "avg_cpu_target"

# ── Model variant definitions ──────────────────────────────────────────────────

LAG_FEATURES = [
    "avg_cpu_lag1",
    "avg_cpu_lag2",
    "avg_cpu_lag3",
    "max_cpu_lag1",
    "avg_memory_lag1",
]

STAT_FEATURES = [
    "avg_cpu_roll_mean6",
    "avg_cpu_roll_std6",
    "cpu_p90",
    "cpu_p95",
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
]

CONTEXT_FEATURES = [
    "priority",
    "scheduling_class",
    "collection_type",
    "resource_request_cpu",
    "resource_request_memory",
    "hour_of_day",
    "day_of_week",
]


def _make_rf(**kwargs: object) -> RandomForestRegressor:
    """Create a RandomForestRegressor with sensible defaults."""
    defaults = dict(n_jobs=-1, random_state=42)
    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)  # type: ignore[arg-type]


# Each variant: (name, feature_list, rf_kwargs, use_grid_search)
MODEL_VARIANTS: list[tuple[str, list[str], dict, bool]] = [
    # Full model — best hyperparams via GridSearchCV
    (
        "full_model",
        FEATURE_COLS,
        {},
        True,  # grid search
    ),
    # Feature-subset: lag only
    (
        "lag_only",
        LAG_FEATURES,
        {"n_estimators": 100, "max_depth": 20},
        False,
    ),
    # Feature-subset: stats only
    (
        "stats_only",
        STAT_FEATURES,
        {"n_estimators": 100, "max_depth": 20},
        False,
    ),
    # Deliberately weak: shallow trees (depth 3)
    (
        "shallow_d3",
        FEATURE_COLS,
        {"n_estimators": 100, "max_depth": 3},
        False,
    ),
    # Deliberately weak: shallow trees (depth 5)
    (
        "shallow_d5",
        FEATURE_COLS,
        {"n_estimators": 100, "max_depth": 5},
        False,
    ),
    # Deliberately weak: tiny forest
    (
        "small_forest",
        FEATURE_COLS,
        {"n_estimators": 10, "max_depth": 20},
        False,
    ),
    # Minimal: single lag feature
    (
        "minimal_lag1",
        ["avg_cpu_lag1"],
        {"n_estimators": 50, "max_depth": 10},
        False,
    ),
]

# GridSearchCV parameter grid for the full model.
PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
}


# ── Data loading & splitting ───────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Load processed feature parquet."""
    path = PROCESSED_DIR / "features.parquet"
    logger.info("Loading features from %s", path)
    df = pd.read_parquet(path)
    logger.info("Features shape: %s", df.shape)
    return df


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically: first 80% → train, last 20% → test.

    Args:
        df: DataFrame sorted by start_time.

    Returns:
        (train_df, test_df)
    """
    df = df.sort_values("start_time").reset_index(drop=True)
    cutoff = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff].copy()
    test_df = df.iloc[cutoff:].copy()
    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))
    return train_df, test_df


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute R², MAE, RMSE."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train_variant(
    name: str,
    features: list[str],
    rf_kwargs: dict,
    use_grid_search: bool,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    Train one model variant and return metrics + predictions.

    Args:
        name: Variant identifier.
        features: Feature column names to use.
        rf_kwargs: Extra kwargs for RandomForestRegressor.
        use_grid_search: If True, run GridSearchCV; otherwise train directly.
        train_df: Training split.
        test_df: Test split.

    Returns:
        Dict with keys: name, metrics, feature_importances, best_params.
    """
    # Keep only features that exist in the data
    avail = [f for f in features if f in train_df.columns]
    if len(avail) < len(features):
        missing = set(features) - set(avail)
        logger.warning("[%s] Missing features (skipped): %s", name, missing)

    X_train = train_df[avail].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[avail].values
    y_test = test_df[TARGET_COL].values

    logger.info("[%s] Training on %d features, %d train samples...", name, len(avail), len(X_train))

    if use_grid_search:
        tscv = TimeSeriesSplit(n_splits=3)
        base_rf = _make_rf()
        gs = GridSearchCV(
            base_rf,
            PARAM_GRID,
            cv=tscv,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
        logger.info("[%s] Best params: %s", name, best_params)
    else:
        model = _make_rf(**rf_kwargs)
        model.fit(X_train, y_train)
        best_params = rf_kwargs

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    logger.info("[%s] R²=%.4f  MAE=%.4f  RMSE=%.4f", name, metrics["r2"], metrics["mae"], metrics["rmse"])

    importance = dict(zip(avail, model.feature_importances_.tolist()))

    return {
        "name": name,
        "metrics": metrics,
        "best_params": best_params,
        "feature_importances": importance,
        "features_used": avail,
        # Store predictions alongside ground truth for SLA simulation
        "predictions": pd.DataFrame(
            {
                "start_time": test_df["start_time"].values,
                "collection_id": test_df["collection_id"].values,
                "instance_index": test_df["instance_index"].values,
                "machine_id": test_df["machine_id"].values,
                "actual_cpu": y_test,
                "predicted_cpu": y_pred,
                "resource_request_cpu": test_df["resource_request_cpu"].values
                if "resource_request_cpu" in test_df.columns
                else np.nan,
            }
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Train all model variants and persist results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    train_df, test_df = time_based_split(df)

    all_metrics: list[dict] = []

    for name, features, rf_kwargs, use_gs in MODEL_VARIANTS:
        result = train_variant(name, features, rf_kwargs, use_gs, train_df, test_df)

        # Save predictions
        pred_path = RESULTS_DIR / f"predictions_{name}.parquet"
        result["predictions"].to_parquet(pred_path, index=False)
        logger.info("Saved predictions → %s", pred_path)

        all_metrics.append(
            {
                "model": result["name"],
                **result["metrics"],
                "best_params": json.dumps(result["best_params"]),
                "features_used": result["features_used"],
                "feature_importances": result["feature_importances"],
            }
        )

    # Persist summary metrics
    metrics_df = pd.DataFrame(
        [
            {
                "model": m["model"],
                "r2": m["r2"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "best_params": m["best_params"],
            }
            for m in all_metrics
        ]
    )
    metrics_path = RESULTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Model metrics saved → %s", metrics_path)

    # Persist feature importances (full model only for main reference)
    fi_records = [
        {
            "model": m["model"],
            "feature": feat,
            "importance": imp,
        }
        for m in all_metrics
        for feat, imp in m.get("feature_importances", {}).items()
    ]
    fi_df = pd.DataFrame(fi_records)
    fi_path = RESULTS_DIR / "feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)
    logger.info("Feature importances saved → %s", fi_path)

    logger.info("Model training complete.")
    print(metrics_df[["model", "r2", "mae", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
