# Project: Predictive Auto-Scaling — R² vs SLA Violation Correlation Study

## Overview
This research project investigates whether higher prediction accuracy (R², MAE) in CPU usage forecasting translates to better operational SLA outcomes (lower SLA violation rates). We use the Google Cluster Trace v3 dataset and Random Forest regression.

## Research Questions
- **RQ2**: Is there a statistically significant correlation between prediction accuracy (R², MAE) and SLA violation rate?
- **RQ3**: What is the trade-off between SLA violation reduction and over-provisioning cost in predictive scaling?

## Dataset
- **Source**: Google Cluster Trace v3 (BigQuery: `google.com:google-cluster-data`)
- **Cell**: `clusterdata_2019_a` (use cell A only)
- **Key tables**: `instance_usage`, `instance_events`, `machine_events`
- **Documentation**: `docs/Google_cluster-usage_traces_v3.pdf`

## Data Access
BigQuery project ID is set via environment variable:
```bash
export GCP_PROJECT_ID="your-project-id-here"
```

## Pipeline

### 1. Data Extraction (`src/data_extraction.py`)
Extract sampled data from BigQuery and save to `data/raw/`.

**instance_usage query requirements**:
- Filter: cell A, first 7 days of trace (keep costs low)
- Sample: ~1M rows max
- Fields needed: start_time, end_time, collection_id, instance_index, machine_id, collection_type, average_usage (cpus, memory), maximum_usage (cpus, memory), resource_request (cpus, memory), assigned_memory, cycles_per_instruction, memory_accesses_per_instruction, cpu_usage_distribution, sample_rate

**instance_events query requirements**:
- Filter: matching collection_ids from instance_usage
- Fields needed: time, type, collection_id, instance_index, machine_id, priority, scheduling_class, collection_type, resource_request, alloc_collection_id

**machine_events query requirements**:
- Fields needed: time, machine_id, type, capacity (cpus, memory)
- Only ADD/UPDATE events (for capacity info)

### 2. Feature Engineering (`src/feature_engineering.py`)
Join instance_usage with instance_events context. Per instance, per time window:

**Lag features**:
- avg_cpu at t-1, t-2, t-3 (previous 3 windows)
- max_cpu at t-1
- avg_memory at t-1

**Statistical features**:
- Rolling mean and std of avg_cpu over last 6 windows (30 min)
- 90th and 95th percentile from cpu_usage_distribution
- cycles_per_instruction, memory_accesses_per_instruction

**Context features** (from instance_events join):
- priority (integer)
- scheduling_class (0-3)
- collection_type (0=job, 1=alloc set)
- resource_request_cpu, resource_request_memory

**Temporal features**:
- hour_of_day (from timestamp, timezone America/New_York for cell A)
- day_of_week

**Target variable**: avg_cpu at t+1 (next window's average CPU usage)

Drop rows with NaN from lag creation. Save to `data/processed/`.

### 3. Model Training (`src/model.py`)
**Model**: scikit-learn RandomForestRegressor

**Train/Test split**: Time-based (first 80% chronological = train, last 20% = test). NO random split.

**Hyperparameter tuning**: GridSearchCV with TimeSeriesSplit(n_splits=3)
- n_estimators: [100, 300, 500]
- max_depth: [10, 20, 30, None]
- min_samples_split: [2, 5, 10]

**Evaluation metrics**: R², MAE, RMSE

**Model variants for RQ2**: Create 6+ models with deliberately varied R² levels:
- Full model (all features, best hyperparams)
- Reduced features (lag only)
- Reduced features (stats only)
- Shallow trees (max_depth=3)
- Shallow trees (max_depth=5)
- Small forest (n_estimators=10)
- Minimal model (1 lag feature only)

Save each model's R², MAE, RMSE and predictions to `results/`.

### 4. SLA Simulation (`src/sla_simulation.py`)

**SLA violation definition (proxy)**:
- Primary: `actual_cpu > provisioned_capacity` for a given window
- Where `provisioned_capacity` = scaling decision based on predicted CPU

**Scaling simulation logic**:
For each threshold τ in [0.5, 0.6, 0.7, 0.8, 0.9]:
- If predicted_cpu > τ * current_capacity → scale out (provision = predicted * safety_margin)
- Else → keep current provisioning (provision = resource_request)
- violation = 1 if actual_cpu > provisioned, else 0

**Calculate for each model variant × each threshold**:
```
sla_violation_rate = count(violation==1) / count(total_windows)
over_provision_cost = sum(provisioned - actual) / sum(actual)  [only where provisioned > actual]
```

**Baseline**: Reactive scaling — use actual_cpu at time t to provision for t+1 (no prediction, just previous value)

**RQ2 analysis**:
- Scatter plot: X=R², Y=SLA violation rate (one point per model variant)
- Compute Pearson and Spearman correlation coefficients
- Test statistical significance (p-value)

**RQ3 analysis**:
- Pareto frontier plot: X=SLA violation rate, Y=over-provisioning cost
- Compare predictive (RF) vs reactive baseline
- One curve per model variant

### 5. Visualization (`src/visualization.py`)
Generate publication-quality plots (matplotlib, IEEE style):
1. R² vs SLA violation rate scatter + correlation line + r/p values
2. Pareto frontier (violation rate vs over-provisioning cost)
3. Feature importance bar plot (top 15 features)
4. Actual vs Predicted CPU scatter plot (best model)
5. Time series plot: actual vs predicted for a sample instance

Save all plots to `results/plots/` as PNG (300 dpi) and PDF.

## Code Style
- Python 3.10+
- Type hints
- Docstrings for all functions
- Logging with `logging` module (not print)
- Each script runnable standalone: `python src/data_extraction.py`
- Also provide `src/run_pipeline.py` that runs everything in sequence

## Dependencies
```
pandas
numpy
scikit-learn
matplotlib
seaborn
google-cloud-bigquery
pyarrow
scipy
```
