# Project: Predictive Auto-Scaling Research

## Structure

```
v1/   — BigQuery-based pipeline (RQ2/RQ3, Google Cluster Trace v3)
v2/   — Local CSV-based pipeline (RQ1, borg_traces_data)
```

---

## v1 — RQ2/RQ3: R² vs SLA Violation Correlation

> Original pipeline using Google Cluster Trace v3 via BigQuery.
> See `v1/` for all source files and docs.

**Research Questions**
- RQ2: Correlation between prediction accuracy (R², MAE) and SLA violation rate
- RQ3: Trade-off between SLA violation reduction and over-provisioning cost

**Dataset**: Google Cluster Trace v3 (BigQuery `google.com:google-cluster-data`, cell A)

**Pipeline**: `v1/src/`
- `data_extraction.py` → BigQuery → `data/raw/`
- `feature_engineering.py` → `data/processed/`
- `model.py` → 7 RF variants with deliberately varied R²
- `sla_simulation.py` → violation rate + over-provisioning cost
- `visualization.py` → publication plots
- `run_pipeline.py` → runs all steps

**Data Access**:
```bash
export GCP_PROJECT_ID="your-project-id-here"
python v1/src/run_pipeline.py
```

---

## v2 — RQ1: Tier-Aware Prediction (Production vs Non-prod)

> Local CSV pipeline, no BigQuery needed.

**Dataset**: `C:\Users\hynbb\Desktop\borg_traces_data - Copy.csv` (1,324,694 rows)

**Key columns**:
- `priority`: tier classification (120–359 = Production, else Non-prod)
- `average_usage`: `{'cpus': X, 'memory': Y}` — parsed to `avg_cpu`, `avg_mem`
- `resource_request`: `{'cpus': X, 'memory': Y}` — parsed to `resource_request_cpu/memory`
- `start_time`: temporal features
- `cpu_usage_distribution`: p90 extraction (index 9)
- `tail_cpu_usage_distribution`: p95 extraction (index 0)
- `failed`: reserved for RQ3 — do not use in v2

**Tier Definition**:
- Production: `priority` 120–359
- Non-prod: everything else

**Targets** (t+1):
- `avg_cpu_target`
- `avg_mem_target`

**Train/Test Split**: chronological 80/20 — no random shuffle

### Feature Set

| Group | Features |
|-------|----------|
| Lag | avg_cpu_lag1/2/3, avg_mem_lag1, max_cpu_lag1 |
| Rolling (6-window) | avg_cpu_roll_mean6, avg_cpu_roll_std6 |
| Percentile | cpu_p90, cpu_p95 |
| Hardware | cycles_per_instruction, memory_accesses_per_instruction |
| Context | priority, scheduling_class, resource_request_cpu, resource_request_memory |
| Temporal | hour_of_day, day_of_week |
| Tier | is_production |
| Interaction (B/C only) | prod_x_hour, prod_x_dow, sc_x_std |

### Models

| Model | Features | Weights | Notes |
|-------|----------|---------|-------|
| A | Base 18 | None | Baseline RF |
| B | 21 (base + interaction) | None | Interaction features |
| C | 21 (base + interaction) | prod=3, non-prod=1 | Weighted RF |

All models: `n_estimators=100, random_state=42`

### Evaluation

Per model, report:
- Overall: R², MAE, RMSE
- Production tier only: R²_prod, MAE_prod, RMSE_prod

### Output Files

- `v2/results/rq1_model_comparison.csv`
- `v2/results/predictions_per_instance.csv`
  - Columns: instance_id, actual_cpu, predicted_cpu, actual_mem, predicted_mem, priority, is_production, cpu_status, mem_status
  - cpu overload: predicted > resource_request_cpu
  - cpu underload: predicted < resource_request_cpu × 0.5
  - (same logic for memory)
- `v2/results/feature_importance.csv`
- `v2/results/pipeline.log`

### Running v2

```bash
cd v2
pip install -r requirements.txt
python src/run_pipeline.py
# or with explicit path:
python src/run_pipeline.py --csv "C:/Users/hynbb/Desktop/borg_traces_data - Copy.csv"
```

---

## Code Style (both versions)
- Python 3.10+
- Type hints
- Docstrings for all functions
- `logging` module (not print)
- Each script runnable standalone
