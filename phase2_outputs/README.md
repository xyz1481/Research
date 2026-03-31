# Phase 2 Outputs — Feature Engineering & Disruption Labeling

**Script:** `../phase2_feature_engineering.py`  
**Status:** COMPLETE  
**Date run:** 2026-03-31

---

## What Phase 2 Does

Takes the raw merged dataset and produces everything needed for the three
parallel feature extractors in Phase 3/4/5.

```
Raw merged CSV
    |
    v
[Sub-task 1]  Disruption label definition
    - delay = Actual_Lead_Time - Scheduled_Lead_Time
    - disruption = 1 if delay > 2 days  (Dolgui et al., 2020)
    |
    +--> [Sub-task 2]  Temporal features
    |       day_of_week, month, quarter, week_of_year, is_weekend,
    |       covid_period, days_since_last_disruption, rolling_7d_avg_delay
    |
    +--> [Sub-task 3]  Supply-chain graph (NetworkX -> PyG)
    |       Nodes = shipping cities, Edges = trade lanes
    |       Edge weight = shipment volume / frequency
    |
    +--> [Sub-task 4]  Class imbalance handling
            SMOTE: 10.2% minority -> 50/50 balanced
            class_weight=balanced: Class 1 weight = 4.91x
```

---

## Output Files

| File | Format | Shape / Size | Description |
|------|--------|-------------|-------------|
| `df_phase2_enriched.parquet` | Parquet | 10,000 × 51 | Full enriched dataset — **load this in Phase 3** |
| `X_features.npy` | NumPy | 10,000 × 21 | Original feature matrix |
| `y_labels.npy` | NumPy | (10,000,) | Binary disruption labels |
| `X_resampled.npy` | NumPy | 17,964 × 21 | SMOTE-balanced feature matrix |
| `y_resampled.npy` | NumPy | (17,964,) | SMOTE-balanced labels |
| `supply_chain_graph.pkl` | Pickle | — | NetworkX `DiGraph` object |
| `graph_node_features.npy` | NumPy | 11 × 4 | Node features: [in_deg, out_deg, betweenness, pagerank] |
| `graph_edge_index.npy` | NumPy | 2 × 6 | COO edge index for PyG |
| `graph_edge_attr.npy` | NumPy | 6 × 4 | Edge attrs: [count, avg_delay, disruption_rate, weight_kg] |
| `graph_node_index.json` | JSON | — | City name → integer node mapping |
| `feature_columns.csv` | CSV | 21 rows | Ordered list of feature names |
| `phase2_summary.json` | JSON | — | All key stats (copy numbers into paper) |
| `delay_distribution_histogram.png` | PNG | — | Figure for Sub-task 1 justification |
| `class_imbalance_before_after.png` | PNG | — | Figure for Sub-task 4 |
| `supply_chain_graph_viz.png` | PNG | — | Figure for Sub-task 3 |
| `phase2_report.html` | HTML | — | **Open in browser** — full visual report |

### Viewable CSVs (`viewable/` subfolder)

| File | Rows | Description |
|------|------|-------------|
| `feature_matrix_original.csv` | 10,000 | All features + label — open in Excel |
| `feature_matrix_resampled.csv` | 17,964 | SMOTE result |
| `enriched_dataframe.csv` | 10,000 | All 51 columns |
| `feature_statistics.csv` | 21 | Mean, std, min, max per feature |
| `class_balance_summary.csv` | 4 | Before/after class counts |
| `graph_nodes.csv` | 11 | City-level centrality scores |
| `graph_edges.csv` | 6 | Trade lane stats |

---

## Key Numbers for Paper

```
Disruption threshold  : 2 days   (cite Dolgui et al., 2020)
Class 0 (no disruption) : 8,982  (89.8% of 10,000)
Class 1 (disruption)    : 1,018  (10.2%)
After SMOTE             : 8,982 / 8,982  (50% / 50%)
class_weight balanced   : {0: 0.5567, 1: 4.9116}
Graph                   : 11 nodes, 6 edges, density=0.0545
Features engineered     : 21 total (9 temporal + 12 operational)
```

---

## How to Load in Python

```python
import numpy as np, pandas as pd, pickle

# Feature matrix + labels
X = np.load("X_features.npy")           # (10000, 21)
y = np.load("y_labels.npy")             # (10000,)

# SMOTE balanced
X_res = np.load("X_resampled.npy")      # (17964, 21)
y_res = np.load("y_resampled.npy")      # (17964,)

# Full enriched dataframe
df = pd.read_parquet("df_phase2_enriched.parquet")

# NetworkX graph
with open("supply_chain_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Graph tensors (for PyG)
import numpy as np
node_feats  = np.load("graph_node_features.npy")  # (11, 4)
edge_index  = np.load("graph_edge_index.npy")      # (2, 6)
edge_attr   = np.load("graph_edge_attr.npy")       # (6, 4)
```

---

## Next Step

**Phase 3:** `../phase3_finbert_encoder.py`  
Generates FinBERT text embeddings from the enriched dataset.  
Input consumed: `df_phase2_enriched.parquet`
