"""
Phase 2 — Feature Engineering and Disruption Labeling
=======================================================
Sub-tasks:
  1. Disruption label definition (delay threshold + histogram)
  2. Temporal feature extraction (DOW, month, quarter, lag, rolling avg, COVID flag)
  3. Supply-chain graph construction (NetworkX -> PyTorch Geometric)
  4. Class-imbalance handling (SMOTE + class_weight reporting)

Dataset: merged_logistics_supply_chain.csv
  - smart_logistics rows  -> Logistics_Delay binary flag (pre-labelled)
  - supply_chain_disruption rows -> Actual_Lead_Time_Days - Scheduled_Lead_Time_Days gives delay
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend — saves PNGs without a GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import os, pickle, json
from pathlib import Path

# ── optional heavy deps (graceful skip if not installed) ──────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed — SMOTE step will be skipped. "
          "Run:  pip install imbalanced-learn")

try:
    import torch
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[WARN] PyTorch / PyTorch Geometric not installed — graph tensors will "
          "be saved as plain NumPy instead. Run:  pip install torch torch_geometric")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"c:\Users\prati\OneDrive\Desktop\Research")
DATA_FILE    = BASE_DIR / "merged_logistics_supply_chain (1).csv"
OUT_DIR      = BASE_DIR / "phase2_outputs"
OUT_DIR.mkdir(exist_ok=True)

DELAY_THRESHOLD = 2          # days — justify: industry SLA ≤ 2 days buffer (Dolgui et al., 2020)
COVID_START     = pd.Timestamp("2020-01-01")
COVID_END       = pd.Timestamp("2022-12-31")

print("=" * 70)
print("  PHASE 2 — Feature Engineering & Disruption Labeling")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading dataset …")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"    Raw shape: {df.shape}")
print(f"    Sources  : {df['source'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SUB-TASK 1 — DISRUPTION LABEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Sub-task 1 — Disruption label …")

# ── 2a. supply_chain_disruption rows ──────────────────────────────────────────
sc_mask = df["source"] == "supply_chain_disruption"
df_sc   = df[sc_mask].copy()

# delay = Actual − Scheduled  (already have Delay_Days column for verification)
df_sc["delay_computed"] = (
    pd.to_numeric(df_sc["Actual_Lead_Time_Days"], errors="coerce") -
    pd.to_numeric(df_sc["Scheduled_Lead_Time_Days"], errors="coerce")
)

# Use the pre-computed Delay_Days where available, fall back to computed
df_sc["delay"] = df_sc["Delay_Days"].combine_first(df_sc["delay_computed"])
df_sc["disruption"] = (df_sc["delay"] > DELAY_THRESHOLD).astype(int)

# ── 2b. smart_logistics rows ──────────────────────────────────────────────────
sl_mask = df["source"] == "smart_logistics"
df_sl   = df[sl_mask].copy()

# Logistics_Delay is already 0/1
df_sl["delay"]       = pd.to_numeric(df_sl["Logistics_Delay"], errors="coerce")
df_sl["disruption"]  = df_sl["delay"].fillna(0).astype(int)

# ── 2c. Merge back ────────────────────────────────────────────────────────────
df_all = pd.concat([df_sc, df_sl], ignore_index=True)
print(f"    Total rows after concat : {len(df_all)}")

# ── 2d. Delay distribution histogram (supply_chain_disruption only, has real delay) ──
delays_sc = df_sc["delay"].dropna()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Delay Distribution — Supply Chain Disruption Dataset\n"
    f"Disruption threshold (red dashed) = {DELAY_THRESHOLD} days",
    fontsize=13, fontweight="bold"
)

# Full range
axes[0].hist(delays_sc, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(DELAY_THRESHOLD, color="crimson", linewidth=2.5,
                linestyle="--", label=f"Threshold = {DELAY_THRESHOLD} d")
axes[0].set_xlabel("Delay (days)", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)
axes[0].set_title("Full Delay Range", fontsize=11)
axes[0].legend(fontsize=10)

# Near-threshold zoom
zoom_data = delays_sc[(delays_sc >= -5) & (delays_sc <= 20)]
axes[1].hist(zoom_data, bins=25, color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].axvline(DELAY_THRESHOLD, color="crimson", linewidth=2.5,
                linestyle="--", label=f"Threshold = {DELAY_THRESHOLD} d")
axes[1].set_xlabel("Delay (days)", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].set_title("Zoom: −5 to 20 Days", fontsize=11)
axes[1].legend(fontsize=10)

plt.tight_layout()
hist_path = OUT_DIR / "delay_distribution_histogram.png"
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"    Histogram saved -> {hist_path}")

# ── 2e. Class ratio ───────────────────────────────────────────────────────────
label_counts = df_all["disruption"].value_counts().sort_index()
print(f"\n    CLASS DISTRIBUTION (raw):")
print(f"      No disruption (0): {label_counts.get(0, 0):>6}  "
      f"({label_counts.get(0,0)/len(df_all)*100:.1f}%)")
print(f"      Disruption    (1): {label_counts.get(1, 0):>6}  "
      f"({label_counts.get(1,0)/len(df_all)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SUB-TASK 2 — TEMPORAL FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Sub-task 2 — Temporal features …")

# Primary date column (supply_chain rows use Order_Date; smart_logistics use Timestamp)
df_all["_date_raw"] = df_all["Order_Date"].combine_first(df_all["Timestamp"])
df_all["date"] = pd.to_datetime(df_all["_date_raw"], errors="coerce")

# Drop rows where we cannot parse any date
n_before = len(df_all)
df_all = df_all.dropna(subset=["date"])
print(f"    Rows with parseable date: {len(df_all)}  (dropped {n_before - len(df_all)} unparseable)")

# Sort by date for lag/rolling computation
df_all = df_all.sort_values("date").reset_index(drop=True)

# ── Basic calendar features ───────────────────────────────────────────────────
df_all["day_of_week"]   = df_all["date"].dt.dayofweek          # 0=Mon … 6=Sun
df_all["month"]         = df_all["date"].dt.month
df_all["quarter"]       = df_all["date"].dt.quarter
df_all["week_of_year"]  = df_all["date"].dt.isocalendar().week.astype(int)
df_all["year"]          = df_all["date"].dt.year
df_all["is_weekend"]    = (df_all["day_of_week"] >= 5).astype(int)

# ── COVID period flag ─────────────────────────────────────────────────────────
df_all["covid_period"]  = (
    (df_all["date"] >= COVID_START) & (df_all["date"] <= COVID_END)
).astype(int)
print(f"    COVID-period rows : {df_all['covid_period'].sum()}")

# ── Numeric delay (from supply_chain_disruption) for rolling stats ─────────────
# For smart_logistics rows we don't have a numeric delay; fill 0 for rolling only
df_all["delay_num"] = df_all["delay"].where(df_all["source"] == "supply_chain_disruption", other=np.nan)
# Use the Logistics_Delay binary as a proxy for smart_logistics delay
df_all["delay_num"] = df_all["delay_num"].fillna(df_all["disruption"].astype(float))

# ── Lag feature: days since last disruption ────────────────────────────────────
disrupt_dates = df_all.loc[df_all["disruption"] == 1, "date"].values
df_all["days_since_last_disruption"] = df_all["date"].apply(
    lambda d: (
        (d - pd.Timestamp(disrupt_dates[disrupt_dates < np.datetime64(d)].max())).days
        if len(disrupt_dates[disrupt_dates < np.datetime64(d)]) > 0 else -1
    )
)

# ── Rolling 7-day average delay (using date index) ───────────────────────────
df_all = df_all.set_index("date")
df_all["rolling_7d_avg_delay"] = (
    df_all["delay_num"]
    .rolling("7D", min_periods=1)
    .mean()
)
df_all = df_all.reset_index()  # bring date back as column

print("    Temporal features created:")
temporal_cols = [
    "day_of_week", "month", "quarter", "week_of_year", "year",
    "is_weekend", "covid_period",
    "days_since_last_disruption", "rolling_7d_avg_delay"
]
print(f"      {temporal_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SUB-TASK 3 — SUPPLY CHAIN GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Sub-task 3 — Building supply chain graph …")

# ── Use supply_chain_disruption rows (they have Origin/Destination) ────────────
df_graph = df_all[df_all["source"] == "supply_chain_disruption"].copy()
df_graph = df_graph.dropna(subset=["Origin_City", "Destination_City"])

print(f"    Rows used for graph: {len(df_graph)}")
print(f"    Unique origins     : {df_graph['Origin_City'].nunique()}")
print(f"    Unique destinations: {df_graph['Destination_City'].nunique()}")

# ── Build edge list: (origin -> destination) with weight = shipment count ───────
edge_df = (
    df_graph.groupby(["Origin_City", "Destination_City"])
    .agg(
        shipment_count  = ("Order_ID",        "count"),
        avg_delay       = ("delay",            "mean"),
        disruption_rate = ("disruption",       "mean"),
        total_weight_kg = ("Order_Weight_Kg",  "sum"),
    )
    .reset_index()
)

# ── NetworkX directed graph ───────────────────────────────────────────────────
G = nx.DiGraph()

# Add nodes (unique cities)
all_nodes = pd.unique(edge_df[["Origin_City", "Destination_City"]].values.ravel())
G.add_nodes_from(all_nodes)

# Add edges
for _, row in edge_df.iterrows():
    G.add_edge(
        row["Origin_City"],
        row["Destination_City"],
        weight          = float(row["shipment_count"]),
        avg_delay       = float(row["avg_delay"]),
        disruption_rate = float(row["disruption_rate"]),
        total_weight_kg = float(row["total_weight_kg"]),
    )

print(f"\n    Graph stats:")
print(f"      Nodes : {G.number_of_nodes()}")
print(f"      Edges : {G.number_of_edges()}")
print(f"      Density: {nx.density(G):.4f}")

# ── Compute node-level graph features ─────────────────────────────────────────
in_degree    = dict(G.in_degree())
out_degree   = dict(G.out_degree())
betweenness  = nx.betweenness_centrality(G, weight="weight", normalized=True)
pagerank     = nx.pagerank(G, weight="weight")

node_list     = list(G.nodes())
node_index    = {n: i for i, n in enumerate(node_list)}

node_features = np.array([
    [
        in_degree[n],
        out_degree[n],
        betweenness[n],
        pagerank[n],
    ]
    for n in node_list
], dtype=np.float32)

print(f"\n    Node feature matrix shape: {node_features.shape}")
print(f"    Features: [in_degree, out_degree, betweenness_centrality, pagerank]")

# ── Edge index + edge attributes ──────────────────────────────────────────────
edge_index_list = []
edge_attr_list  = []

for u, v, data in G.edges(data=True):
    edge_index_list.append([node_index[u], node_index[v]])
    edge_attr_list.append([
        data["weight"],
        data["avg_delay"],
        data["disruption_rate"],
        data["total_weight_kg"],
    ])

edge_index_np = np.array(edge_index_list, dtype=np.int64).T   # shape (2, E)
edge_attr_np  = np.array(edge_attr_list,  dtype=np.float32)   # shape (E, 4)

# ── Convert to PyTorch Geometric Data object ──────────────────────────────────
if PYG_AVAILABLE:
    pyg_data = PyGData(
        x          = torch.tensor(node_features, dtype=torch.float),
        edge_index = torch.tensor(edge_index_np,  dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr_np,   dtype=torch.float),
    )
    torch.save(pyg_data, OUT_DIR / "supply_chain_graph.pt")
    print(f"\n    PyG Data object saved -> {OUT_DIR / 'supply_chain_graph.pt'}")
    print(f"    x.shape          : {pyg_data.x.shape}")
    print(f"    edge_index.shape : {pyg_data.edge_index.shape}")
    print(f"    edge_attr.shape  : {pyg_data.edge_attr.shape}")
else:
    np.save(OUT_DIR / "graph_node_features.npy", node_features)
    np.save(OUT_DIR / "graph_edge_index.npy",    edge_index_np)
    np.save(OUT_DIR / "graph_edge_attr.npy",     edge_attr_np)
    print(f"\n    Graph arrays (NumPy) saved to {OUT_DIR}/")

# ── Save node lookup ──────────────────────────────────────────────────────────
with open(OUT_DIR / "graph_node_index.json", "w") as f:
    json.dump(node_index, f, indent=2)

# ── Visualise top-N nodes subgraph ────────────────────────────────────────────
top_n       = min(25, G.number_of_nodes())
top_nodes   = sorted(pagerank, key=pagerank.get, reverse=True)[:top_n]
subgraph    = G.subgraph(top_nodes)

fig, ax = plt.subplots(figsize=(14, 10))
pos     = nx.spring_layout(subgraph, seed=42, k=1.8)
edge_w  = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]
max_w   = max(edge_w) if edge_w else 1
node_sz = [pagerank[n] * 30000 for n in subgraph.nodes()]

nx.draw_networkx_nodes(
    subgraph, pos, node_size=node_sz,
    node_color=[betweenness[n] for n in subgraph.nodes()],
    cmap=plt.cm.plasma, alpha=0.85, ax=ax
)
nx.draw_networkx_edges(
    subgraph, pos,
    width=[w / max_w * 4 for w in edge_w],
    edge_color="#555555", arrows=True,
    arrowsize=15, alpha=0.6, ax=ax
)
nx.draw_networkx_labels(subgraph, pos, font_size=7, font_color="white", ax=ax)

ax.set_title(
    f"Supply Chain Graph — Top {top_n} Nodes by PageRank\n"
    "Node size ∝ PageRank | Node color = Betweenness centrality | "
    "Edge width ∝ shipment volume",
    fontsize=11
)
ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                           norm=plt.Normalize(vmin=0, vmax=max(betweenness.values())))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Betweenness centrality", shrink=0.6)
graph_viz_path = OUT_DIR / "supply_chain_graph_viz.png"
plt.savefig(graph_viz_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Graph visualisation saved -> {graph_viz_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SUB-TASK 4 — CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Sub-task 4 — Class imbalance handling …")

# ── Select numeric feature columns available in BOTH source datasets ──────────
FEATURE_COLS = [
    "day_of_week", "month", "quarter", "week_of_year", "is_weekend",
    "covid_period", "days_since_last_disruption", "rolling_7d_avg_delay",
    # supply_chain_disruption specific (NaN for smart_logistics -> fill 0)
    "Geopolitical_Risk_Index", "Weather_Severity_Index",
    "Inflation_Rate_Pct", "Shipping_Cost_USD", "Order_Weight_Kg",
    "Base_Lead_Time_Days", "Scheduled_Lead_Time_Days",
    # smart_logistics specific (NaN for supply_chain rows -> fill 0)
    "Inventory_Level", "Temperature", "Humidity",
    "Waiting_Time", "Asset_Utilization", "Demand_Forecast",
]

# Coerce all feature cols to numeric; fill NaN with column median
df_ml = df_all.copy()
for col in FEATURE_COLS:
    if col in df_ml.columns:
        df_ml[col] = pd.to_numeric(df_ml[col], errors="coerce")
    else:
        df_ml[col] = 0.0

df_ml[FEATURE_COLS] = df_ml[FEATURE_COLS].fillna(df_ml[FEATURE_COLS].median())
df_ml[FEATURE_COLS] = df_ml[FEATURE_COLS].fillna(0.0)          # catch all-NaN columns
df_ml[FEATURE_COLS] = df_ml[FEATURE_COLS].replace([np.inf, -np.inf], 0.0)  # no inf
df_ml = df_ml.dropna(subset=["disruption"])

X = df_ml[FEATURE_COLS].values.astype(np.float32)
y = df_ml["disruption"].values.astype(int)

print(f"\n    Feature matrix X shape : {X.shape}")
print(f"    Label vector  y shape  : {y.shape}")
print(f"\n    Class counts BEFORE imbalance handling:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"      Class {u}: {c:>6}  ({c/len(y)*100:.1f}%)")

# ── SMOTE (if available) ──────────────────────────────────────────────────────
if SMOTE_AVAILABLE:
    print("\n    Applying SMOTE …")
    k_neighbors = min(5, counts.min() - 1)
    if k_neighbors < 1:
        print("    [SKIP] Minority class too small for SMOTE; using class_weight only.")
        X_res, y_res = X, y
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"\n    Class counts AFTER SMOTE:")
        unique_r, counts_r = np.unique(y_res, return_counts=True)
        for u, c in zip(unique_r, counts_r):
            print(f"      Class {u}: {c:>6}  ({c/len(y_res)*100:.1f}%)")
else:
    X_res, y_res = X, y
    print("    SMOTE skipped (imbalanced-learn not installed).")

# ── class_weight='balanced' ratio for sklearn classifiers ────────────────────
n_samples  = len(y)
n_classes  = len(np.unique(y))
class_wts  = {}
for cls in np.unique(y):
    class_wts[cls] = n_samples / (n_classes * np.sum(y == cls))
print(f"\n    class_weight='balanced' values:")
for cls, wt in class_wts.items():
    print(f"      Class {cls}: {wt:.4f}")

# ── Class imbalance bar chart (before vs after) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Imbalance: Before vs After Resampling", fontsize=13, fontweight="bold")

colors = ["#4878D0", "#EE854A"]
labels = ["No Disruption (0)", "Disruption (1)"]

# Before
ax = axes[0]
bc = [np.sum(y == 0), np.sum(y == 1)]
bars = ax.bar(labels, bc, color=colors, edgecolor="white", linewidth=1.2)
ax.set_title("BEFORE (Original)", fontsize=11)
ax.set_ylabel("Count", fontsize=10)
for bar, val in zip(bars, bc):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bc) * 0.01,
            f"{val}\n({val/sum(bc)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

# After
ax = axes[1]
ac = [np.sum(y_res == 0), np.sum(y_res == 1)]
bars = ax.bar(labels, ac, color=colors, edgecolor="white", linewidth=1.2)
title_tag = "SMOTE" if SMOTE_AVAILABLE and k_neighbors >= 1 else "No Resampling"
ax.set_title(f"AFTER ({title_tag})", fontsize=11)
ax.set_ylabel("Count", fontsize=10)
for bar, val in zip(bars, ac):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ac) * 0.01,
            f"{val}\n({val/sum(ac)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
imb_path = OUT_DIR / "class_imbalance_before_after.png"
plt.savefig(imb_path, dpi=150)
plt.close()
print(f"\n    Imbalance chart saved -> {imb_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE ALL PHASE 2 ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Saving artefacts …")

# Enriched dataframe
enriched_path = OUT_DIR / "df_phase2_enriched.parquet"
df_all.to_parquet(enriched_path, index=False)
print(f"    Enriched DataFrame  -> {enriched_path}")

# Feature matrix + labels (original)
np.save(OUT_DIR / "X_features.npy",     X)
np.save(OUT_DIR / "y_labels.npy",       y)

# Resampled (if SMOTE ran)
np.save(OUT_DIR / "X_resampled.npy",    X_res)
np.save(OUT_DIR / "y_resampled.npy",    y_res)

# Feature column names
pd.Series(FEATURE_COLS).to_csv(OUT_DIR / "feature_columns.csv", index=False, header=False)

# NetworkX graph
with open(OUT_DIR / "supply_chain_graph.pkl", "wb") as f:
    pickle.dump(G, f)

# Summary JSON
summary = {
    "total_rows"                : int(len(df_all)),
    "disruption_threshold_days" : DELAY_THRESHOLD,
    "class_distribution_original": {
        "class_0": int(np.sum(y == 0)),
        "class_1": int(np.sum(y == 1)),
    },
    "class_distribution_resampled": {
        "class_0": int(np.sum(y_res == 0)),
        "class_1": int(np.sum(y_res == 1)),
    },
    "class_weights_balanced": {str(k): round(v, 4) for k, v in class_wts.items()},
    "graph_nodes"  : G.number_of_nodes(),
    "graph_edges"  : G.number_of_edges(),
    "graph_density": round(nx.density(G), 6),
    "feature_count": len(FEATURE_COLS),
    "temporal_features": temporal_cols,
    "smote_applied": SMOTE_AVAILABLE and len(y_res) > len(y),
    "pyg_available": PYG_AVAILABLE,
    "covid_window" : f"{COVID_START.date()} -> {COVID_END.date()}",
}
with open(OUT_DIR / "phase2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"    Summary JSON        -> {OUT_DIR / 'phase2_summary.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 2 COMPLETE -- All outputs saved to:")
print(f"  {OUT_DIR}")
print("=" * 70)
print(f"""
  OUTPUT FILES
  ---------------------------------------------------------
  delay_distribution_histogram.png   (Sub-task 1 justification plot)
  class_imbalance_before_after.png   (Sub-task 4 imbalance plot)
  supply_chain_graph_viz.png         (Sub-task 3 graph visualisation)
  df_phase2_enriched.parquet         (enriched DataFrame for Phase 3)
  X_features.npy / y_labels.npy     (original feature matrix + labels)
  X_resampled.npy / y_resampled.npy (SMOTE-resampled, if available)
  feature_columns.csv                (list of feature names for Phase 3)
  supply_chain_graph.pkl             (NetworkX DiGraph)
  supply_chain_graph.pt              (PyG Data object, if torch available)
  graph_node_index.json              (city -> integer node mapping)
  phase2_summary.json                (all key stats for paper reporting)
  ---------------------------------------------------------

  KEY NUMBERS FOR YOUR PAPER
  ---------------------------------------------------------
  Disruption threshold : {DELAY_THRESHOLD} days  (cite: Dolgui et al., 2020)
  Graph: {summary['graph_nodes']} nodes, {summary['graph_edges']} edges, density={summary['graph_density']}
  Class 0: {summary['class_distribution_original']['class_0']}   Class 1: {summary['class_distribution_original']['class_1']}
  SMOTE applied        : {summary['smote_applied']}
""")

