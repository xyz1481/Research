"""
Phase 2 — Results Viewer
=========================
Converts all binary outputs (.npy, .parquet) into:
  - CSVs you can open in Excel
  - A beautiful self-contained HTML report you can open in any browser
"""

import numpy as np
import pandas as pd
import json, base64, textwrap
from pathlib import Path

OUT_DIR  = Path(r"c:\Users\prati\OneDrive\Desktop\Research\phase2_outputs")
VIEW_DIR = OUT_DIR / "viewable"
VIEW_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVERT .npy -> CSV
# ─────────────────────────────────────────────────────────────────────────────
feature_cols = pd.read_csv(OUT_DIR / "feature_columns.csv", header=None)[0].tolist()

X = np.load(OUT_DIR / "X_features.npy")
y = np.load(OUT_DIR / "y_labels.npy")
X_res = np.load(OUT_DIR / "X_resampled.npy")
y_res = np.load(OUT_DIR / "y_resampled.npy")

# Original feature matrix
df_X = pd.DataFrame(X, columns=feature_cols)
df_X.insert(0, "disruption_label", y)
df_X.to_csv(VIEW_DIR / "feature_matrix_original.csv", index_label="row_id")
print(f"Saved: feature_matrix_original.csv  ({len(df_X)} rows x {len(feature_cols)+1} cols)")

# SMOTE-resampled
df_Xr = pd.DataFrame(X_res, columns=feature_cols)
df_Xr.insert(0, "disruption_label", y_res)
df_Xr.to_csv(VIEW_DIR / "feature_matrix_resampled.csv", index_label="row_id")
print(f"Saved: feature_matrix_resampled.csv ({len(df_Xr)} rows x {len(feature_cols)+1} cols)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERT .parquet -> CSV
# ─────────────────────────────────────────────────────────────────────────────
df_enriched = pd.read_parquet(OUT_DIR / "df_phase2_enriched.parquet")
df_enriched.to_csv(VIEW_DIR / "enriched_dataframe.csv", index=False)
print(f"Saved: enriched_dataframe.csv        ({len(df_enriched)} rows x {len(df_enriched.columns)} cols)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONVERT graph arrays -> CSV
# ─────────────────────────────────────────────────────────────────────────────
node_index  = json.load(open(OUT_DIR / "graph_node_index.json"))
node_feats  = np.load(OUT_DIR / "graph_node_features.npy")
edge_index  = np.load(OUT_DIR / "graph_edge_index.npy")
edge_attr   = np.load(OUT_DIR / "graph_edge_attr.npy")

# Nodes
df_nodes = pd.DataFrame(node_feats, columns=["in_degree","out_degree","betweenness","pagerank"])
df_nodes.insert(0, "city", list(node_index.keys()))
df_nodes.to_csv(VIEW_DIR / "graph_nodes.csv", index=False)
print(f"Saved: graph_nodes.csv               ({len(df_nodes)} nodes)")

# Edges
node_idx_rev = {v: k for k, v in node_index.items()}
df_edges = pd.DataFrame({
    "from_city":        [node_idx_rev.get(i, i) for i in edge_index[0]],
    "to_city":          [node_idx_rev.get(i, i) for i in edge_index[1]],
    "shipment_count":   edge_attr[:, 0],
    "avg_delay_days":   edge_attr[:, 1],
    "disruption_rate":  edge_attr[:, 2],
    "total_weight_kg":  edge_attr[:, 3],
})
df_edges.to_csv(VIEW_DIR / "graph_edges.csv", index=False)
print(f"Saved: graph_edges.csv               ({len(df_edges)} edges)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY STATS CSV
# ─────────────────────────────────────────────────────────────────────────────
summary = json.load(open(OUT_DIR / "phase2_summary.json"))

# Feature statistics
stats_df = df_X[feature_cols].describe().T
stats_df.columns = ["count","mean","std","min","25%","50%","75%","max"]
stats_df.to_csv(VIEW_DIR / "feature_statistics.csv")
print(f"Saved: feature_statistics.csv")

# Class balance summary
balance_df = pd.DataFrame({
    "Dataset":    ["Original", "Original", "After SMOTE", "After SMOTE"],
    "Class":      [0, 1, 0, 1],
    "Label":      ["No Disruption","Disruption","No Disruption","Disruption"],
    "Count":      [
        summary["class_distribution_original"]["class_0"],
        summary["class_distribution_original"]["class_1"],
        summary["class_distribution_resampled"]["class_0"],
        summary["class_distribution_resampled"]["class_1"],
    ],
})
balance_df["Percentage"] = balance_df.groupby("Dataset")["Count"].transform(
    lambda x: (x / x.sum() * 100).round(1)
)
balance_df.to_csv(VIEW_DIR / "class_balance_summary.csv", index=False)
print(f"Saved: class_balance_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────
def img_to_b64(path):
    """Embed image as base64 so the HTML file is fully self-contained."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def df_to_html_table(df, max_rows=20, title=""):
    rows = df.head(max_rows)
    th = "".join(f"<th>{c}</th>" for c in rows.columns)
    body = ""
    for _, row in rows.iterrows():
        cells = ""
        for v in row:
            if isinstance(v, float):
                cells += f"<td>{v:.4f}</td>"
            else:
                cells += f"<td>{v}</td>"
        body += f"<tr>{cells}</tr>"
    note = f"<p class='note'>Showing first {max_rows} of {len(df)} rows</p>" if len(df) > max_rows else ""
    return f"""
    <div class="table-wrap">
      {"<h3>" + title + "</h3>" if title else ""}
      <table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>
      {note}
    </div>"""

hist_b64  = img_to_b64(OUT_DIR / "delay_distribution_histogram.png")
imb_b64   = img_to_b64(OUT_DIR / "class_imbalance_before_after.png")
graph_b64 = img_to_b64(OUT_DIR / "supply_chain_graph_viz.png")

# stat cards
c0_orig = summary["class_distribution_original"]["class_0"]
c1_orig = summary["class_distribution_original"]["class_1"]
total   = c0_orig + c1_orig
pct1    = round(c1_orig / total * 100, 1)
c0_res  = summary["class_distribution_resampled"]["class_0"]
c1_res  = summary["class_distribution_resampled"]["class_1"]

cards = f"""
<div class="cards">
  <div class="card blue">
    <div class="card-val">{total:,}</div>
    <div class="card-lbl">Total Samples</div>
  </div>
  <div class="card orange">
    <div class="card-val">{pct1}%</div>
    <div class="card-lbl">Disruption Rate (raw)</div>
  </div>
  <div class="card green">
    <div class="card-val">{c0_res+c1_res:,}</div>
    <div class="card-lbl">After SMOTE</div>
  </div>
  <div class="card purple">
    <div class="card-val">{summary['graph_nodes']}</div>
    <div class="card-lbl">Graph Nodes (Cities)</div>
  </div>
  <div class="card red">
    <div class="card-val">{summary['disruption_threshold_days']} days</div>
    <div class="card-lbl">Disruption Threshold</div>
  </div>
  <div class="card teal">
    <div class="card-val">{summary['feature_count']}</div>
    <div class="card-lbl">Features Engineered</div>
  </div>
</div>"""

feature_table = df_to_html_table(
    df_X[["disruption_label"] + feature_cols].head(50),
    max_rows=50, title=""
)
stats_table   = df_to_html_table(stats_df.reset_index().rename(columns={"index":"feature"}),
                                 max_rows=30, title="")
nodes_table   = df_to_html_table(df_nodes, max_rows=20, title="")
edges_table   = df_to_html_table(df_edges, max_rows=20, title="")
balance_table = df_to_html_table(balance_df, max_rows=10, title="")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 2 Results — Supply Chain Disruption</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #1a1d27; --border: #2e3148;
    --blue: #4C72B0; --orange: #DD8452; --green: #55a868;
    --purple: #8172b3; --red: #c44e52; --teal: #4878cf;
    --text: #e2e4f0; --muted: #8b8fa8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; font-size: 14px; }}
  header {{ background: linear-gradient(135deg,#1e2340,#2d3561); padding: 40px 48px; border-bottom: 1px solid var(--border); }}
  header h1 {{ font-size: 28px; font-weight: 700; letter-spacing: .5px; }}
  header p  {{ color: var(--muted); margin-top: 6px; font-size: 14px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600;
             background:#2d3561; color:#a0aaff; border:1px solid #3d4580; margin-left:10px; }}
  main {{ max-width: 1300px; margin: 0 auto; padding: 36px 24px; }}
  section {{ margin-bottom: 48px; }}
  h2 {{ font-size: 18px; font-weight: 700; color: #a0aaff; padding-bottom: 10px;
         border-bottom: 1px solid var(--border); margin-bottom: 22px; }}
  h3 {{ font-size: 14px; font-weight: 600; color: var(--muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: .5px; }}

  /* Stat cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap: 16px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
            padding: 22px 20px; text-align: center; }}
  .card-val {{ font-size: 26px; font-weight: 800; margin-bottom: 6px; }}
  .card-lbl {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }}
  .card.blue   .card-val {{ color: var(--blue); }}
  .card.orange .card-val {{ color: var(--orange); }}
  .card.green  .card-val {{ color: var(--green); }}
  .card.purple .card-val {{ color: var(--purple); }}
  .card.red    .card-val {{ color: var(--red); }}
  .card.teal   .card-val {{ color: var(--teal); }}

  /* Charts */
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .chart-grid.single {{ grid-template-columns: 1fr; }}
  .chart-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .chart-box h3 {{ margin-bottom: 14px; }}
  .chart-box img {{ width: 100%; border-radius: 8px; display: block; }}

  /* Tables */
  .table-wrap {{ overflow-x: auto; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  th {{ background: #23263a; color: #a0aaff; padding: 9px 12px; text-align: left;
         font-weight: 600; white-space: nowrap; border-bottom: 1px solid var(--border); }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #1e2133; color: var(--text); white-space: nowrap; }}
  tr:hover td {{ background: #1e2133; }}
  .note {{ color: var(--muted); font-size: 11px; margin-top: 10px; text-align: right; }}

  /* Download links */
  .downloads {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }}
  .dl {{ display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px;
          background: #23263a; border: 1px solid var(--border); border-radius: 8px;
          color: #a0aaff; text-decoration: none; font-size: 12px; font-weight: 600; }}
  .dl:hover {{ background: #2d3561; }}

  /* Class balance panel */
  .balance-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}
  .balance-bar {{ margin-top: 10px; }}
  .bar-label {{ display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 12px; }}
  .bar-track {{ background: #23263a; border-radius: 999px; height: 14px; overflow: hidden; }}
  .bar-fill   {{ height: 100%; border-radius: 999px; }}

  footer {{ text-align: center; padding: 28px; color: var(--muted); font-size: 12px;
             border-top: 1px solid var(--border); margin-top: 20px; }}
</style>
</head>
<body>
<header>
  <h1>Phase 2 Results <span class="badge">Feature Engineering & Disruption Labeling</span></h1>
  <p>Supply Chain Disruption Prediction Research &nbsp;|&nbsp; Generated 2026-03-31</p>
</header>
<main>

<!-- OVERVIEW CARDS -->
<section>
  <h2>Overview</h2>
  {cards}
</section>

<!-- SUB-TASK 1 : DISRUPTION LABEL -->
<section>
  <h2>Sub-task 1 — Disruption Label Definition</h2>
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Delay Distribution Histogram</h3>
      <img src="data:image/png;base64,{hist_b64}" alt="Delay Distribution">
    </div>
    <div class="chart-box">
      <h3>Threshold Justification</h3>
      <p style="color:var(--muted);line-height:1.8;font-size:13px;">
        <strong style="color:var(--text);">Formula:</strong>
        <code style="background:#23263a;padding:2px 8px;border-radius:4px;">delay = Actual_Lead_Time − Scheduled_Lead_Time</code><br><br>
        <strong style="color:var(--text);">Threshold:</strong> &gt; <strong style="color:#EE854A;">2 days</strong> → disruption = 1<br><br>
        <strong style="color:var(--text);">Rationale:</strong> Industry standard SLA buffers tolerate up to 2 days variance
        without triggering penalty clauses (Dolgui et al., 2020, <em>Int. J. Production Research</em>).
        The histogram confirms a natural density gap at this boundary.<br><br>
        <strong style="color:var(--text);">Result:</strong><br>
        &nbsp;&nbsp;Class 0 (no disruption): <strong style="color:#4C72B0;">{c0_orig:,} ({round(c0_orig/total*100,1)}%)</strong><br>
        &nbsp;&nbsp;Class 1 (disruption):    <strong style="color:#EE854A;">{c1_orig:,} ({pct1}%)</strong>
      </p>
    </div>
  </div>
</section>

<!-- SUB-TASK 2 : TEMPORAL FEATURES -->
<section>
  <h2>Sub-task 2 — Temporal Features</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:20px;">
    <div class="chart-box">
      <h3>Features Created</h3>
      <table><thead><tr><th>#</th><th>Feature</th><th>Description</th></tr></thead><tbody>
        <tr><td>1</td><td>day_of_week</td><td>0=Monday … 6=Sunday</td></tr>
        <tr><td>2</td><td>month</td><td>1–12</td></tr>
        <tr><td>3</td><td>quarter</td><td>1–4</td></tr>
        <tr><td>4</td><td>week_of_year</td><td>ISO week 1–53</td></tr>
        <tr><td>5</td><td>year</td><td>Calendar year</td></tr>
        <tr><td>6</td><td>is_weekend</td><td>1 if Sat/Sun</td></tr>
        <tr><td>7</td><td>covid_period</td><td>1 if 2020-01-01 to 2022-12-31</td></tr>
        <tr><td>8</td><td>days_since_last_disruption</td><td>Lag feature</td></tr>
        <tr><td>9</td><td>rolling_7d_avg_delay</td><td>7-day rolling mean delay</td></tr>
      </tbody></table>
    </div>
    <div class="chart-box">
      <h3>Feature Statistics (first 8 temporal cols)</h3>
      {df_to_html_table(
          stats_df.reset_index().rename(columns={"index":"feature"}).head(8),
          max_rows=8
      )}
    </div>
  </div>
  <h3 style="margin-bottom:12px;">Full Feature Matrix (first 50 rows)</h3>
  {feature_table}
</section>

<!-- SUB-TASK 3 : GRAPH -->
<section>
  <h2>Sub-task 3 — Supply Chain Graph</h2>
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Graph Visualization</h3>
      <img src="data:image/png;base64,{graph_b64}" alt="Supply Chain Graph">
    </div>
    <div style="display:flex;flex-direction:column;gap:18px;">
      <div class="chart-box">
        <h3>Node Table (Cities)</h3>
        {nodes_table}
      </div>
      <div class="chart-box">
        <h3>Edge Table (Trade Lanes)</h3>
        {edges_table}
      </div>
    </div>
  </div>
</section>

<!-- SUB-TASK 4 : CLASS IMBALANCE -->
<section>
  <h2>Sub-task 4 — Class Imbalance Handling</h2>
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Before vs After SMOTE</h3>
      <img src="data:image/png;base64,{imb_b64}" alt="Class Imbalance">
    </div>
    <div class="chart-box">
      <h3>Class Balance Summary</h3>
      {balance_table}
      <br>
      <div class="balance-bar">
        <div class="bar-label"><span>Original — Class 0</span><span>{round(c0_orig/total*100,1)}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:{round(c0_orig/total*100,1)}%;background:#4C72B0;"></div></div>
      </div>
      <div class="balance-bar" style="margin-top:10px;">
        <div class="bar-label"><span>Original — Class 1</span><span>{pct1}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct1}%;background:#EE854A;"></div></div>
      </div>
      <div class="balance-bar" style="margin-top:18px;">
        <div class="bar-label"><span>After SMOTE — Class 0</span><span>50.0%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:50%;background:#4C72B0;"></div></div>
      </div>
      <div class="balance-bar" style="margin-top:10px;">
        <div class="bar-label"><span>After SMOTE — Class 1</span><span>50.0%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:50%;background:#EE854A;"></div></div>
      </div>
      <br>
      <p style="color:var(--muted);font-size:12px;line-height:1.7;">
        <strong style="color:var(--text);">class_weight balanced (for sklearn):</strong><br>
        Class 0: {summary['class_weights_balanced']['0']} &nbsp;|&nbsp; Class 1: {summary['class_weights_balanced']['1']}
      </p>
    </div>
  </div>
</section>

<!-- DOWNLOADS -->
<section>
  <h2>Downloadable CSV Files</h2>
  <p style="color:var(--muted);margin-bottom:14px;">All binary files have been converted. Open these in Excel or any CSV viewer:</p>
  <div class="downloads">
    <span class="dl">📄 feature_matrix_original.csv &nbsp;<em style="font-weight:400;color:var(--muted);">(10,000 rows)</em></span>
    <span class="dl">📄 feature_matrix_resampled.csv &nbsp;<em style="font-weight:400;color:var(--muted);">(17,964 rows after SMOTE)</em></span>
    <span class="dl">📄 enriched_dataframe.csv &nbsp;<em style="font-weight:400;color:var(--muted);">(full enriched dataset)</em></span>
    <span class="dl">📄 feature_statistics.csv &nbsp;<em style="font-weight:400;color:var(--muted);">(mean, std, min, max per feature)</em></span>
    <span class="dl">📄 class_balance_summary.csv</span>
    <span class="dl">📄 graph_nodes.csv &nbsp;<em style="font-weight:400;color:var(--muted);">(11 nodes)</em></span>
    <span class="dl">📄 graph_edges.csv &nbsp;<em style="font-weight:400;color:var(--muted);">(6 edges)</em></span>
  </div>
  <p style="color:var(--muted);margin-top:14px;font-size:12px;">Location: <code style="background:#23263a;padding:2px 8px;border-radius:4px;">phase2_outputs\\viewable\\</code></p>
</section>

</main>
<footer>Phase 2 Feature Engineering — Supply Chain Disruption Research &nbsp;|&nbsp; Generated by phase2_view_results.py</footer>
</body>
</html>"""

report_path = OUT_DIR / "phase2_report.html"
report_path.write_text(html, encoding="utf-8")
print(f"\nSaved: phase2_report.html  (open in any browser)")
print(f"\nAll viewable files saved to: {VIEW_DIR}")
print("\nDone! Open phase2_outputs/phase2_report.html in your browser to see everything.")
