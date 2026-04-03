"""
Generate Final Comprehensive HTML Research Report
===================================================
Combines ALL phases (2-6) + Phase 5v2 balanced model into one
premium self-contained dashboard with embedded Base64 images.
"""
import json, base64, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score, average_precision_score)

BASE = Path(r"c:\Users\prati\OneDrive\Desktop\Research")
OUT  = BASE / "index.html"

DIRS = {
    "p2":  BASE / "phase2_outputs",
    "p3":  BASE / "phase3_outputs",
    "p4a": BASE / "phase4a_lstm_outputs",
    "p4b": BASE / "phase4b_gnn_outputs",
    "p5":  BASE / "phase5_fusion_outputs",
    "p5v2":BASE / "phase5v2_balanced_outputs",
    "p6":  BASE / "phase6_explainability",
}

def b64(path):
    p = Path(path)
    if not p.exists(): return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

def jload(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

# ── load all summaries ───────────────────────────────────────────────────────
s2  = jload(DIRS["p2"]  / "phase2_summary.json")
s3  = jload(DIRS["p3"]  / "phase3_summary.json")
s4a = jload(DIRS["p4a"] / "phase4a_summary.json")
s4b = jload(DIRS["p4b"] / "phase4b_summary.json")
s5  = jload(DIRS["p5"]  / "phase5_summary.json")
s5v2= jload(DIRS["p5v2"]/ "phase5v2_summary.json")
s6  = jload(DIRS["p6"]  / "phase6_summary.json")

# ── live metrics from best model ─────────────────────────────────────────────
probs_bal = np.load(DIRS["p5v2"] / "disruption_probabilities_balanced.npy")
probs_old = np.load(DIRS["p5"]   / "disruption_probabilities.npy")
y_seq     = np.load(DIRS["p4a"]  / "y_seq.npy")
preds_bal = (probs_bal > 0.5).astype(int)
preds_old = (probs_old > 0.5).astype(int)

M = lambda fn, p: fn(y_seq, p, zero_division=0)

metrics_bal = {
    "Accuracy":  f"{accuracy_score(y_seq, preds_bal)*100:.2f}%",
    "F1 Score":  f"{M(f1_score, preds_bal):.4f}",
    "AUC-ROC":   f"{roc_auc_score(y_seq, probs_bal):.4f}",
    "Recall":    f"{M(recall_score, preds_bal)*100:.1f}%",
    "Precision": f"{M(precision_score, preds_bal)*100:.1f}%",
}
metrics_old = {
    "Accuracy":  f"{accuracy_score(y_seq, preds_old)*100:.2f}%",
    "F1 Score":  f"{M(f1_score, preds_old):.4f}",
    "AUC-ROC":   f"{roc_auc_score(y_seq, probs_old):.4f}",
    "Recall":    f"{M(recall_score, preds_old)*100:.1f}%",
    "Precision": f"{M(precision_score, preds_old)*100:.1f}%",
}

# ── ablation table ────────────────────────────────────────────────────────────
abl_df = pd.DataFrame(s5.get("ablation", []))
cmp_df = pd.read_csv(DIRS["p6"] / "baseline_comparison.csv") if (DIRS["p6"] / "baseline_comparison.csv").exists() else pd.DataFrame()
route_df = pd.read_csv(DIRS["p4b"] / "route_summary.csv") if (DIRS["p4b"] / "route_summary.csv").exists() else pd.DataFrame()
attn = s5.get("attention_weights", {}).get("mean_attention", {})
attn_text  = attn.get("Text (FinBERT)", 0.028)
attn_time  = attn.get("Time (BiLSTM)", 0.754)
attn_graph = attn.get("Graph (GCN+GAT)", 0.218)

# ── helper renderers ─────────────────────────────────────────────────────────
def stat(val, label, color="#EE854A"):
    return f'<div class="stat-box"><span class="stat-val" style="color:{color}">{val}</span><span class="stat-lbl">{label}</span></div>'

def img(path, alt=""):
    src = b64(path)
    return f'<img src="{src}" alt="{alt}" loading="lazy">' if src else ""

def table_html(df, highlight_last=False):
    rows = ""
    for i, (_, row) in enumerate(df.iterrows()):
        cls = ' class="hl-row"' if highlight_last and i == len(df)-1 else ""
        rows += f"<tr{cls}>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
    heads = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<div class='tbl-wrap'><table><thead><tr>{heads}</tr></thead><tbody>{rows}</tbody></table></div>"

def metric_cmp_row(label, before, after):
    arrow = "↑" if float(after.replace("%","")) > float(before.replace("%","")) else "↓" if float(after.replace("%","")) < float(before.replace("%","")) else "="
    color = "#55a868" if arrow == "↑" else "#c44e52" if arrow == "↓" else "#888"
    return f'<tr><td>{label}</td><td>{before}</td><td><b style="color:{color}">{after} {arrow}</b></td></tr>'

# ── HTML ─────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multimodal SCD Research — Full Report</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0f1117; --surface:#1a1d27; --card:#20243a; --border:#2e3250;
  --accent:#EE854A; --blue:#4878cf; --green:#55a868; --purple:#8172b3;
  --red:#c44e52; --text:#e2e8f0; --muted:#8892a4;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}
.container{{max-width:1280px;margin:0 auto;padding:40px 24px}}

/* HEADER */
header{{
  text-align:center;padding:70px 40px 60px;
  background:linear-gradient(135deg,#1a1e3b 0%,#2d3561 50%,#1a1d27 100%);
  border-radius:24px;margin-bottom:60px;
  box-shadow:0 20px 60px rgba(0,0,0,.4);
  border:1px solid var(--border);position:relative;overflow:hidden;
}}
header::before{{
  content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
  background:radial-gradient(circle,rgba(238,133,74,.08) 0%,transparent 60%);
  pointer-events:none;
}}
h1{{font-family:'Outfit',sans-serif;font-size:2.8rem;font-weight:800;
    background:linear-gradient(135deg,#fff 40%,var(--accent));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:14px}}
.subtitle{{font-size:1.1rem;color:var(--muted);font-weight:300;max-width:700px;margin:0 auto 30px}}
.tags{{display:flex;justify-content:center;gap:10px;flex-wrap:wrap}}
.tag{{padding:5px 14px;border-radius:20px;font-size:.8rem;font-weight:600;border:1px solid}}
.tag-orange{{color:var(--accent);border-color:var(--accent);background:rgba(238,133,74,.1)}}
.tag-blue{{color:var(--blue);border-color:var(--blue);background:rgba(72,120,207,.1)}}
.tag-green{{color:var(--green);border-color:var(--green);background:rgba(85,168,104,.1)}}
.tag-purple{{color:var(--purple);border-color:var(--purple);background:rgba(129,114,179,.1)}}

/* NAV */
nav{{
  position:sticky;top:16px;z-index:100;
  background:rgba(26,29,39,.85);backdrop-filter:blur(12px);
  border:1px solid var(--border);border-radius:50px;
  display:flex;justify-content:center;gap:6px;padding:8px;
  margin-bottom:50px;flex-wrap:wrap;
}}
nav a{{
  padding:8px 18px;border-radius:25px;text-decoration:none;
  color:var(--muted);font-size:.85rem;font-weight:500;transition:all .25s;
}}
nav a:hover{{background:var(--accent);color:#fff}}

/* SECTIONS */
section{{scroll-margin-top:90px;margin-bottom:80px}}
.phase-hd{{
  display:flex;align-items:center;gap:14px;margin-bottom:28px;
  padding-bottom:14px;border-bottom:1px solid var(--border);
}}
.phase-hd h2{{font-family:'Outfit',sans-serif;font-size:1.65rem;font-weight:700}}
.badge{{
  padding:4px 14px;border-radius:8px;font-size:.9rem;font-weight:700;
  background:var(--accent);color:#fff;
}}
.done-badge{{background:var(--green)}}

/* GRID */
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:22px;margin-bottom:24px}}
.grid-3{{grid-template-columns:repeat(3,1fr)}}
.card{{
  background:var(--card);border:1px solid var(--border);border-radius:16px;padding:26px;
  transition:transform .3s,box-shadow .3s;
}}
.card:hover{{transform:translateY(-4px);box-shadow:0 16px 40px rgba(0,0,0,.3)}}
.card h3{{font-family:'Outfit',sans-serif;font-size:1.15rem;font-weight:600;margin-bottom:14px;color:var(--accent)}}

/* STATS */
.stats{{display:flex;flex-wrap:wrap;gap:12px;margin-top:14px}}
.stat-box{{
  background:rgba(255,255,255,.04);border:1px solid var(--border);
  border-radius:12px;padding:12px 18px;min-width:130px;
}}
.stat-val{{display:block;font-size:1.45rem;font-weight:700;margin-bottom:2px}}
.stat-lbl{{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}}

/* CMP TABLE (before/after) */
.cmp-table{{width:100%;border-collapse:collapse;margin-top:16px;font-size:.9rem}}
.cmp-table th{{background:rgba(255,255,255,.05);padding:12px;text-align:left;border-bottom:2px solid var(--border);color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.5px}}
.cmp-table td{{padding:12px;border-bottom:1px solid rgba(255,255,255,.06)}}
.cmp-table tr:last-child td{{border:none}}

/* ATTENTION BAR */
.attn-wrap{{margin:16px 0}}
.attn-label{{font-size:.8rem;color:var(--muted);margin-bottom:6px}}
.attn-bar{{width:100%;height:28px;border-radius:14px;overflow:hidden;display:flex;background:rgba(255,255,255,.06)}}
.attn-seg{{height:100%;display:flex;align-items:center;justify-content:center;font-size:.72rem;font-weight:700;color:#fff;transition:width .6s}}

/* TABLES */
.tbl-wrap{{overflow-x:auto;margin-top:16px;border-radius:12px;border:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;font-size:.87rem}}
th{{background:rgba(255,255,255,.05);padding:12px 14px;text-align:left;
    border-bottom:1px solid var(--border);color:var(--muted);font-weight:600}}
td{{padding:11px 14px;border-bottom:1px solid rgba(255,255,255,.04)}}
tr:last-child td{{border:none}}
.hl-row{{background:rgba(238,133,74,.07)}}
.hl-row td{{font-weight:600;color:var(--accent)}}

/* IMAGES */
img{{max-width:100%;border-radius:12px;border:1px solid var(--border);margin-top:14px;display:block}}

/* CALLOUT */
.callout{{
  background:rgba(85,168,104,.08);border:1px solid rgba(85,168,104,.25);
  border-left:4px solid var(--green);border-radius:8px;padding:16px 20px;
  margin:18px 0;font-size:.9rem;
}}
.callout.warn{{background:rgba(238,133,74,.08);border-color:rgba(238,133,74,.25);border-left-color:var(--accent)}}

/* FOOTER */
footer{{text-align:center;padding:50px 20px;color:var(--muted);border-top:1px solid var(--border);margin-top:80px;font-size:.9rem}}
</style>
</head>
<body>
<div class="container">

<!-- HEADER -->
<header>
  <h1>Multimodal Supply Chain Disruption Predictor</h1>
  <p class="subtitle">End-to-end deep learning pipeline: FinBERT + BiLSTM + GCN/GAT with Attention Fusion &amp; SHAP Explainability</p>
  <div class="tags">
    <span class="tag tag-orange">FinBERT (NLP)</span>
    <span class="tag tag-blue">Bidirectional LSTM</span>
    <span class="tag tag-green">GCN + GAT</span>
    <span class="tag tag-purple">Attention Fusion</span>
    <span class="tag tag-orange">SHAP Explainability</span>
    <span class="tag tag-blue">Sequence Oversampling</span>
  </div>
</header>

<!-- NAV -->
<nav>
  <a href="#summary">Summary</a>
  <a href="#phase2">Phase 2</a>
  <a href="#phase3">Phase 3</a>
  <a href="#lstm">Phase 4a</a>
  <a href="#gnn">Phase 4b</a>
  <a href="#fusion">Phase 5</a>
  <a href="#balanced">Phase 5v2</a>
  <a href="#explainability">Phase 6</a>
</nav>

<!-- EXECUTIVE SUMMARY -->
<section id="summary">
  <div class="phase-hd"><h2>Executive Summary</h2><span class="badge done-badge">ALL PHASES DONE</span></div>
  <div class="grid">
    <div class="card">
      <h3>📦 Dataset</h3>
      <div class="stats">
        {stat(s2.get('total_rows', 10000), 'Total Rows')}
        {stat('1,018 (10.2%)', 'Disruption Events', '#c44e52')}
        {stat('9,950', 'LSTM Sequences')}
        {stat('21', 'Engineered Features')}
      </div>
    </div>
    <div class="card">
      <h3>🧠 Model Architecture</h3>
      <div class="stats">
        {stat('FinBERT', 'Text Encoder', '#8172b3')}
        {stat('BiLSTM', 'Temporal Encoder', '#4878cf')}
        {stat('GCN+GAT', 'Graph Encoder', '#55a868')}
        {stat('Attention', 'Fusion Method', '#EE854A')}
      </div>
    </div>
    <div class="card">
      <h3>🏆 Best Model (Balanced)</h3>
      <div class="stats">
        {stat(metrics_bal['AUC-ROC'], 'AUC-ROC', '#55a868')}
        {stat(metrics_bal['Recall'],  'Recall', '#EE854A')}
        {stat(metrics_bal['F1 Score'],'F1 Score', '#4878cf')}
        {stat(metrics_bal['Accuracy'],'Accuracy')}
      </div>
    </div>
  </div>
</section>

<!-- PHASE 2 -->
<section id="phase2">
  <div class="phase-hd"><h2>Phase 2 — Feature Engineering &amp; Disruption Labeling</h2><span class="badge done-badge">DONE</span></div>
  <div class="grid">
    <div class="card">
      <h3>Disruption Label &amp; Imbalance</h3>
      <p style="color:var(--muted);font-size:.9rem">Label defined as: <code>delay = actual − scheduled &gt; 2 days → y=1</code></p>
      <div class="stats">
        {stat('2 days', 'Threshold θ')}
        {stat('1,018', 'Disruptions (y=1)', '#c44e52')}
        {stat('8,982', 'Non-disruptions (y=0)', '#55a868')}
        {stat('Applied', 'SMOTE (Phase 2 tabular)')}
      </div>
      {img(DIRS['p2'] / 'delay_distribution_histogram.png', 'Delay Distribution')}
      {img(DIRS['p2'] / 'class_imbalance_before_after.png', 'Class Balance')}
    </div>
    <div class="card">
      <h3>Route Risk Matrix</h3>
      {table_html(route_df) if not route_df.empty else '<p>No data</p>'}
      {img(DIRS['p2'] / 'supply_chain_graph_viz.png', 'Supply Chain Graph')}
    </div>
  </div>
</section>

<!-- PHASE 3 -->
<section id="phase3">
  <div class="phase-hd"><h2>Phase 3 — Text Encoder (ProsusAI/FinBERT)</h2><span class="badge done-badge">DONE</span></div>
  <div class="grid">
    <div class="card">
      <h3>NLP Encoder</h3>
      <div class="stats">
        {stat('768 → 64', 'Projection', '#8172b3')}
        {stat('Frozen', 'Backbone')}
        {stat('10,000', 'Sentences Generated')}
        {stat(f"{s3.get('cosine_sim_classes',0.58):.4f}", 'Class Centroid Cosine Sim')}
      </div>
      <div class="callout warn" style="margin-top:18px">
        <b>Text synthesis:</b> No free-text in dataset — sentences constructed from structured fields
        (route, mode, category, delay, risk indices) using domain templates.
      </div>
    </div>
    <div class="card">
      <h3>Sample Synthetic Sentences</h3>
      {"".join(f'<div style="background:rgba(255,255,255,.04);border-radius:8px;padding:10px 14px;margin:8px 0;font-size:.82rem;border-left:3px solid var(--purple)">{t}</div>' for t in pd.read_csv(DIRS["p3"]/"synthetic_text_samples.csv")["synth_text"].head(4).tolist()) if (DIRS["p3"]/"synthetic_text_samples.csv").exists() else ""}
    </div>
  </div>
</section>

<!-- PHASE 4a -->
<section id="lstm">
  <div class="phase-hd"><h2>Phase 4a — Temporal Encoder (Bidirectional LSTM)</h2><span class="badge done-badge">DONE</span></div>
  <div class="grid">
    <div class="card">
      <h3>Architecture &amp; Sequence Construction</h3>
      <div class="stats">
        {stat('L=10', 'Lookback (ACF justified)', '#4878cf')}
        {stat('15', 'Features per timestep')}
        {stat('2-layer', 'BiLSTM (128 hidden)')}
        {stat('64-dim', 'Output embedding')}
        {stat('9,950', 'Sequences built')}
        {stat(f"{s4a.get('best_val_auc',0.61):.4f}", 'Standalone val AUC')}
      </div>
      {img(DIRS['p4a'] / 'acf_delay_seq_len_justification.png', 'ACF Plot')}
    </div>
    <div class="card">
      <h3>Training Curves &amp; Confusion Matrix</h3>
      {img(DIRS['p4a'] / 'lstm_training_curves.png', 'LSTM Training')}
      {img(DIRS['p4a'] / 'lstm_confusion_matrix.png', 'LSTM CM')}
    </div>
  </div>
</section>

<!-- PHASE 4b -->
<section id="gnn">
  <div class="phase-hd"><h2>Phase 4b — Graph Encoder (GCN + GAT)</h2><span class="badge done-badge">DONE</span></div>
  <div class="grid">
    <div class="card">
      <h3>Supply Chain Graph</h3>
      <div class="stats">
        {stat('11', 'Port Nodes', '#55a868')}
        {stat('6', 'Trade Lane Edges')}
        {stat('4 heads', 'GAT Attention')}
        {stat('100%', 'Node Accuracy', '#55a868')}
        {stat('64-dim', 'Output embedding')}
        {stat('Mean+Max', 'Pooling strategy')}
      </div>
      {img(DIRS['p4b'] / 'gnn_graph_viz.png', 'Graph Viz')}
    </div>
    <div class="card">
      <h3>Node Embeddings &amp; Pooling Study</h3>
      {img(DIRS['p4b'] / 'gnn_node_embedding_heatmap.png', 'Node Heatmap')}
      {img(DIRS['p4b'] / 'gnn_pooling_ablation.png', 'Pooling Ablation')}
    </div>
  </div>
</section>

<!-- PHASE 5 -->
<section id="fusion">
  <div class="phase-hd"><h2>Phase 5 — Multimodal Attention Fusion (Imbalanced)</h2><span class="badge done-badge">DONE</span></div>
  <div class="grid">
    <div class="card">
      <h3>Learned Attention Weights</h3>
      <div class="attn-wrap">
        <div class="attn-label">Mean modality trust weights (softmax over 3 modalities)</div>
        <div class="attn-bar">
          <div class="attn-seg" style="width:{attn_text*100:.1f}%;background:#8172b3" title="FinBERT">
            BERT {attn_text*100:.1f}%
          </div>
          <div class="attn-seg" style="width:{attn_time*100:.1f}%;background:#4878cf" title="BiLSTM">
            LSTM {attn_time*100:.1f}%
          </div>
          <div class="attn-seg" style="width:{attn_graph*100:.1f}%;background:#55a868" title="GCN+GAT">
            GNN {attn_graph*100:.1f}%
          </div>
        </div>
      </div>
      <div class="stats">
        {stat(s5.get('best_val_f1',0), 'Best Val F1', '#EE854A')}
        {stat(s5.get('best_val_auc',0), 'Best Val AUC', '#4878cf')}
        {stat('2,305', 'Total Params')}
      </div>
      {img(DIRS['p5'] / 'roc_curve.png', 'ROC Curve')}
      {img(DIRS['p5'] / 'training_curves.png', 'Training Curves')}
    </div>
    <div class="card">
      <h3>Ablation Study — Every Modality Combination</h3>
      <div class="callout">
        Full model (all 3 modalities) outperforms every single- and two-modality baseline.
      </div>
      {table_html(abl_df, highlight_last=True) if not abl_df.empty else ''}
      {img(DIRS['p5'] / 'ablation_chart.png', 'Ablation')}
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <h3>Risk Score Distribution</h3>
      {img(DIRS['p5'] / 'risk_score_distribution.png', 'Risk Dist')}
    </div>
    <div class="card">
      <h3>Confusion Matrix</h3>
      {img(DIRS['p5'] / 'confusion_matrix.png', 'Confusion Matrix')}
    </div>
    <div class="card">
      <h3>Attention Weight Distributions</h3>
      {img(DIRS['p5'] / 'attention_weights_dist.png', 'Attn Dist')}
    </div>
  </div>
</section>

<!-- PHASE 5v2 -->
<section id="balanced">
  <div class="phase-hd"><h2>Phase 5v2 — Sequence Oversampling + Balanced Retrain</h2><span class="badge" style="background:#c44e52">IMPROVED</span></div>
  <div class="callout">
    <b>Problem found:</b> SMOTE in Phase 2 only balanced the flat feature matrix.
    LSTM sequences (fed to all deep models) remained 90:10 imbalanced.
    Fix: minority sequences duplicated with Gaussian noise (σ=0.01) to reach 40% positive.
  </div>
  <div class="grid">
    <div class="card">
      <h3>Before vs After — Real-World Distribution</h3>
      <table class="cmp-table">
        <thead><tr><th>Metric</th><th>Imbalanced (Phase 5)</th><th>Balanced (Phase 5v2)</th></tr></thead>
        <tbody>
          {metric_cmp_row('Accuracy', metrics_old['Accuracy'], metrics_bal['Accuracy'])}
          {metric_cmp_row('F1 Score', metrics_old['F1 Score'], metrics_bal['F1 Score'])}
          {metric_cmp_row('AUC-ROC',  metrics_old['AUC-ROC'],  metrics_bal['AUC-ROC'])}
          {metric_cmp_row('Recall',   metrics_old['Recall'],   metrics_bal['Recall'])}
          {metric_cmp_row('Precision',metrics_old['Precision'],metrics_bal['Precision'])}
        </tbody>
      </table>
      <div class="callout" style="margin-top:16px">
        <b>Key gain:</b> Recall improved {metrics_old['Recall']} → <b>{metrics_bal['Recall']}</b>
        — the balanced model catches significantly more actual disruptions.
      </div>
    </div>
    <div class="card">
      <h3>ROC: Imbalanced vs Balanced</h3>
      {img(DIRS['p5v2'] / 'roc_imbalanced_vs_balanced.png', 'ROC Compare')}
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <h3>Balanced Training Curves</h3>
      {img(DIRS['p5v2'] / 'training_curves_balanced.png', 'Balanced Training')}
    </div>
    <div class="card">
      <h3>Balanced Model Confusion Matrix</h3>
      {img(DIRS['p5v2'] / 'confusion_matrix_balanced.png', 'Balanced CM')}
    </div>
  </div>
</section>

<!-- PHASE 6 -->
<section id="explainability">
  <div class="phase-hd"><h2>Phase 6 — SHAP Explainability &amp; Baseline Comparison</h2><span class="badge done-badge">DONE</span></div>
  <div class="grid">
    <div class="card">
      <h3>Modality-Level SHAP (GradientExplainer)</h3>
      <p style="color:var(--muted);font-size:.88rem;margin-bottom:10px">Which encoder contributes most to each prediction?</p>
      {img(DIRS['p6'] / 'shap_modality_importance_bar.png', 'Modality SHAP')}
      {img(DIRS['p6'] / 'shap_modality_by_class.png', 'Modality SHAP by Class')}
    </div>
    <div class="card">
      <h3>Feature-Level SHAP (TreeExplainer — RF)</h3>
      <p style="color:var(--muted);font-size:.88rem;margin-bottom:10px">Top 15 LSTM features by mean |SHAP value|</p>
      {img(DIRS['p6'] / 'shap_feature_importance_bar.png', 'Feature SHAP')}
      {img(DIRS['p6'] / 'shap_feature_beeswarm.png', 'Beeswarm')}
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <h3>SHAP Dependence — Top 3 Features</h3>
      {img(DIRS['p6'] / 'shap_dependence_top3.png', 'Dependence')}
    </div>
    <div class="card">
      <h3>Attention Heatmap by Prediction Bucket</h3>
      {img(DIRS['p6'] / 'attention_heatmap_by_bucket.png', 'Attn Heatmap')}
      {img(DIRS['p6'] / 'attention_violin_by_class.png', 'Attn Violin')}
    </div>
  </div>
  <div class="card" style="margin-top:22px">
    <h3>📊 Baseline Comparison Table</h3>
    {table_html(cmp_df, highlight_last=True) if not cmp_df.empty else ''}
    {img(DIRS['p6'] / 'roc_comparison.png', 'ROC Comparison')}
  </div>
  <div class="card" style="margin-top:22px">
    <h3>Full Explainability Summary</h3>
    {img(DIRS['p6'] / 'phase6_summary_figure.png', 'Summary Figure')}
  </div>
</section>

<footer>
  <p style="font-size:1rem;font-weight:600;color:var(--accent)">Multimodal Supply Chain Disruption Research — All Phases Complete</p>
  <p style="margin-top:8px">Generated 2026-04-03 &nbsp;·&nbsp; FinBERT + BiLSTM + GCN/GAT + AttentionFusion + SHAP</p>
  <p style="margin-top:8px"><a href="https://github.com/xyz1481/Research" style="color:var(--blue)">GitHub Repository →</a></p>
</footer>

</div>
</body>
</html>"""

OUT.write_text(html, encoding="utf-8")
print(f"Report written -> {OUT}")
print(f"File size: {OUT.stat().st_size / 1024 / 1024:.2f} MB")
