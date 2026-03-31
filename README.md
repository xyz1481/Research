# Supply Chain Disruption Prediction — Research Project

> **Multimodal deep learning for supply-chain risk prediction using FinBERT + LSTM + GCN/GAT fusion**

---

## Project Architecture

```
Phase 1  Data collection & merging
Phase 2  Feature engineering & disruption labeling    [DONE]
Phase 3  LLM encoder (FinBERT text embeddings)        [DONE]
Phase 4a Temporal encoder (Bidirectional LSTM)         [DONE]
Phase 4b Graph encoder (GCN / GAT)                    [DONE]
Phase 5  Multimodal fusion + risk prediction          [DONE]
Phase 6  Explainability (SHAP) & evaluation           [DONE]  <- ALL PHASES COMPLETE
```

---

## Repository Layout

```
Research/
|-- README.md                          <- you are here (master overview)
|
|-- merged_logistics_supply_chain.csv  <- Phase 1 merged dataset (11,000 rows)
|
|-- phase2_feature_engineering.py      <- Phase 2 script
|-- phase2_view_results.py             <- Phase 2 HTML report generator
|-- phase2_outputs/
|   |-- README.md                      <- Phase 2 output guide
|   |-- phase2_report.html             <- browser-viewable results
|   |-- viewable/                      <- CSV versions of everything
|   |-- df_phase2_enriched.parquet     <- enriched dataset for Phase 3
|   |-- X_features.npy                 <- feature matrix (10000 x 21)
|   |-- y_labels.npy                   <- disruption labels (10000,)
|   |-- X_resampled.npy                <- SMOTE-balanced X (17964 x 21)
|   |-- y_resampled.npy                <- SMOTE-balanced y (17964,)
|   |-- supply_chain_graph.pkl         <- NetworkX DiGraph object
|   |-- graph_node_features.npy        <- (11, 4) node feature matrix
|   |-- graph_edge_index.npy           <- (2, 6) COO edge index
|   |-- graph_edge_attr.npy            <- (6, 4) edge attributes
|   |-- graph_node_index.json          <- city -> integer mapping
|   |-- phase2_summary.json            <- all key stats (copy to paper)
|   |-- delay_distribution_histogram.png
|   |-- class_imbalance_before_after.png
|   `-- supply_chain_graph_viz.png
|
|-- phase3_finbert_encoder.py          <- Phase 3 script
|-- phase3_outputs/
|   |-- README.md                      <- Phase 3 output guide
|   |-- text_embeddings_768.npy        <- raw CLS embeddings (10000 x 768)
|   |-- text_features_64d.npy          <- projected embeddings (10000 x 64)
|   |-- text_projector_init.pt         <- MLP projector weights
|   |-- df_phase3_with_text_features.parquet
|   |-- synthetic_text_samples.csv     <- all 10,000 generated sentences
|   |-- text_features_sample.csv       <- first 10 rows (inspectable)
|   `-- phase3_summary.json
```

---

## Dataset

| Source | Rows | Key columns |
|--------|------|-------------|
| `supply_chain_disruption` | 10,000 | Origin/Dest city, route, transport mode, lead times, delay, disruption event, risk indices |
| `smart_logistics` | 1,000 | IoT sensor telemetry (GPS, temp, traffic, inventory) |

**Disruption label rule:**  
`disruption = 1` if `Actual_Lead_Time - Scheduled_Lead_Time > 2 days`  
*(Threshold justified by industry SLA buffers; Dolgui et al., 2020)*

---

## Quick-Start: Reproduce All Phases

```bash
# Phase 2 -- Feature engineering + graph + SMOTE
python phase2_feature_engineering.py

# View Phase 2 results in browser
python phase2_view_results.py
# then open: phase2_outputs/phase2_report.html

# Phase 3 -- FinBERT text embeddings  (~25 min on CPU)
python phase3_finbert_encoder.py
```

---

## Key Numbers (for paper)

| Metric | Value |
|--------|-------|
| Total rows | 10,000 |
| Disruption rate (raw) | 10.2% |
| After SMOTE | 50.0% / 50.0% |
| Features engineered | 21 |
| Temporal features | 9 |
| Graph nodes | 11 cities |
| Graph edges | 6 trade lanes |
| FinBERT embedding dim | 768 |
| Projected text dim | 64 |

---

## Citations

- Araci, D. (2019). *FinBERT: Financial sentiment analysis with pre-trained language models.* arXiv:1908.10063
- Devlin, J. et al. (2019). *BERT: Pre-training of deep bidirectional transformers.* NAACL-HLT 2019
- Dolgui, A. et al. (2020). *Ripple effect in the supply chain: An analysis and recent review.* IJPR, 56(1)
- Chawla, N. et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* JAIR, 16, 321–357
