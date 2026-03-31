# Phase 4b Outputs -- GNN Encoder (GCN + GAT)

**Script:** `../phase4b_gnn_encoder.py`
**Status:** COMPLETE
**Date run:** 2026-03-31

## Pipeline

```
Phase 2 supply_chain_graph.pkl  +  enriched dataset
    |
    v
[Step 1]  Enrich node features per city (11 nodes)
    avg_delay, disruption_rate, total_shipments,
    avg_geo_risk, avg_weather, avg_cost_usd, avg_weight_kg,
    + topology: in_degree, out_degree, betweenness, pagerank
    -> X_node: (11, 11)
    |
    v
[Step 2]  Edge index (2, 6) + edge_attr (6, 4)
    edge_attr: [shipment_count, avg_delay, disruption_rate, total_weight_kg]
    |
    v
[Step 3]  GCN Layer 1  (uniform neighbourhood aggregation)
    in=11 -> hidden=64  +  BatchNorm + ReLU + Dropout(0.3)
    |
    v
[Step 4]  GAT Layer 2  (4 attention heads)
    hidden=64 -> hidden=64  +  BatchNorm + ReLU + Dropout(0.3)
    |
    +--> [Mean pool] -> (1, 64)  --+
    +--> [Max  pool] -> (1, 64)  --+-> concat -> (1, 128)
                                              |
    v
[Step 5]  Projection MLP
    Linear(128, 128) > ReLU > Dropout(0.2) > Linear(128, 64) > LayerNorm
    -> graph_embedding: (1, 64)
    |
    v
[Step 6]  Lookup by Route_Type -> Origin_City -> node embedding
    -> graph_features_per_sample: (9950, 64)
       aligns sample-by-sample with LSTM time_features_64d.npy
```

## Key Numbers (for paper)

```
Graph nodes                : 11 cities
Graph edges                : 6 trade lanes
Node features              : 11  (7 stats + 4 topology)
Highest disruption route   : Suez (15.3% disruption rate, 3412 shipments)
GCN hidden dim             : 64
GAT heads                  : 4
Pooling                    : mean + max concat -> 128 -> 64
Pretraining node accuracy  : 1.0000
Per-sample graph emb shape : (9950, 64)  (aligns with LSTM output)
```

## Design Decisions (copy to paper)

### Why GCN -> GAT stacking?
GCN (layer 1) performs **uniform** neighbourhood aggregation, ideal for
an initial smoothing pass that spreads risk information across directly
connected ports. GAT (layer 2) performs **attention-weighted** aggregation:
with 4 parallel heads, it learns that the Suez corridor (3,412 shipments)
propagates risk more strongly than the Commodity corridor (1,608 shipments).
This two-layer design is architecturally deliberate (Velickovic et al., 2018).

### Why mean + max pooling?
Mean pooling captures **systemic risk level** (average stress across all
ports). Max pooling captures **worst-case disruption hotspot** (Mumbai, IN:
16.4% disruption rate). Concatenating both gives the fusion head a richer
signal; the ablation plot confirms clear embedding differences between
strategies.

### Per-sample lookup vs broadcast
Rather than broadcasting the same global graph vector to all 9950
samples (which ignores route heterogeneity), we assign each sample the node
embedding of its shipment's origin city, then project it. This means samples
on the Suez corridor receive Mumbai's risk-aware graph embedding, while
Atlantic samples receive Hamburg's — a more semantically faithful assignment.

## Output Files

| File | Shape | Description |
|------|-------|-------------|
| `graph_features_per_sample.npy` | (9950, 64) | **Phase 5 fusion input** |
| `gnn_encoder.pt` | -- | Encoder weights |
| `gnn_node_classifier.pt` | -- | Pretraining classifier |
| `node_embeddings.npy` | (11, 64) | Per-city node embeddings |
| `graph_emb_mean_max.npy` | (64,) | Global graph embedding |
| `city_node_stats.csv` | 11 rows | City stats + embeddings |
| `route_summary.csv` | 5 rows | Per-corridor summary |
| `gnn_graph_viz.png` | -- | Disruption-rate coloured graph |
| `gnn_node_embedding_heatmap.png` | -- | Node embedding heatmap |
| `gnn_pooling_ablation.png` | -- | Mean vs Max vs Mean+Max |
| `gnn_pretraining_curve.png` | -- | Training loss + accuracy |
| `phase4b_summary.json` | -- | All stats for paper |

## Next Step

**Phase 5 (Multimodal Fusion):** concatenate
- `phase3_outputs/text_features_64d.npy`      (FinBERT, 10,000 x 64)
- `phase4a_lstm_outputs/time_features_64d.npy` (LSTM,    9,950  x 64)
- `phase4b_gnn_outputs/graph_features_per_sample.npy` (GNN, 9,950 x 64)
-> fused 192-dim vector -> MLP classifier -> disruption probability

## Citations

- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with GCN. ICLR 2017
- Velickovic, P. et al. (2018). Graph attention networks. ICLR 2018
- Xu, K. et al. (2019). How powerful are GNNs? ICLR 2019
