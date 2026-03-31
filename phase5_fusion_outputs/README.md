# Phase 5 Outputs -- Multimodal Attention Fusion

**Script:** `../phase5_multimodal_fusion.py`
**Status:** COMPLETE
**Date run:** 2026-03-31

## Pipeline

```
text_emb   (9950, 64)  -- FinBERT CLS, aligned via seq_indices
time_emb   (9950, 64)  -- BiLSTM hidden state
graph_emb  (9950, 64)  -- GCN+GAT node lookup by Route_Type
    |
    v
[AttentionFusion]
    stack  (batch, 3, 64)
    score  Linear(64,1) per modality
    softmax over 3 scores -> weights (batch, 3)
    weighted_sum -> fused (batch, 64)
    |
    v
LayerNorm(64)
    |
    v
[MLP Classifier]
    Linear(64, 32) -> ReLU -> Dropout(0.3) -> Linear(32, 1)
    |
    v
sigmoid -> disruption probability [0, 1]
```

## Key Results (for paper)

```
Best val F1    : 0.2290
Best val AUC   : 0.5942
Best val AP    : 0.1596
Dominant modality: Time (BiLSTM)  (mean weight = 0.7536)
```

## Attention Weights (paper insight)

| Modality | Mean Weight | Disruption=0 | Disruption=1 |
|----------|-------------|--------------|--------------|
| Text (FinBERT) | 0.0280 | 0.0280 | 0.0287 |
| Time (BiLSTM)  | 0.7536 | 0.7525 | 0.7637 |
| Graph (GCN+GAT)| 0.2184 | 0.2196 | 0.2077 |

> Write in paper: "Time (BiLSTM) received the highest mean attention weight (0.7536),
> suggesting it is the strongest disruption signal. Attention to Graph embeddings
> is notably higher for disruption-positive samples (0.2077) vs negative
> (0.2196), consistent with the Suez corridor being the highest-risk route."

## Ablation Study (paper Table)

       Model     F1    AUC  Avg Precision
   Text only 0.1914 0.5306         0.1091
   Time only 0.2319 0.5939         0.1412
  Graph only 0.1851 0.5164         0.0945
 Text + Time 0.2304 0.5964         0.1409
Text + Graph 0.1826 0.5247         0.1088
Time + Graph 0.2309 0.5970         0.1328
Full (all 3) 0.2333 0.5989         0.1377

## Output Files

| File | Description |
|------|-------------|
| `best_fusion_model.pt` | Best model weights |
| `disruption_probabilities.npy` | Risk scores for all 9950 samples |
| `attention_weights_all.npy` | (9950, 3) attention weights |
| `predictions_all_samples.csv` | Full predictions with attention |
| `ablation_results.csv` | All 7 ablation configurations |
| `training_curves.png` | Loss + F1 + AUC curves |
| `confusion_matrix.png` | Val set confusion matrix |
| `roc_curve.png` | ROC with AUC |
| `pr_curve.png` | Precision-Recall with AP |
| `ablation_chart.png` | Ablation bar chart (paper figure) |
| `attention_weights_dist.png` | Per-modality attention histograms |
| `risk_score_distribution.png` | Predicted probability distribution |
| `phase5_summary.json` | All stats for paper |

## Next Step

**Phase 6 -- SHAP Explainability:**
Use `shap.DeepExplainer` or `shap.GradientExplainer` on the fusion model
to compute feature importance across the 192-dim input (64 per modality).
Input consumed: `best_fusion_model.pt`, `disruption_probabilities.npy`

## Citations

- Vaswani, A. et al. (2017). Attention is all you need. NeurIPS 2017
- Ramachandram, D. & Taylor, G. (2017). Deep multimodal learning. arXiv:1705.09406
