# Phase 4a Outputs -- Bidirectional LSTM Temporal Encoder

**Script:** `../phase4a_lstm_encoder.py`
**Status:** COMPLETE
**Date run:** 2026-03-31

## Pipeline

```
Phase 2 enriched dataset
    |
    v
[Step 1]  Categorical encoding
    Route_Type -> route_enc  (0-4)
    Transportation_Mode -> mode_enc  (0-1)
    Product_Category -> category_enc  (0-N)
    |
    v
[Step 2]  ACF plot  -> SEQ_LEN = 10 justified
    |
    v
[Step 3]  Sliding window sequences  (grouped by Route_Type)
    X_seq: (9950, 10, 15)   y_seq: (9950,)
    |
    v
[Step 4]  StandardScaler on features
    |
    v
[Step 5]  Bidirectional LSTM
    hidden=128 x2=256, layers=2, dropout=0.3
    -> concat [h_fwd, h_bwd]   (batch, 256)
    -> Linear(256,128) > ReLU > Dropout(0.2)
    -> Linear(128,64) > LayerNorm
    |
    v
    time_features_64d.npy   shape (9950, 64)  <-- Phase 5 fusion input
```

## Key Numbers (for paper)

```
SEQ_LEN           : 10   (ACF drops within 95% CI at lag ~10)
Total sequences   : 9950
Train / Val split : 80% / 20%
Class 0           : 8935
Class 1           : 1015
pos_weight        : 8.803  (handles class imbalance in loss)
Best val AUC      : 0.6098
Best val loss     : 1.2395
Output dim        : 64  (matches FinBERT projector, GNN projector)
```

## Output Files

| File | Shape | Description |
|------|-------|-------------|
| `time_features_64d.npy` | (9950, 64) | **Phase 5 fusion input** |
| `lstm_encoder.pt` | -- | Frozen encoder weights for fusion |
| `lstm_classifier.pt` | -- | Full classifier (encoder + head) |
| `X_seq.npy` | (9950, 10, 15) | Raw input sequences |
| `y_seq.npy` | (9950,) | Sequence labels |
| `seq_indices.npy` | (9950,) | Maps sequence -> original df row |
| `lstm_feature_scaler.pkl` | -- | StandardScaler for inference |
| `label_encoders.json` | -- | Categorical encoding maps |
| `acf_delay_seq_len_justification.png` | -- | ACF figure for paper |
| `lstm_training_curves.png` | -- | Loss + AUC curves for paper |
| `lstm_confusion_matrix.png` | -- | Confusion matrix for paper |
| `phase4a_summary.json` | -- | All stats for paper |

## How to Load in Python

```python
import numpy as np, torch

time_feat = np.load("time_features_64d.npy")   # (9950, 64)

# Reload encoder for inference
class LSTMEncoder(torch.nn.Module):
    # ... (see phase4a_lstm_encoder.py)
    pass

encoder = LSTMEncoder(input_size=15)
encoder.load_state_dict(torch.load("lstm_encoder.pt"))
encoder.eval()
```

## Next Step

**Phase 4b (GCN/GAT):** graph encoder on the supply-chain graph.
Input produced here: `time_features_64d.npy` (64-dim per sequence)
