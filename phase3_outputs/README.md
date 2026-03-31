# Phase 3 Outputs — LLM Encoder (FinBERT)

**Script:** `../phase3_finbert_encoder.py`  
**Status:** COMPLETE  
**Date run:** 2026-03-31  
**Runtime:** ~25 minutes (CPU) / ~3 minutes (GPU)

---

## What Phase 3 Does

Implements the **LLM encoder branch** of the multimodal pipeline:

```
Phase 2 enriched dataset (10,000 rows)
    |
    v
[Step 1]  Text synthesis (Option A -- structured -> natural language)
    Template: "A {mode} shipment of {category} goods departed from {origin}
               bound for {dest} via the {route} route. The shipment was
               scheduled for {N} days and was {delayed/on-time}..."
    |
    v
[Step 2]  FinBERT tokenizer  (max_length=128 tokens per sentence)
    |
    v
[Step 3]  FinBERT encoder  (ProsusAI/finbert, FROZEN -- 109.5M params)
    |        extracts [CLS] token from last hidden state
    v
    Raw text embeddings  (10,000 x 768)
    |
    v
[Step 4]  TextProjector MLP  (trained weights saved separately)
    |        Linear(768, 256) -> ReLU -> Dropout(0.2)
    |        -> Linear(256, 64) -> LayerNorm
    v
    Projected text features  (10,000 x 64)  <-- goes into Phase 4 fusion
```

---

## Design Decisions (copy to paper methodology)

### Why FinBERT over plain BERT?
FinBERT (Araci, 2019) is pre-trained on financial text — Bloomberg articles,
earnings calls, financial news — which shares vocabulary and style with
supply-chain risk language (port congestion, tariff escalation, geopolitical
instability). This domain alignment produces better representations than
general-English BERT without any task-specific fine-tuning.

### Why frozen weights?
We have no labeled domain corpus for fine-tuning FinBERT on supply-chain text.
Frozen feature extraction ("zero-shot transfer") is standard practice and is
fully defensible in peer-reviewed ML papers (Devlin et al., 2019; Pan & Yang,
2010). The TextProjector MLP is the only trainable component in this branch;
its weights are optimised end-to-end during Phase 4 multimodal fusion.

### Why Option A (text simulation)?
Several published works synthesize NLP inputs from structured data when real
text is unavailable (e.g., tabular-to-text for financial report generation).
We are transparent about this: the synthetic sentences encode the same semantic
information as the structured columns but in a format that activates FinBERT's
financial-domain representations.

---

## Output Files

| File | Format | Shape | Description |
|------|--------|-------|-------------|
| `text_embeddings_768.npy` | NumPy float32 | 10,000 × 768 | Raw frozen FinBERT CLS vectors |
| `text_features_64d.npy` | NumPy float32 | 10,000 × 64 | **Phase 4 input** — projected embeddings |
| `text_projector_init.pt` | PyTorch | — | MLP projector weights (fine-tuned in Phase 4) |
| `df_phase3_with_text_features.parquet` | Parquet | 10,000 × 115 | Full dataset + 64 text feature cols |
| `synthetic_text_samples.csv` | CSV | 10,000 rows | All generated sentences (for paper transparency) |
| `text_features_sample.csv` | CSV | 10 rows | Sample with all 64 feature values |
| `phase3_summary.json` | JSON | — | Metadata: model, shape, cosine sim, etc. |

---

## Key Numbers for Paper

```
FinBERT model          : ProsusAI/finbert  (Araci, 2019)
Parameters             : 109.5M  (all frozen)
Embedding dim (raw)    : 768
Embedding dim (proj)   : 64
Batch size used        : 32
Max token length       : 128
TextProjector arch     : Linear(768,256) > ReLU > Dropout(0.2) > Linear(256,64) > LayerNorm
Cosine sim (class 0 vs 1 centroids): see phase3_summary.json
```

---

## How to Load in Python

```python
import numpy as np, pandas as pd
import torch

# Raw 768-dim embeddings
emb_768 = np.load("text_embeddings_768.npy")       # (10000, 768)

# Projected 64-dim features (use these in Phase 4)
text_feat = np.load("text_features_64d.npy")        # (10000, 64)

# Full dataset with text features as columns
df = pd.read_parquet("df_phase3_with_text_features.parquet")

# Reload projector (for Phase 4 end-to-end training)
import torch.nn as nn

class TextProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.proj(x)

projector = TextProjector()
projector.load_state_dict(torch.load("text_projector_init.pt"))
```

---

## Next Step

**Phase 4 (LSTM/Transformer):** temporal sequence modelling on shipment delays.  
Input consumed: `df_phase3_with_text_features.parquet`, `text_features_64d.npy`

---

## Citations

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT 2019*
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE, 22(10)*
