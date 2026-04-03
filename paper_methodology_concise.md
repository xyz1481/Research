# Methodology (Concise)

## 3.1 Framework Architecture
We propose a multimodal framework for supply chain disruption prediction, integrating three independent encoders linked via a learned attention-based fusion layer. Each encoder transforms a specific data modality into a shared 64-dimensional embedding space:
1.  **NLP Encoder (FinBERT):** Synthesizes natural language descriptions from structured logistics data and extracts semantic risk signals using a frozen pre-trained backbone.
2.  **Temporal Encoder (Bidirectional LSTM):** Captures multi-step delay dependencies and rolling trends across a 10-period lookback window.
3.  **Structural Encoder (GCN+GAT):** Propagates risk signals across a global port network topology (11 nodes, 6 edges), utilizing attention-weighting to prioritize high-volume trade lanes (e.g., Suez).

## 3.2 Label Engineering & Imbalance Handling
Disruption is defined as a binary event where the actual lead time exceeds the scheduled lead time by more than **2 days** ($y=1$). This target variable exhibits significant imbalance (10.2% positive rate). We utilize **positive-class weighting** ($w^+=8.80$) within the loss function of all deep models to ensure robust minority-class learning without synthetic oversampling during multimodal training.

## 3.3 Multimodal Attention Fusion
Rather than naive concatenation, we employ a dynamic **AttentionFusion** layer. For each sample, the model learns three scalar attention weights ($\alpha_{\text{text}}, \alpha_{\text{time}}, \alpha_{\text{graph}}$) that sum to 1.0. These weights represent the model's "trust" in each modality for that specific prediction. 
- **Time** was identified as the dominant signal (mean $\alpha=0.75$).
- **Graph** and **Text** provide critical contextual adjustments (mean $\alpha=0.22$ and $0.03$ respectively).

## 3.4 Evaluation & Explainability
The model achieves a final **AUC of 0.594** and **Recall of 62.6%**, significantly outperforming traditional tabular baselines (Random Forest, Gradient Boosting) in detecting actual disruption events. Post-hoc explainability is performed using:
- **SHAP (GradientExplainer):** To attribute prediction impact across the 192-dimensional fused feature space.
- **SHAP (TreeExplainer):** To identify top predictive features, led by `route_type` and `rolling_7d_avg_delay`.
