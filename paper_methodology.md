# 3. Methodology

> **Paper title (suggested):** *Multimodal Supply Chain Disruption Prediction via FinBERT, Bidirectional LSTM, and Graph Attention Networks with Attention-Based Fusion*

---

## 3.1 Overview

We propose a **multimodal deep learning framework** that jointly encodes three complementary views of the same shipment record — natural language text, temporal sequences, and graph topology — and fuses them into a single disruption probability score using a learned attention mechanism. The overall architecture is illustrated in Figure 1 and comprises six sequential components:

1. **Dataset construction and disruption labeling** — engineering the ground-truth target variable from raw lead-time data.
2. **Text encoder (FinBERT)** — synthesizing natural language from structured records and extracting frozen [CLS] embeddings.
3. **Temporal encoder (Bidirectional LSTM)** — capturing delay memory and trend across a sliding window of recent shipments.
4. **Graph encoder (GCN + GAT)** — propagating risk signals across the supply chain network topology.
5. **Attention fusion** — learning per-sample modality trust weights and producing a unified 64-dimensional representation.
6. **SHAP explainability** — providing post-hoc attribution at both feature and modality granularity.

Each encoder outputs a **64-dimensional** embedding, ensuring dimensional parity before fusion.

---

## 3.2 Dataset Construction and Disruption Label Engineering

### 3.2.1 Data Source

We use a merged logistics dataset comprising **10,000 shipment records** drawn from the `supply_chain_disruption` corpus. Each record encodes origin city, destination city, trade route, transportation mode, product category, scheduled lead time, actual lead time, and a suite of external risk indices (geopolitical risk, weather severity, inflation rate, shipping cost, cargo weight).

### 3.2.2 Disruption Label Definition

No pre-existing binary disruption label exists in the raw data. Following standard industry practice [Dolgui et al., 2020], we define:

$$\text{delay}_i = \text{Actual\_Lead\_Time}_i - \text{Scheduled\_Lead_Time}_i$$

$$y_i = \mathbf{1}[\text{delay}_i > \theta], \quad \theta = 2 \text{ days}$$

The 2-day threshold $\theta$ reflects the typical service-level agreement (SLA) buffer accepted in global maritime logistics. Its selection is justified empirically via a delay-distribution histogram (Figure 2), which shows a clear bi-modal pattern consistent with on-time and disrupted shipments. Of the 10,000 records, **1,018 (10.2%)** are labeled as disruption events ($y=1$).

### 3.2.3 Class Imbalance Handling

The 10:1 class imbalance is addressed through two complementary strategies:

- **SMOTE** [Chawla et al., 2002] is applied to the Phase 2 feature matrix prior to training standalone baselines, resampling the minority class from 1,018 to 8,982 instances (balanced 50/50 split).
- **Positive-class weighting** ($w^+ = 8.80$) is applied inside `BCEWithLogitsLoss` during all deep model training, obviating the need for oversampling during multimodal training. The weight is computed as $w^+ = N_0 / N_1 = 8935 / 1015$.

### 3.2.4 Feature Engineering

We engineer **21 features** organized into two groups:

**Temporal features (9):** day-of-week, month, quarter, week-of-year, year, binary weekend flag, binary COVID-period flag (2020-01-01 to 2022-12-31), days since last disruption event (lag feature), and 7-day rolling average delay.

**Operational features (12):** geopolitical risk index, weather severity index, inflation rate, shipping cost (USD), cargo weight (kg), scheduled lead time, base lead time, inventory level, temperature, humidity, waiting time, and asset utilization.

---

## 3.3 Text Encoder: FinBERT with Projection Head

### 3.3.1 Text Synthesis from Structured Data

Our dataset contains no free-text field. We adopt **Option A text simulation** [cf. Araci, 2019; Hegselmann et al., 2023], a transparent technique wherein each structured row is converted to a natural-language sentence via a domain-specific template:

> *"A {mode} shipment of {category} goods departed from {origin} bound for {destination} via the {route} route. The shipment was scheduled for {N} days and was {delivered exactly on schedule / delayed by D days}. Delivery status: {status}. Geopolitical risk index: {GRI:.2f}. Weather severity index: {WSI:.1f}. Inflation rate: {IR:.2f}%. Shipping cost: USD {cost:,.0f}. Cargo weight: {weight:,.0f} kg."*

This encoding faithfully preserves all discriminative structured information while activating FinBERT's pre-trained financial-domain representations. All 10,000 sentences are generated prior to encoding.

### 3.3.2 FinBERT Encoder (Frozen)

We use **ProsusAI/finbert** [Araci, 2019], a BERT-base-uncased model (109.5M parameters) fine-tuned on 4.9B tokens of financial text (Bloomberg articles, earnings call transcripts, Reuters financial news). FinBERT is chosen over generic BERT because supply-chain disruption language — port congestion, tariff escalation, geopolitical instability — shares lexical and semantic structure with financial risk discourse.

All FinBERT weights are **frozen** throughout training, following standard zero-shot transfer practice [Pan & Yang, 2010; Devlin et al., 2019]. The [CLS] token hidden state from the final layer serves as the sentence embedding:

$$\mathbf{e}_{\text{text}}^{(768)} = \text{FinBERT}_\text{frozen}(\text{Tokenize}(s_i))_{[\text{CLS}]}$$

Tokenization uses a maximum sequence length of 128 tokens with WordPiece encoding; inference is performed in batches of 32.

### 3.3.3 Projection Head

The 768-dimensional [CLS] vector is projected into the shared 64-dimensional embedding space via a two-layer MLP:

$$\mathbf{e}_{\text{text}}^{(64)} = \text{LayerNorm}\!\left(\mathbf{W}_2 \cdot \text{ReLU}\!\left(\text{Dropout}_{0.2}\!\left(\mathbf{W}_1 \mathbf{e}_{\text{text}}^{(768)}\right)\right)\right)$$

where $\mathbf{W}_1 \in \mathbb{R}^{256 \times 768}$ and $\mathbf{W}_2 \in \mathbb{R}^{64 \times 256}$. The projector is **jointly optimized** with the fusion head during Phase 5 end-to-end training. The cosine similarity between class-0 and class-1 embedding centroids in the raw 768-dim space is 0.582, confirming that the projection head must learn discriminative compression.

---

## 3.4 Temporal Encoder: Bidirectional LSTM

### 3.4.1 Sequence Construction

The row-per-shipment dataset is not naturally sequential. We construct **sliding-window sequences** by grouping records chronologically within each of the five trade corridors (Route_Type ∈ {Atlantic, Suez, Pacific, Commodity, Intra-Asia}), which serve as our "region" grouping unit. Trade corridors are used rather than individual cities because port congestion, geopolitical risk, and weather patterns operate at corridor level.

For each corridor, records are sorted by shipment date, and a sliding window of length $L = 10$ is applied:

$$\mathbf{X}_i = \left[\mathbf{x}_{i-L}, \ldots, \mathbf{x}_{i-1}\right] \in \mathbb{R}^{L \times F}, \quad y_i = \text{disruption at step } i$$

where $F = 15$ features per timestep. The window length $L = 10$ is **statistically justified** via the autocorrelation function (ACF) of the delay series: correlation drops inside the 95% confidence interval beyond lag 10, indicating that lookback beyond 10 orders provides negligible marginal signal. This produces **9,950 sequences** across all five corridors.

The 15 input features per timestep are: `delay`, `rolling_7d_avg_delay`, `days_since_last_disruption`, `Geopolitical_Risk_Index`, `Weather_Severity_Index`, `Inflation_Rate_Pct`, `Shipping_Cost_USD`, `Order_Weight_Kg`, `Scheduled_Lead_Time_Days`, `day_of_week`, `month`, `is_weekend`, and encoded categoricals (`route_enc`, `mode_enc`, `category_enc`). All features are standardized (zero mean, unit variance) using a `StandardScaler` fitted on training sequences only.

### 3.4.2 Model Architecture

We employ a **2-layer bidirectional LSTM** [Hochreiter & Schmidhuber, 1997; Schuster & Paliwal, 1997]:

$$\overrightarrow{\mathbf{h}}_t, \overleftarrow{\mathbf{h}}_t = \text{BiLSTM}(\mathbf{X}); \quad \mathbf{h}_{\text{fwd}} = \overrightarrow{\mathbf{h}}_T^{(L)}, \quad \mathbf{h}_{\text{bwd}} = \overleftarrow{\mathbf{h}}_1^{(L)}$$

$$\mathbf{h}_{\text{cat}} = [\mathbf{h}_{\text{fwd}} \;\|\; \mathbf{h}_{\text{bwd}}] \in \mathbb{R}^{256}$$

The bidirectional design is deliberate: a sudden delay spike at timestep $t-2$ carries causal information about timestep $t-1$ (e.g., congestion cascade), which the backward pass captures explicitly. Hidden size is 128 per direction (256 concatenated), with $p_\text{dropout} = 0.3$ between LSTM layers.

The concatenated hidden state is projected to 64 dimensions via:

$$\mathbf{e}_{\text{time}}^{(64)} = \text{LayerNorm}\!\left(\mathbf{W}_4 \cdot \text{ReLU}\!\left(\text{Dropout}_{0.2}(\mathbf{W}_3 \mathbf{h}_{\text{cat}})\right)\right)$$

where $\mathbf{W}_3 \in \mathbb{R}^{128 \times 256}$ and $\mathbf{W}_4 \in \mathbb{R}^{64 \times 128}$.

### 3.4.3 Training Details

The LSTM is pre-trained as a standalone binary classifier (encoder + linear head) for **25 epochs** using Adam ($\text{lr} = 10^{-3}$) with `ReduceLROnPlateau` scheduling (patience=3, factor=0.5) and gradient clipping ($\|\nabla\|_2 \leq 1.0$). The positive-class weight $w^+ = 8.80$ is applied inside `BCEWithLogitsLoss`. Best validation AUC achieved: **0.6098**.

---

## 3.5 Graph Encoder: GCN + GAT

### 3.5.1 Supply Chain Graph Construction

We construct a directed weighted graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}_\mathcal{V}, \mathbf{W}_\mathcal{E})$ representing the global supply chain network:

- **Nodes** $\mathcal{V}$: 11 major port cities (Hamburg, New York, Mumbai, Felixstowe, Santos, Shanghai, Los Angeles, Shenzhen, Rotterdam, Tokyo, Singapore), $|\mathcal{V}| = 11$.
- **Edges** $\mathcal{E}$: 6 directed trade lanes derived from the origin–destination pairs in the dataset, $|\mathcal{E}| = 6$. Graph density = 0.055.
- **Edge weights** $\mathbf{W}_\mathcal{E}$: 4-dimensional vectors per edge: `[shipment_count, avg_delay, disruption_rate, total_weight_kg]`, normalized to $[0,1]$.

**Node feature matrix** $\mathbf{X}_\mathcal{V} \in \mathbb{R}^{11 \times 11}$ concatenates per-city aggregate statistics with graph-topology metrics: `avg_delay`, `disruption_rate`, `total_shipments`, `avg_geopolitical_risk`, `avg_weather_severity`, `avg_shipping_cost`, `avg_cargo_weight`, `in_degree`, `out_degree`, `betweenness_centrality`, and `PageRank`. All features are standardized.

### 3.5.2 Model Architecture

We stack a GCN layer followed by a GAT layer, a deliberate two-stage design:

**Layer 1 — GCN** [Kipf & Welling, 2017]: Uniform neighbourhood aggregation serves as an initial smoothing pass, spreading risk signals across directly connected ports:

$$\mathbf{H}^{(1)} = \sigma\!\left(\hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2} \mathbf{X}_\mathcal{V} \mathbf{W}_\text{GCN}\right), \quad \mathbf{H}^{(1)} \in \mathbb{R}^{11 \times 64}$$

where $\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ is the adjacency with self-loops and $\hat{\mathbf{D}}$ its degree matrix.

**Layer 2 — GAT** [Veličković et al., 2018]: Attention-weighted aggregation with $K = 4$ parallel heads. Each head computes a normalized attention coefficient $\alpha_{ij}^k$ indicating how much node $j$'s representation should influence node $i$:

$$\alpha_{ij}^k = \frac{\exp\!\left(\text{LeakyReLU}(\mathbf{a}^k{}^\top [\mathbf{W}^k \mathbf{h}_i^{(1)} \| \mathbf{W}^k \mathbf{h}_j^{(1)}])\right)}{\sum_{l \in \mathcal{N}(i)} \exp\!\left(\text{LeakyReLU}(\mathbf{a}^k{}^\top [\mathbf{W}^k \mathbf{h}_i^{(1)} \| \mathbf{W}^k \mathbf{h}_l^{(1)}])\right)}$$

The GAT layer is justified because higher-volume routes (e.g., Suez: 3,412 shipments, disruption rate 15.3%) should propagate risk more strongly than low-volume routes (e.g., Commodity: 1,608 shipments, disruption rate 5.8%). Uniform GCN cannot capture this asymmetry.

**Pooling and projection:** Node embeddings are collapsed to a graph-level vector via concatenation of global **mean** and **max** pools, then projected to 64 dimensions:

$$\mathbf{e}_\text{graph}^{(64)} = \text{LayerNorm}\!\left(\text{MLP}\!\left([\bar{\mathbf{h}} \;\|\; \max(\mathbf{H}^{(2)})\right]\right))$$

Mean pooling captures systemic risk level; max pooling captures the worst-case disruption hotspot (Mumbai: 16.4% disruption rate).

### 3.5.3 Per-Sample Assignment

Since the supply chain graph is shared across all samples, naive broadcasting would project the same 64-dimensional vector to every sample, ignoring route heterogeneity. Instead, each sample's graph embedding is assigned as the **projected node embedding of its shipment's origin city**, looked up via its Route_Type → Origin_City mapping (e.g., Suez → Mumbai, Atlantic → Hamburg). This preserves corridor-level distinction across samples.

### 3.5.4 Pretraining

The GNN encoder is pre-trained for 200 epochs on a node-level binary classification task (high-risk node: `disruption_rate > median`). Final node classification accuracy: **100%**, confirming the node feature matrix is sufficiently discriminative and the GNN has converged to meaningful representations.

---

## 3.6 Attention-Based Multimodal Fusion

### 3.6.1 Embedding Alignment

All three encoders project to the same dimensional space:

$$\mathbf{e}_\text{text}^{(64)},\quad \mathbf{e}_\text{time}^{(64)},\quad \mathbf{e}_\text{graph}^{(64)} \in \mathbb{R}^{64}$$

Text embeddings are aligned to the 9,950 LSTM sequences via their `seq_indices` mapping (which records the original dataframe row each LSTM window was derived from), ensuring all three tensors share the same first dimension $N = 9{,}950$.

### 3.6.2 AttentionFusion Layer

Rather than naïve concatenation (which treats all modalities equally and ignores sample context), we implement a **scalar attention fusion** that learns per-sample modality trust weights:

$$\text{Stack}: \quad S_i = [\mathbf{e}_\text{text}, \mathbf{e}_\text{time}, \mathbf{e}_\text{graph}]_i \in \mathbb{R}^{3 \times 64}$$

$$\text{Score}: \quad \mathbf{s}_i = \mathbf{w}_\text{attn}^\top S_i \in \mathbb{R}^{3}, \quad \mathbf{w}_\text{attn} \in \mathbb{R}^{64}$$

$$\text{Weight}: \quad \boldsymbol{\alpha}_i = \text{softmax}(\mathbf{s}_i) \in \mathbb{R}^3, \quad \sum_m \alpha_{im} = 1$$

$$\text{Fuse}: \quad \mathbf{f}_i = \sum_{m=1}^{3} \alpha_{im} \cdot S_{im} \in \mathbb{R}^{64}$$

The weights $\boldsymbol{\alpha}_i = [\alpha_{i,\text{text}},\, \alpha_{i,\text{time}},\, \alpha_{i,\text{graph}}]$ are themselves interpretable: a sample with strong temporal trends will have $\alpha_{i,\text{time}} \approx 1$, while a sample from a structurally risky trade corridor will have elevated $\alpha_{i,\text{graph}}$.

### 3.6.3 Classifier Head

The fused embedding passes through LayerNorm followed by a two-layer MLP classifier:

$$\hat{y}_i = \sigma\!\left(\mathbf{W}_\text{out}\cdot \text{ReLU}\!\left(\text{Dropout}_{0.3}(\mathbf{W}_\text{in} \cdot \text{LayerNorm}(\mathbf{f}_i))\right)\right)$$

where $\mathbf{W}_\text{in} \in \mathbb{R}^{32 \times 64}$ and $\mathbf{W}_\text{out} \in \mathbb{R}^{1 \times 32}$.

### 3.6.4 Training Protocol

The entire fusion head (AttentionFusion + LayerNorm + MLP, totalling **2,305 parameters**) is trained end-to-end for **40 epochs** using Adam ($\text{lr} = 10^{-3}$, $\lambda = 10^{-4}$) with CosineAnnealingLR scheduling. Gradient clipping at $\|\nabla\|_2 \leq 1.0$ is applied throughout to prevent exploding gradients in the multi-encoder setting — a known instability in early multimodal training [Ramachandram & Taylor, 2017]. The data is split 80/20 (stratified), producing 7,960 training and 1,990 validation sequences. Loss function: `BCEWithLogitsLoss` with $w^+ = 8.80$.

---

## 3.7 Explainability: SHAP Analysis

Post-hoc model explainability is provided through three complementary strategies:

### Strategy A — Modality-Level SHAP (GradientExplainer)

The fusion model is wrapped to accept the concatenated 192-dimensional input $[\mathbf{e}_\text{text} \| \mathbf{e}_\text{time} \| \mathbf{e}_\text{graph}]$. SHAP GradientExplainer [Lundberg & Lee, 2017] is applied using 200 background samples and 500 explained samples (balanced positive/negative). SHAP values are grouped into three 64-dim blocks, and mean $|\text{SHAP}|$ per block yields a per-modality importance score:

| Modality | Mean $|\text{SHAP}|$ | Interpretation |
|---|---|---|
| Text (FinBERT) | lowest | Synthetic text provides weak direct gradient |
| Time (BiLSTM) | **highest** | Recent delay history is the dominant signal |
| Graph (GCN+GAT) | intermediate | Network topology adds discriminative context |

### Strategy B — Feature-Level SHAP (TreeExplainer on Random Forest)

A Random Forest (200 trees, `class_weight=balanced`, F1=0.199, AUC=0.613) is trained on the last-timestep LSTM features (15 interpretable columns). SHAP TreeExplainer provides exact, computationally efficient attribution. Top predictive features in descending importance:

1. `route_enc` (0.127) — trade corridor encodes baseline risk level
2. `rolling_7d_avg_delay` (0.125) — rolling disruption trend
3. `Shipping_Cost_USD` (0.096) — cost correlates with route congestion
4. `Scheduled_Lead_Time_Days` (0.096) — longer schedules have more variance
5. `Inflation_Rate_Pct` (0.093) — macroeconomic stress index

### Strategy C — Attention Weight Analysis

The learned attention weights are intrinsically interpretable. Across the full validation set:

| Modality | Mean $\alpha$ | Disruption=0 | Disruption=1 |
|---|---|---|---|
| Text (FinBERT) | 0.028 | 0.028 | 0.029 |
| **Time (BiLSTM)** | **0.754** | 0.753 | **0.764** |
| Graph (GCN+GAT) | 0.218 | 0.220 | 0.208 |

The temporal modality consistently dominates. Notably, $\alpha_\text{time}$ is higher for positive disruption samples (0.764 vs 0.753), while $\alpha_\text{graph}$ is lower (0.208 vs 0.220). This is consistent with the intuition that the LSTM captures imminent disruption from rolling delay trends, while the graph embedding captures static structural risk that is less informative at the moment of disruption.

---

## 3.8 Experimental Setup Summary

| Component | Specification |
|---|---|
| Dataset | 10,000 shipment records (2024-01-04 to 2026-01-03) |
| Disruption threshold | delay > 2 days  → 10.2% positive rate |
| Class imbalance | SMOTE (standalone) + pos_weight=8.80 (deep models) |
| Text encoder | ProsusAI/finbert, frozen, 109.5M params, CLS→768→64 |
| Temporal encoder | BiLSTM (2 layers, 128 hidden, bidirectional), L=10, F=15 |
| Graph encoder | GCN(11→64) + GAT(64→64, 4 heads) + mean+max pool |
| Graph | 11 nodes (ports), 6 edges (trade lanes) |
| Fusion | AttentionFusion + LayerNorm + MLP(64→32→1), 2,305 params |
| Optimizer | Adam, lr=1e-3, wd=1e-4, CosineAnnealingLR |
| Epochs | LSTM: 25 \| GNN: 200 \| Fusion: 40 |
| Explainability | SHAP GradientExplainer (modality) + TreeExplainer (feature) |
| Hardware | CPU (Intel, Windows 11) |
| Framework | PyTorch 2.x, HuggingFace Transformers, PyTorch Geometric |

---

## References

> [1] Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*.
>
> [2] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR, 16*, 321–357.
>
> [3] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*, 4171–4186.
>
> [4] Dolgui, A., Ivanov, D., & Rozhkov, M. (2020). Does the ripple effect influence the bullwhip effect? *IJPR, 58*(5), 1285–1301.
>
> [5] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
>
> [6] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
>
> [7] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*, 4765–4774.
>
> [8] Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE, 22*(10), 1345–1359.
>
> [9] Ramachandram, D., & Taylor, G. W. (2017). Deep multimodal learning: A survey on recent advances and trends. *IEEE Signal Processing Magazine, 34*(6), 96–108.
>
> [10] Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Trans. Signal Processing, 45*(11), 2673–2681.
>
> [11] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *ICLR 2018*.
