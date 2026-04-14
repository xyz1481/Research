import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import json
from pathlib import Path

# Paths
BASE_DIR       = Path(r"c:\Users\prati\OneDrive\Desktop\Research")
PHASE4A        = BASE_DIR / "phase4a_lstm_outputs"
PHASE5V3_OUT   = BASE_DIR / "phase5v3_improved_outputs"
PHASE6_OUT     = BASE_DIR / "phase6_explainability"

# Load targets
y_seq = np.load(PHASE4A / "y_seq.npy").astype(int)

# Load Trimodal probs
trimodal_probs = np.load(PHASE5V3_OUT / "disruption_probabilities_v3.npy")

# We need to recreate the time-only probs on the full dataset or re-load it if we saved it. 
# Wait, Phase 6 didn't save time-only probs. Let's just recreate it quickly:
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

time_np = np.load(PHASE4A / "time_features_64d.npy")
X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(
    time_np, y_seq, test_size=0.15, stratify=y_seq, random_state=42
)
rf_time = RandomForestClassifier(
    n_estimators=100, max_depth=5, min_samples_leaf=10, 
    class_weight="balanced_subsample", n_jobs=-1, random_state=42
)
rf_time.fit(X_train_t, y_train_t)
time_only_probs = rf_time.predict_proba(time_np)[:, 1]

# Base metrics
base_auc_tri = roc_auc_score(y_seq, trimodal_probs)
base_auc_time = roc_auc_score(y_seq, time_only_probs)
print(f"Original AUC - Trimodal : {base_auc_tri:.4f}")
print(f"Original AUC - Time-Only: {base_auc_time:.4f}")

# Bootstrap testing
n_bootstraps = 1000
rng = np.random.default_rng(42)

diffs = []
tri_aucs = []
time_aucs = []

print("\nRunning Bootstrap test with 1000 iterations...")
for i in range(n_bootstraps):
    # Sample indices with replacement
    indices = rng.choice(len(y_seq), size=len(y_seq), replace=True)
    
    y_true_b = y_seq[indices]
    
    # Must have both classes in bootstrap sample
    if len(np.unique(y_true_b)) < 2:
        continue
        
    y_pred_tri_b = trimodal_probs[indices]
    y_pred_time_b = time_only_probs[indices]
    
    auc_tri = roc_auc_score(y_true_b, y_pred_tri_b)
    auc_time = roc_auc_score(y_true_b, y_pred_time_b)
    
    tri_aucs.append(auc_tri)
    time_aucs.append(auc_time)
    diffs.append(auc_tri - auc_time)

# Confidence intervals
sorted_diffs = np.sort(diffs)
ci_lower = sorted_diffs[int(0.025 * len(sorted_diffs))]
ci_upper = sorted_diffs[int(0.975 * len(sorted_diffs))]

# p-value: what % of time is Trimodal <= Time-Only?
p_value = np.mean(np.array(diffs) <= 0)

print(f"\n--- Statistical Significance Test Results ---")
print(f"Trimodal 95% CI: [{np.percentile(tri_aucs, 2.5):.4f}, {np.percentile(tri_aucs, 97.5):.4f}]")
print(f"Time-Only 95% CI: [{np.percentile(time_aucs, 2.5):.4f}, {np.percentile(time_aucs, 97.5):.4f}]")
print(f"Average Improvement: +{np.mean(diffs):.4f}")
print(f"95% Confidence Interval of Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"P-value (H0: Trimodal <= Time-Only): {p_value:.6f}")

if p_value < 0.05:
    print("\nCONCLUSION: YES, the Trimodal model beats the Time-Only baseline by a statistically meaningful margin (p < 0.05).")
else:
    print("\nCONCLUSION: NO, the difference is not statistically significant.")
