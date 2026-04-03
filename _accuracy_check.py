import json, numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score, average_precision_score)

p5 = json.load(open('phase5_fusion_outputs/phase5_summary.json'))
p6 = json.load(open('phase6_explainability/phase6_summary.json'))

probs = np.load('phase5_fusion_outputs/disruption_probabilities.npy')
y     = np.load('phase4a_lstm_outputs/y_seq.npy')
preds = (probs > 0.5).astype(int)

acc = accuracy_score(y, preds)
f1  = f1_score(y, preds, zero_division=0)
auc = roc_auc_score(y, probs)
p   = precision_score(y, preds, zero_division=0)
r   = recall_score(y, preds, zero_division=0)
ap  = average_precision_score(y, probs)

print(f"Total samples        : {len(y)}")
print(f"Disruption events    : {int(y.sum())} ({y.mean()*100:.1f}%)")
print()
print("=== FULL-DATA METRICS (9,950 sequences) ===")
print(f"  Accuracy           : {acc*100:.2f}%")
print(f"  F1 Score           : {f1:.4f}")
print(f"  AUC-ROC            : {auc:.4f}")
print(f"  Precision          : {p:.4f}")
print(f"  Recall             : {r:.4f}")
print(f"  Average Precision  : {ap:.4f}")
print()
print("=== VALIDATION-SET BEST (from training) ===")
print(f"  Best val F1        : {p5['best_val_f1']}")
print(f"  Best val AUC       : {p5['best_val_auc']}")
print()
print("=== BASELINES (Phase 6) ===")
print(f"  Random Forest  F1  : {p6['rf_baseline']['F1']}   AUC: {p6['rf_baseline']['AUC']}")
print(f"  Gradient Boost F1  : {p6['gbm_baseline']['F1']}   AUC: {p6['gbm_baseline']['AUC']}")
print()
print("=== ABLATION (best val F1 / AUC per config) ===")
for m in p5['ablation']:
    marker = " <-- OURS" if m['Model'] == 'Full (all 3)' else ""
    print(f"  {m['Model']:22s}  F1={m['F1']:.4f}  AUC={m['AUC']:.4f}{marker}")
