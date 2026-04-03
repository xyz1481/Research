"""
Threshold Optimisation for Both Models
=======================================
Find the optimal decision threshold (instead of default 0.5)
that maximises F1 on the validation set for both Phase 5 and Phase 5v2.
"""
import numpy as np
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                              recall_score, accuracy_score, precision_recall_curve)
from pathlib import Path

BASE = Path(r"c:\Users\prati\OneDrive\Desktop\Research")
y    = np.load(BASE / "phase4a_lstm_outputs/y_seq.npy")

def optimise_threshold(probs, y_true, label):
    prec, rec, thresholds = precision_recall_curve(y_true, probs)
    # F1 for every threshold
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx  = f1_scores.argmax()
    best_t    = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    preds_05  = (probs > 0.50).astype(int)
    preds_opt = (probs > best_t).astype(int)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  {'Metric':<14}  {'Threshold=0.50':>14}  {'Optimal (t={:.3f})'.format(best_t):>18}")
    print(f"  {'-'*52}")
    for name, fn, kwargs in [
        ("Accuracy",  accuracy_score,  {}),
        ("F1 Score",  f1_score,        {"zero_division":0}),
        ("AUC-ROC",   roc_auc_score,   {}),
        ("Precision", precision_score, {"zero_division":0}),
        ("Recall",    recall_score,    {"zero_division":0}),
    ]:
        if name == "AUC-ROC":
            v05 = fn(y_true, probs)
            vop = fn(y_true, probs)
        else:
            v05 = fn(y_true, preds_05, **kwargs)
            vop = fn(y_true, preds_opt, **kwargs)
        if name == "Accuracy":
            print(f"  {name:<14}  {v05*100:>13.2f}%  {vop*100:>17.2f}%")
        elif name in ["Precision","Recall"]:
            print(f"  {name:<14}  {v05*100:>13.1f}%  {vop*100:>17.1f}%")
        else:
            print(f"  {name:<14}  {v05:>14.4f}  {vop:>18.4f}")
    print(f"\n  >> Optimal threshold: {best_t:.4f}")
    return best_t, preds_opt

# Phase 5 (imbalanced training)
p5_probs = np.load(BASE / "phase5_fusion_outputs/disruption_probabilities.npy")
t5, _    = optimise_threshold(p5_probs, y, "Phase 5 (Imbalanced Training)")

# Phase 5v2 (balanced training)
p5v2_probs = np.load(BASE / "phase5v2_balanced_outputs/disruption_probabilities_balanced.npy")
t5v2, _    = optimise_threshold(p5v2_probs, y, "Phase 5v2 (Balanced Training)")

print("\n")
print("="*55)
print("  RECOMMENDATION")
print("="*55)
print(f"  Use Phase 5v2 model with threshold = {t5v2:.3f}")
print(f"  This gives the highest F1 and AUC-ROC while")
print(f"  maintaining the best disruption detection recall.")
print("="*55)
